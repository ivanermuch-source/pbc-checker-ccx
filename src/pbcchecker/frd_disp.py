"""Потоковое чтение блоков DISP из .frd CalculiX (ASCII, ccx >= 2.8).

Формат блока (проверен на боевых прогонах ccx 2.20, АВТОсбор runs/*/specimen.frd):

      1PSTEP                          2
     100CL  1  0.1000000E-01
     -4  DISP        4    1
     -5  D1 ...
     -1        93  0.00000E+00 0.00000E+00-1.30000E-03
     -3

Строка узла: " -1", номер I10 (колонки 3:13), значения 3 x E12.5
(колонки 13+12k : 25+12k). При переполнении формата поля слипаются —
тогда откат на split() (тот же приём, что в stellaraster/homogenization/ccx_dat.py).
DISPI (мнимая часть в частотных задачах) пропускается.
Битая структура (незакрытый '-3', пустой блок, нечитаемая строка узла) —
ValueError с номером строки, а не тихая потеря данных.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

from .inp_equations import to_float


@dataclass(frozen=True)
class DispBlock:
    step: int          # номер шага из строки 100CL; 0 если не найден
    label: str         # "step 1 (time 0.1000000E-01)" или "block 3"
    nodes: np.ndarray  # int32 (n,) — возрастающие по файлу, без сортировки
    u: np.ndarray      # float64 (n, 3) — ux, uy, uz


def _sniff_binary(frd_path: Path, probe: int = 1 << 16) -> None:
    """Бинарный .frd (переименованный .fil, output=freq/binary) не содержит
    '-4  DISP' — без детекта получили бы ложный совет про *NODE FILE."""
    with Path(frd_path).open("rb") as f:
        if b"\x00" in f.read(probe):
            raise ValueError(f"{frd_path}: файл содержит нулевые байты — похоже на "
                             "бинарный .frd/.fil; инструмент читает только ASCII .frd")


def read_ccx_version(frd_path: Path) -> str:
    """Строка версии из заголовка '1UVERSION', например 'Version 2.20'."""
    with Path(frd_path).open(encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            if "1UVERSION" in line:
                return line.split("1UVERSION", 1)[1].strip() or "unknown"
            st = line.lstrip()
            if st.startswith("-4") or st.startswith("100CL"):
                break  # заголовок кончился
    return "unknown"


def iter_disp_blocks(frd_path: Path) -> Iterator[DispBlock]:
    """Все блоки DISP по порядку. Память O(узлов блока), файл читается потоково."""
    _sniff_binary(frd_path)
    cur_nodes: List[int] = []
    cur_vals: List[Tuple[float, float, float]] = []
    in_disp = False
    step = 0
    time_s = ""
    block_idx = 0

    def emit():
        nonlocal cur_nodes, cur_vals, in_disp, block_idx
        label = (f"step {step} (time {time_s})" if time_s else f"block {block_idx}")
        blk = DispBlock(step, label,
                        np.asarray(cur_nodes, dtype=np.int32),
                        np.asarray(cur_vals, dtype=np.float64).reshape(-1, 3))
        cur_nodes, cur_vals = [], []
        in_disp = False
        block_idx += 1
        return blk

    with Path(frd_path).open(encoding="utf-8-sig", errors="ignore") as f:
        for lineno, line in enumerate(f, 1):
            if line.lstrip().startswith("100CL"):
                parts = line.split()
                try:
                    step = int(parts[1])
                    time_s = parts[2] if len(parts) > 2 else ""
                except (IndexError, ValueError):
                    pass
                continue
            if line.lstrip().startswith("-4"):
                head = line.split()
                name = head[1] if len(head) > 1 else ""
                if name == "DISP":
                    if in_disp and cur_nodes:
                        raise ValueError(f"{frd_path}: строка {lineno}: блок DISP "
                                         f"(step {step}) не закрыт '-3'")
                    cur_nodes, cur_vals = [], []
                    in_disp = True
                elif in_disp:
                    # другой датасет начался без завершения DISP (нестандарт) — закрыть
                    if cur_nodes:
                        yield emit()
                    else:
                        in_disp = False
                continue
            if in_disp and line.lstrip().startswith("-3"):
                if not cur_nodes:
                    raise ValueError(f"{frd_path}: строка {lineno}: пустой блок DISP "
                                     "(нет строк узлов '-1')")
                yield emit()
                continue
            if in_disp and line.lstrip().startswith("-1"):
                try:
                    node = int(line[3:13])
                    vals = tuple(float(line[13 + 12 * k:25 + 12 * k]) for k in range(3))
                except ValueError:
                    parts = line.split()
                    if len(parts) < 5:
                        raise ValueError(f"{frd_path}: строка {lineno}: нечитаемая строка "
                                         f"узла DISP: {line.rstrip()!r}") from None
                    try:
                        node = int(parts[1])
                        vals = tuple(to_float(v) for v in parts[2:5])
                    except ValueError:
                        raise ValueError(f"{frd_path}: строка {lineno}: нечитаемая строка "
                                         f"узла DISP: {line.rstrip()!r}") from None
                cur_nodes.append(node)
                cur_vals.append(vals)  # type: ignore[arg-type]
        if in_disp and cur_nodes:
            yield emit()  # EOF вместо '-3' — данные уже собраны, не теряем
