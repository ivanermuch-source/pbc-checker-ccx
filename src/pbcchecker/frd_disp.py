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
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np


@dataclass(frozen=True)
class DispBlock:
    step: int          # номер шага из строки 100CL; 0 если не найден
    label: str         # "step 1 (time 0.1000000E-01)" или "block 3"
    nodes: np.ndarray  # int32 (n,) — возрастающие по файлу, без сортировки
    u: np.ndarray      # float64 (n, 3) — ux, uy, uz


def read_ccx_version(frd_path: Path) -> str:
    """Строка версии из заголовка '1UVERSION', например 'Version 2.20'."""
    with Path(frd_path).open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "1UVERSION" in line:
                return line.split("1UVERSION", 1)[1].strip() or "unknown"
            if line.startswith("-4") or line.lstrip().startswith("-4"):
                break  # заголовок кончился
    return "unknown"


def iter_disp_blocks(frd_path: Path) -> Iterator[DispBlock]:
    """Все блоки DISP по порядку. Память O(узлов блока), файл читается потоково."""
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

    with Path(frd_path).open(encoding="utf-8", errors="ignore") as f:
        for line in f:
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
                    cur_nodes, cur_vals = [], []
                    in_disp = True
                elif in_disp and name != "DISP":
                    # другой датасет начался без завершения DISP (нестандарт) — закрыть
                    yield emit()
                continue
            if in_disp and line.lstrip().startswith("-3"):
                yield emit()
                continue
            if in_disp and line.lstrip().startswith("-1"):
                try:
                    node = int(line[3:13])
                    vals = tuple(float(line[13 + 12 * k:25 + 12 * k]) for k in range(3))
                except ValueError:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    node = int(parts[1])
                    vals = tuple(float(v) for v in parts[2:5])
                cur_nodes.append(node)
                cur_vals.append(vals)  # type: ignore[arg-type]
        if in_disp and cur_nodes:
            yield emit()
