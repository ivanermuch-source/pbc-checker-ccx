"""Потоковое чтение блоков DISP из .frd CalculiX (ASCII, ccx >= 2.8).

Формат блока (проверен на боевых прогонах ccx 2.20):

      1PSTEP                          2
     100CL  1  0.1000000E-01
     -4  DISP        4    1
     -5  D1 ...
     -1        93  0.00000E+00 0.00000E+00-1.30000E-03
     -3

Строка узла: " -1", номер I10 (колонки 3:13), значения 3 x E12.5
(колонки 13+12k : 25+12k), итого ровно 49 символов + перевод строки.
Однородный блок парсится векторно (uint8-матрица -> числа, без питоновского
цикла по строкам); нестандартные строки — построчный разбор со split()-fallback.
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

# ширина стандартной строки узла: " -1" + I10 + 3 x E12.5 + "\n"
_NODE_LINE_LEN = 50
_P10_NODE = 10 ** np.arange(9, -1, -1, dtype=np.int64)     # веса I10 справа налево
# точный разбор E12.5 (побитово как float()): при 0<=k<=13 mant*5^k — точное
# целое < 2^53, домножение на 2^k через ldexp точно; при -22<=k<0 деление
# mant/10^|k| — оба операнда точны, одно округление = округлению strtod.
# Вне k in [-22, 13] точной схемы нет — блок уходит в построчный fallback.
_P5I = 5 ** np.arange(0, 14, dtype=np.int64)
_P10F = 10.0 ** np.arange(0, 23, dtype=np.float64)   # 10^j точно при j <= 22
_K_MIN, _K_MAX = -22, 13


@dataclass(frozen=True)
class DispBlock:
    """Один блок DISP: узлы и перемещения одного шага из .frd."""
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


def _digits(a: np.ndarray) -> np.ndarray:
    """uint8-колонки -> int64-цифры; не-цифры дают значения вне 0..9."""
    return a.astype(np.int64) - 48


def _e12_5_to_float(f: np.ndarray) -> np.ndarray:
    """Колонки 12 символов E12.5 (uint8) -> float64, побитово как float().

    ValueError — если поле не E12.5 или экспонента вне точного диапазона
    (тогда вызывающий уходит в построчный fallback)."""
    d = _digits(f[:, [1, 3, 4, 5, 6, 7]])
    if not ((d >= 0) & (d <= 9)).all():
        raise ValueError("нецифровая мантисса")
    if not (f[:, 2] == 46).all() or not (f[:, 8] == 69).all():       # '.' и 'E'
        raise ValueError("не E12.5")
    ed = _digits(f[:, 10:12])
    if not ((ed >= 0) & (ed <= 9)).all():
        raise ValueError("нецифровая экспонента")
    if not ((f[:, 0] == 32) | (f[:, 0] == 45)).all() or \
       not ((f[:, 9] == 43) | (f[:, 9] == 45)).all():
        raise ValueError("неожиданный знак")
    mant = d @ np.array([100000, 10000, 1000, 100, 10, 1], dtype=np.int64)
    e = ed @ np.array([10, 1], dtype=np.int64)
    e = np.where(f[:, 9] == 45, -e, e)               # знак экспоненты
    k = e - 5
    if k.min() < _K_MIN or k.max() > _K_MAX:
        raise ValueError("экспонента вне точного диапазона ldexp")
    pos = k >= 0
    # clip: where вычисляет обе ветви, индексы должны быть корректны для всех k
    vp = np.ldexp((mant * _P5I[np.clip(k, 0, 13)]).astype(np.float64),
                  np.clip(k, 0, None).astype(np.int32))         # x2^k только при k>=0
    vn = mant.astype(np.float64) / _P10F[np.clip(-k, 0, 22)]
    val = np.where(pos, vp, vn) * np.where(f[:, 0] == 45, -1.0, 1.0)
    return val


def _parse_uniform_block(raw: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Векторный разбор блока одинаковых 50-символьных строк узлов.

    ValueError — если блок не однородно-стандартный (не ASCII, не цифры
    в ожидаемых колонках): вызывающий откатывается на построчный разбор."""
    text = "".join(raw).encode("ascii")            # UnicodeEncodeError -> fallback
    b = np.frombuffer(text, dtype=np.uint8).reshape(len(raw), _NODE_LINE_LEN)

    nd = _digits(b[:, 3:13])                       # номер узла I10
    if not (((nd >= 0) & (nd <= 9)) | (b[:, 3:13] == 32)).all():
        raise ValueError("нецифровой номер узла")
    nodes = nd.clip(0, 9) @ _P10_NODE

    cols = np.empty((len(raw), 3), dtype=np.float64)
    for k in range(3):
        cols[:, k] = _e12_5_to_float(b[:, 13 + 12 * k:25 + 12 * k])
    return nodes.astype(np.int32), cols


def iter_disp_blocks(frd_path: Path) -> Iterator[DispBlock]:
    """Все блоки DISP по порядку. Память O(узлов блока), файл читается потоково.

    Горячий путь — строки узлов ' -1' (99% строк): копятся в буфер и парсятся
    векторно на закрытии блока; редкие строки '-3/-4/100CL' — во второй очереди."""
    _sniff_binary(frd_path)
    cur_nodes: List[int] = []      # нестандартные строки узлов (построчно)
    cur_vals: List[Tuple[float, float, float]] = []
    append_n = cur_nodes.append
    append_v = cur_vals.append
    raw: List[str] = []            # стандартные 50-символьные строки узлов
    raw_ln: List[int] = []         # их номера строк — для ошибок fallback
    raw_append = raw.append
    raw_ln_append = raw_ln.append
    in_disp = False
    step = 0
    time_s = ""
    block_idx = 0

    def fallback_lines() -> None:
        """Построчный разбор буфера raw (нестандартный блок), ошибки с номером строки."""
        for ln, line in zip(raw_ln, raw):
            try:
                append_n(int(line[3:13]))
                append_v((float(line[13:25]), float(line[25:37]),
                          float(line[37:49])))
            except ValueError:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        append_n(int(parts[1]))
                        append_v(tuple(to_float(v) for v in parts[2:5]))
                        continue
                    except ValueError:
                        pass
                raise ValueError(f"{frd_path}: строка {ln}: нечитаемая строка "
                                 f"узла DISP: {line.rstrip()!r}") from None

    def emit():
        """Собрать DispBlock из буферов и очистить их."""
        nonlocal in_disp, block_idx
        label = (f"step {step} (time {time_s})" if time_s else f"block {block_idx}")
        if raw and not cur_nodes:                  # однородный блок — векторно
            try:
                nodes, u = _parse_uniform_block(raw)
                blk = DispBlock(step, label, nodes, u)
            except (ValueError, UnicodeEncodeError):
                fallback_lines()
                blk = DispBlock(step, label,
                                np.asarray(cur_nodes, dtype=np.int32),
                                np.asarray(cur_vals, dtype=np.float64).reshape(-1, 3))
        else:
            if raw:                                # смешанный блок — построчно
                fallback_lines()
            blk = DispBlock(step, label,
                            np.asarray(cur_nodes, dtype=np.int32),
                            np.asarray(cur_vals, dtype=np.float64).reshape(-1, 3))
        cur_nodes.clear()
        cur_vals.clear()
        raw.clear()
        raw_ln.clear()
        in_disp = False
        block_idx += 1
        return blk

    def on_100cl(line: str) -> None:
        """Разбор строки '100CL <шаг> <время>'; битая — молча пропускается."""
        nonlocal step, time_s
        parts = line.split()
        try:
            step = int(parts[1])
            time_s = parts[2] if len(parts) > 2 else ""
        except (IndexError, ValueError):
            pass

    with Path(frd_path).open(encoding="utf-8-sig", errors="ignore") as f:
        for lineno, line in enumerate(f, 1):
            if in_disp:
                if line.startswith(" -1"):          # стандартная строка узла
                    if len(line) == _NODE_LINE_LEN:
                        raw_append(line)
                        raw_ln_append(lineno)
                        continue
                    try:                            # нестандартная длина — редко
                        node = int(line[3:13])
                        vals = (float(line[13:25]), float(line[25:37]),
                                float(line[37:49]))
                    except ValueError:
                        parts = line.split()
                        if len(parts) < 5:
                            raise ValueError(
                                f"{frd_path}: строка {lineno}: нечитаемая строка "
                                f"узла DISP: {line.rstrip()!r}") from None
                        try:
                            node = int(parts[1])
                            vals = tuple(to_float(v) for v in parts[2:5])
                        except ValueError:
                            raise ValueError(
                                f"{frd_path}: строка {lineno}: нечитаемая строка "
                                f"узла DISP: {line.rstrip()!r}") from None
                    append_n(node)
                    append_v(vals)
                    continue
                st = line.lstrip()                  # редкие строки блока
                if st.startswith("-3"):
                    if not cur_nodes and not raw:
                        raise ValueError(f"{frd_path}: строка {lineno}: пустой блок "
                                         "DISP (нет строк узлов '-1')")
                    yield emit()
                    continue
                if st.startswith("-1"):             # нестандартный отступ — через split
                    parts = line.split()
                    if len(parts) < 5:
                        raise ValueError(f"{frd_path}: строка {lineno}: нечитаемая строка "
                                         f"узла DISP: {line.rstrip()!r}")
                    try:
                        node = int(parts[1])
                        vals = tuple(to_float(v) for v in parts[2:5])
                    except ValueError:
                        raise ValueError(f"{frd_path}: строка {lineno}: нечитаемая строка "
                                         f"узла DISP: {line.rstrip()!r}") from None
                    append_n(node)
                    append_v(vals)
                    continue
                if not st.startswith("-4"):
                    if st.startswith("100CL"):
                        on_100cl(line)
                    continue                        # '-5', PSTEP и прочее — мимо
                head = line.split()
                name = head[1] if len(head) > 1 else ""
                if name == "DISP":
                    if cur_nodes or raw:
                        raise ValueError(f"{frd_path}: строка {lineno}: блок DISP "
                                         f"(step {step}) не закрыт '-3'")
                    in_disp = True
                # другой датасет без завершения DISP (нестандарт) — закрыть
                elif cur_nodes or raw:
                    yield emit()
                else:
                    in_disp = False
                continue
            st = line.lstrip()                      # вне блока: заголовок/шаг
            if st.startswith("100CL"):
                on_100cl(line)
            elif st.startswith("-4") and line.split()[1:2] == ["DISP"]:
                in_disp = True
        if in_disp and (cur_nodes or raw):
            yield emit()  # EOF вместо '-3' — данные уже собраны, не теряем
