"""Бенчмарк pbcchecker: большой RVE, разложение по стадиям.

Запуск:  python tools/bench.py [n_nodes n_eq n_steps]
По умолчанию 100k узлов, 40k уравнений, 5 шагов (~500k строк узлов в .frd).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from pbcchecker.frd_disp import iter_disp_blocks, read_ccx_version
from pbcchecker.inp_equations import parse_equations
from pbcchecker.residual import compute_residuals
from pbcchecker import check_job

TMP = ROOT / "tools" / "_bench"
TMP.mkdir(exist_ok=True)


def make_big(n_nodes: int, n_eq: int, n_steps: int):
    # параметры в имени файла: смена аргументов не измерит молча старый кэш
    tag = f"{n_nodes // 1000}k_{n_eq // 1000}k_{n_steps}"
    inp, frd = TMP / f"bench_{tag}.inp", TMP / f"bench_{tag}.frd"
    half = n_nodes // 2
    if not inp.exists():
        with inp.open("w", encoding="utf-8") as f:
            f.write("*NODE, NSET=ALL\n")
            f.writelines(f"{i}, 0.0, 0.0, 0.0\n" for i in range(1, n_nodes + 1))
            for j in range(1, n_eq + 1):
                f.write("*EQUATION\n2\n")
                f.write(f"{j}, 1, 1.0, {j + half}, 1, -1.0\n")
    if not frd.exists():
        w = frd.open("w", encoding="utf-8")
        w.write("    1C\n    1UVERSION           Version 2.20\n    1UPGM               CalculiX\n")
        for s in range(1, n_steps + 1):
            w.write(f"  100CL  {s}  0.{s}000000E-02\n")
            w.write("    -4  DISP        4    1\n    -5  D1          1    2    3\n")
            for i in range(1, n_nodes + 1):
                ux = 1.0e-04 * (1 + (i - 1) % half % 7)   # одинаково в паре (i, i+half)
                w.write(" -1%10d%12.5E%12.5E%12.5E\n" % (i, ux * s, 0.0, 0.0))
            w.write("    -3\n")
        w.close()
    for stale in TMP.glob("pbc_report_*"):     # прошлые прогоны не пухят каталог
        stale.unlink()
    return inp, frd


def stage(name, fn, reps=1):
    """Замер стадии: min по reps повторам (лучший прогон = минимум шума ОС)."""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    print(f"{name:34s} {best:8.3f} c", flush=True)
    return out


def main():
    n_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    n_eq = int(sys.argv[2]) if len(sys.argv) > 2 else 40_000
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    inp, frd = make_big(n_nodes, n_eq, n_steps)
    print(f"узлов {n_nodes}, уравнений {n_eq}, шагов {n_steps}; frd = {frd}")
    stage("make fixture (cache)", lambda: make_big(n_nodes, n_eq, n_steps))

    equations = stage("parse_equations", lambda: parse_equations(inp), reps=3)
    blocks = stage("iter_disp_blocks (list)", lambda: list(iter_disp_blocks(frd)), reps=3)
    stage("read_ccx_version", lambda: read_ccx_version(frd), reps=3)
    stage("compute_residuals", lambda: compute_residuals(equations, blocks), reps=5)
    stage("check_job (без отчёта)", lambda: check_job(inp, frd), reps=5)
    stage("check_job (--out-dir)", lambda: check_job(inp, frd, out_dir=TMP), reps=3)


if __name__ == "__main__":
    main()
