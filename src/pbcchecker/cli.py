"""CLI: pbc-check job.inp [--frd job.frd] [--tol 1e-3] [--out-dir DIR]

Коды выхода: 0 — PASS/N/A, 1 — FAIL, 2 — любая ошибка (нет .frd / нет DISP /
битые уравнения / узел без DISP / неожиданный сбой) — чтобы CI не путал
crash с FAIL.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .gate import check_job


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="pbc-check",
        description="Проверка PBC-невязок *EQUATION по DISP из .frd (CalculiX)")
    ap.add_argument("inp", help="job .inp с *EQUATION")
    ap.add_argument("--frd", default=None, help=".frd с блоками DISP (по умолчанию <inp>.frd)")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="порог относительной невязки, > 0 (по умолчанию 1e-3)")
    ap.add_argument("--out-dir", default=None,
                    help="каталог для pbc_report_<job>_<дата>.{json,md} (audit_trace)")
    ap.add_argument("--version", action="version", version=f"pbcchecker {__version__}")
    args = ap.parse_args(argv)
    if not (args.tol > 0):   # отсекает и <=0, и nan
        ap.error(f"--tol должен быть > 0 (получено {args.tol!r})")

    try:
        v = check_job(Path(args.inp), Path(args.frd) if args.frd else None,
                      tol=args.tol, out_dir=Path(args.out_dir) if args.out_dir else None)
    except Exception as e:   # любой сбой — код 2, не неотличимый от FAIL traceback
        print(f"ОШИБКА ДАННЫХ: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    if v.verdict == "N/A":
        print("N/A: в deck нет *EQUATION — проверка PBC неприменима.")
        return 0
    status = "PASS" if v.passed else "FAIL"
    print(f"{status}: max относительная невязка {v.max_rel_residual:.3e} "
          f"(порог {args.tol:g}, шаг «{v.worst_step}», уравнение {v.worst_eq}, "
          f"всего уравнений {v.n_equations})")
    for p in v.report_paths:
        print(f"отчёт: {p}")
    return 0 if v.passed else 1


if __name__ == "__main__":
    sys.exit(main())
