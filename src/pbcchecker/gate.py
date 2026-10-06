"""Гейт check_job(): верхний API для CLI и для вызова из внешних пайплайнов (за флагом-выключателем)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .frd_disp import iter_disp_blocks, read_ccx_version
from .inp_equations import Equation, parse_equations
from .report import GateOutcome, build_report, write_reports
from .residual import ResidualResult, compute_residuals


@dataclass(frozen=True)
class Verdict:
    """Итог check_job для CLI и внешних пайплайнов."""
    passed: Optional[bool]           # None = N/A (нет уравнений)
    verdict: str                     # PASS | FAIL | N/A
    max_rel_residual: float
    worst_step: str
    worst_eq: int
    n_equations: int
    result: Optional[ResidualResult]
    report_paths: List[Path]
    worst_equation: str = field(default="")   # render худшего уравнения (для CLI)


def check_job(inp: Path, frd: Optional[Path] = None, tol: float = 1e-3,
              out_dir: Optional[Path] = None) -> Verdict:
    """Проверить PBC-невязки прогона. frd по умолчанию = inp с суффиксом .frd.

    Семантика: PASS — max относительная невязка ≤ tol на всех шагах;
    FAIL — выше tol хотя бы на одном; N/A — в deck нет *EQUATION (прогон без PBC).
    N/A тоже пишет отчёт при заданном out_dir — audit_trace без дыр.
    """
    inp = Path(inp)
    frd = Path(frd) if frd else inp.with_suffix(".frd")
    equations = parse_equations(inp)
    frd_exists = frd.exists()
    if not equations:
        paths: List[Path] = []
        if out_dir is not None:   # sha256/отчёт — только когда нужен audit_trace
            outcome = GateOutcome("N/A", tol, 0.0, "", -1, "")
            data = build_report(inp, frd if frd_exists else None, None, outcome,
                                read_ccx_version(frd) if frd_exists else "unknown", [])
            paths = write_reports(data, Path(out_dir), inp.stem)
        return Verdict(None, "N/A", 0.0, "", -1, 0, None, paths)
    if not frd_exists:
        raise FileNotFoundError(f"{frd}: нет файла результатов (.frd). "
                                f"Добавьте в deck '*NODE FILE' + 'U' и перезапустите ccx.")

    blocks = list(iter_disp_blocks(frd))
    result = compute_residuals(equations, blocks)
    worst = max(result.steps, key=lambda s: s.max_rel_residual)
    passed = worst.max_rel_residual <= tol
    worst_idx = max(range(len(equations)), key=lambda i: result.eq_max_rel[i])
    verdict = "PASS" if passed else "FAIL"
    paths = []
    if out_dir is not None:       # рендер всех уравнений + sha256 — только для отчёта
        floor_note = (f"ASCII .frd хранит перемещения в E12.5 — сертифицируемый уровень "
                      f"≈{worst.floor_abs:.1e} абс. для этого шага; невязки ниже пола "
                      f"неотличимы от шума формата")
        outcome = GateOutcome(verdict, tol, worst.max_rel_residual, worst.label,
                              worst.worst_eq, floor_note)
        data = build_report(inp, frd, result, outcome, read_ccx_version(frd),
                            [eq.render() for eq in equations])
        paths = write_reports(data, Path(out_dir), inp.stem)
    return Verdict(passed, verdict, worst.max_rel_residual, worst.label,
                   worst.worst_eq, len(equations), result, paths,
                   equations[worst_idx].render())
