"""Отчёты: JSON + Markdown в стиле audit_trace (дата, sha256 входов, вердикт)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from ._version import __version__
from .residual import ResidualResult


@dataclass(frozen=True)
class GateOutcome:
    verdict: str            # PASS | FAIL | N/A
    tol: float
    max_rel_residual: float
    worst_step: str
    worst_eq: int
    floor_note: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_report(inp: Path, frd: Path, result: ResidualResult,
                 outcome: GateOutcome, ccx_version: str,
                 equations_rendered: List[str]) -> dict:
    return {
        "tool": {"name": "pbcchecker", "version": __version__},
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {"inp": str(inp), "inp_sha256": sha256(inp),
                   "frd": str(frd), "frd_sha256": sha256(frd)},
        "ccx_version": ccx_version,
        "n_equations": result.n_equations,
        "equations": equations_rendered,
        "missing_nodes": list(result.missing_nodes),
        "tol_rel": outcome.tol,
        "verdict": outcome.verdict,
        "max_rel_residual": outcome.max_rel_residual,
        "worst_step": outcome.worst_step,
        "worst_equation_index": outcome.worst_eq,
        "sensitivity_floor": outcome.floor_note,
        "steps": [asdict(s) for s in result.steps],
    }


def write_reports(data: dict, out_dir: Optional[Path], job_stem: str) -> List[Path]:
    """JSON всегда; Markdown рядом. Без out_dir — только возврат путей stdout-режима."""
    written: List[Path] = []
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if out_dir is None:
        return written
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / f"pbc_report_{job_stem}_{stamp}.json"
    jpath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(jpath)
    mpath = out_dir / f"pbc_report_{job_stem}_{stamp}.md"
    mpath.write_text(to_markdown(data), encoding="utf-8")
    written.append(mpath)
    return written


def to_markdown(d: dict) -> str:
    lines = [
        f"# PBC residual check — {d['verdict']}",
        "",
        f"*Дата (UTC):* {d['date_utc']}  ",
        f"*pbcchecker:* {d['tool']['version']}  ",
        f"*ccx:* {d['ccx_version']}",
        "",
        f"- .inp: `{d['inputs']['inp']}` (sha256 `{d['inputs']['inp_sha256'][:16]}…`)",
        f"- .frd: `{d['inputs']['frd']}` (sha256 `{d['inputs']['frd_sha256'][:16]}…`)",
        f"- уравнений: {d['n_equations']}, порог (отн.): {d['tol_rel']:g}",
        f"- макс. относительная невязка: **{d['max_rel_residual']:.3e}** "
        f"(шаг «{d['worst_step']}», уравнение {d['worst_equation_index']})",
        f"- предел чувствительности: {d['sensitivity_floor']}",
        "",
        "| шаг | max|u| | max|r| | max r/|u| | пол |r| | худшее ур. | узлов |",
        "|---|---|---|---|---|---|",
    ]
    for s in d["steps"]:
        lines.append(f"| {s['label']} | {s['max_abs_u']:.3e} | {s['max_abs_residual']:.3e} "
                     f"| {s['max_rel_residual']:.3e} | {s['floor_abs']:.1e} "
                     f"| {s['worst_eq']} | {s['n_nodes']} |")
    if d["missing_nodes"]:
        lines += ["", f"⚠ Узлы уравнений без DISP: {d['missing_nodes'][:10]}"]
    lines += ["", "## Худшие уравнения", ""]
    lines += [f"- `{eq}`" for eq in d["equations"][:10]]
    return "\n".join(lines) + "\n"
