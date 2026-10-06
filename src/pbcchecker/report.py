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

# сколько худших уравнений показывать в отчёте
TOP_WORST = 10


@dataclass(frozen=True)
class GateOutcome:
    """Вердикт гейта для отчёта (заполняется в gate.check_job)."""
    verdict: str            # PASS | FAIL | N/A
    tol: float
    max_rel_residual: float
    worst_step: str
    worst_eq: int
    floor_note: str


def sha256(path: Path) -> str:
    """sha256 файла потоково (чанками по 1 МБ)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_report(inp: Path, frd: Optional[Path], result: Optional[ResidualResult],
                 outcome: GateOutcome, ccx_version: str,
                 equations_rendered: List[str]) -> dict:
    """Отчёт-словарь; result=None для вердикта N/A (frd может отсутствовать)."""
    if result is not None:
        # топ худших уравнений — по max относительной невязке по всем шагам,
        # а не по порядку в deck
        order = sorted(range(result.n_equations),
                       key=lambda i: result.eq_max_rel[i], reverse=True)[:TOP_WORST]
        worst = [{"index": i, "max_rel_residual": result.eq_max_rel[i],
                  "equation": equations_rendered[i]} for i in order]
        steps = [asdict(s) for s in result.steps]
        n_equations, eq_rendered = result.n_equations, equations_rendered
    else:
        worst, steps, n_equations, eq_rendered = [], [], 0, []
    return {
        "tool": {"name": "pbcchecker", "version": __version__},
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {"inp": str(inp), "inp_sha256": sha256(inp),
                   "frd": str(frd) if frd else None,
                   "frd_sha256": sha256(frd) if frd else None},
        "ccx_version": ccx_version,
        "n_equations": n_equations,
        "equations": eq_rendered,
        "worst_equations": worst,
        "tol_rel": outcome.tol,
        "verdict": outcome.verdict,
        "max_rel_residual": outcome.max_rel_residual,
        "worst_step": outcome.worst_step,
        "worst_equation_index": outcome.worst_eq,
        "sensitivity_floor": outcome.floor_note,
        "steps": steps,
    }


def write_reports(data: dict, out_dir: Optional[Path], job_stem: str) -> List[Path]:
    """Пишет JSON + Markdown в out_dir; без out_dir ничего не пишет и возвращает [].

    Имя содержит дату UTC; при повторном прогоне того же дня добавляется время
    HHMMSS — audit_trace остаётся append-only, отчёты не перезаписываются."""
    written: List[Path] = []
    if out_dir is None:
        return written
    now = datetime.now(timezone.utc)
    out_dir.mkdir(parents=True, exist_ok=True)
    uniq, n = f"{now:%Y-%m-%d}", 1
    while (out_dir / f"pbc_report_{job_stem}_{uniq}.json").exists():   # без перезаписи
        uniq, n = f"{now:%Y-%m-%d}_{now:%H%M%S}" + (f"_{n}" if n > 1 else ""), n + 1
    jpath = out_dir / f"pbc_report_{job_stem}_{uniq}.json"
    jpath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(jpath)
    mpath = out_dir / f"pbc_report_{job_stem}_{uniq}.md"
    mpath.write_text(to_markdown(data), encoding="utf-8")
    written.append(mpath)
    return written


def to_markdown(d: dict) -> str:
    """Markdown-представление отчёта (данные из build_report)."""
    frd_sha = d["inputs"]["frd_sha256"]
    frd_line = (f"- .frd: `{d['inputs']['frd']}` (sha256 `{frd_sha[:16]}…`)"
                if frd_sha else "- .frd: не найден (N/A)")
    lines = [
        f"# PBC residual check — {d['verdict']}",
        "",
        f"*Дата (UTC):* {d['date_utc']}  ",
        f"*pbcchecker:* {d['tool']['version']}  ",
        f"*ccx:* {d['ccx_version']}",
        "",
        f"- .inp: `{d['inputs']['inp']}` (sha256 `{d['inputs']['inp_sha256'][:16]}…`)",
        frd_line,
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
    if d["worst_equations"]:
        lines += ["", f"## Худшие уравнения (top-{len(d['worst_equations'])} "
                      "по max|r|/max|u|)", ""]
        lines += [f"- {w['max_rel_residual']:.3e} `{w['equation']}`"
                  for w in d["worst_equations"]]
    return "\n".join(lines) + "\n"
