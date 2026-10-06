"""Синтетические .inp/.frd фикстуры: формат повторяет боевые файлы ccx 2.20
(узел I10, значения 3×E12.5), см. README «Форматы»."""
from __future__ import annotations

import pytest


def make_inp(path, equations, extra="*BOUNDARY\nALL,1,3,0.0\n"):
    """equations: список списков термов (node, dof, coef)."""
    lines = ["*NODE, NSET=ALL"]
    lines += [f"{n}, {n % 4}, {n % 3}, 0.0" for n in range(1, 6)]
    lines += ["*ELEMENT, TYPE=C3D8, ELSET=EALL", "1, 1, 2, 3, 4, 5, 4, 3, 2", extra or ""]
    for terms in equations:
        lines.append("*EQUATION")
        lines.append(str(len(terms)))
        flat = [str(v) for t in terms for v in t]
        for i in range(0, len(flat), 6):  # перенос по 6 значений, как ccx при 80 колонках
            lines.append(", ".join(flat[i:i + 6]))
    lines.append("*END STEP")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_frd(path, steps, version="Version 2.20"):
    """steps: список (step_no, time, {node: (ux, uy, uz)}). Значения пишутся E12.5."""
    out = ["    1C",
           "    1UDATE              06.october.2026",
           "    1UVERSION           " + version,
           "    1UPGM               CalculiX"]
    for step_no, time_s, disp in steps:
        out.append("  100CL  %d  %s" % (step_no, time_s))
        out.append("    -4  DISP        4    1")
        out.append("    -5  D1          1    2    3")
        for node, (ux, uy, uz) in disp.items():
            out.append(" -1%10d%12.5E%12.5E%12.5E" % (node, ux, uy, uz))
        out.append("    -3")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def tmp_job(tmp_path):
    class Job:
        def __init__(self, root):
            self.root = root
            self.inp = root / "job.inp"
            self.frd = root / "job.frd"
    return Job(tmp_path)
