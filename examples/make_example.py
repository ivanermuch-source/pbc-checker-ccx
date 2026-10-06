"""Генерирует демонстрационный прогон examples/demo_job.{inp,frd} и проверяет его.

Запуск:  python examples/make_example.py
"""
from pathlib import Path

from pbcchecker import check_job

HERE = Path(__file__).parent

# три связи призрак-представитель (как в АВТОсбор: ghost первый, коэф. +1/-1)
EQS = [[(11, 1, 1.0), (21, 1, -1.0)],
       [(12, 2, 1.0), (22, 2, -1.0)],
       [(13, 3, 2.0), (23, 3, -2.0)]]  # произвольный масштаб коэффициента
STEPS = [(1, "0.1000000E-01", {11: (1.5e-4, 0, 2e-4), 12: (0, 1e-4, 1e-4),
                               13: (1e-4, 2e-4, 2.5e-4),
                               21: (1.5e-4, 0, 2e-4), 22: (0, 1e-4, 1e-4),
                               23: (1e-4, 2e-4, 2.5e-4)}),
         (2, "0.2000000E-01", {11: (3e-4, 0, 4e-4), 12: (0, 2e-4, 2e-4),
                               13: (2e-4, 4e-4, 5e-4),
                               21: (3e-4, 0, 4e-4), 22: (0, 2e-4, 2e-4),
                               23: (2e-4, 4e-4, 5e-4)})]

lines = ["*NODE, NSET=ALL"]
for n in range(11, 24):
    lines.append(f"{n}, {n % 3}, {(n + 1) % 3}, 0.0")
lines += ["*ELEMENT, TYPE=C3D8, ELSET=EALL", "1, 11, 12, 13, 21, 22, 23, 12, 13",
          "*STEP", "*STATIC"]
for terms in EQS:
    lines.append("*EQUATION")
    lines.append(str(len(terms)))
    lines.append(", ".join(f"{v}" for t in terms for v in t))
lines.append("*NODE FILE\nU\n*END STEP")
(HERE / "demo_job.inp").write_text("\n".join(lines), encoding="utf-8")

out = ["    1C", "    1UVERSION           Version 2.20", "    1UPGM               CalculiX"]
for step_no, time_s, disp in STEPS:
    out.append("  100CL  %d  %s" % (step_no, time_s))
    out.append("    -4  DISP        4    1")
    out.append("    -5  D1          1    2    3")
    for node, (ux, uy, uz) in disp.items():
        out.append(" -1%10d%12.5E%12.5E%12.5E" % (node, ux, uy, uz))
    out.append("    -3")
(HERE / "demo_job.frd").write_text("\n".join(out) + "\n", encoding="utf-8")

v = check_job(HERE / "demo_job.inp", HERE / "demo_job.frd", out_dir=HERE / "reports")
print(f"{v.verdict}: max_rel={v.max_rel_residual:.3e}, уравнений {v.n_equations}")
print("отчёты:", *[str(p) for p in v.report_paths], sep="\n  ")
