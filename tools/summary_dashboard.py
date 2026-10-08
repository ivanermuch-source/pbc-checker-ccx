"""Сводный дашборд результатов дополнительной валидации — одно изображение.

Шесть панелей «порог vs результат»: M (oracle-арифметика), F (робастность
CLI на мусоре), C1/C2 (калибровка порога), P (аудит *EQUATION на RVE-кейсе
и его дефектных вариантах — считается в момент генерации через audit_pbc)
и статус-таблица серий. Данные M/F/C — из docs/metrics/metrics.json.

Принципы читаемости (после ревью пользователем): крупный шрифт, короткие
однострочные заголовки панелей; назначение линий порога объясняется в
подписи оси X, а не текстом возле линии; подписи значений у порога — внутрь
бара белым; никаких аннотаций поверх чужих баров.

Запуск (после tools/metrics_report.py):  python tools/summary_dashboard.py
Выход: docs/metrics/fig_summary_dashboard.png
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from pbcchecker.inp_equations import parse_equations
from pbcchecker.pbc_audit import audit_pbc

ROOT = Path(__file__).resolve().parents[1]

C_MEAS, C_THR, C_RUNS, C_DEF, C_SYM = "#1f77b4", "#d62728", "#b0b0b0", "#d9534f", "#e8850c"
C_OK, C_WAIT = "#e8f5e9", "#fffde7"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 11.5,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})


def _rve_variants(tmp: Path) -> List[Tuple[str, float, float, int]]:
    """RVE-кейс P-аудита: (метка, coverage, symmetry, n_dups) — валидный
    сетап и три дефектных, значения честно из audit_pbc."""
    pairs = [(100 + j, 200 + j) for j in range(1, 7)]

    def build(name: str, *, drop=None, asym=False, dup=False) -> Path:
        lines = ["*NODE"]
        for a, b in pairs:
            j = a - 101
            y, z = j % 3, j // 3
            lines += [f"{a}, 0.0, {y}.0, {z}.0", f"{b}, 1.0, {y}.0, {z}.0"]
        for i, (a, b) in enumerate(pairs):
            if drop is not None and i == drop:
                continue
            for d in (1, 2, 3):
                cb = -0.5 if (asym and i == 1 and d == 1) else -1.0
                lines += ["*EQUATION", "2", f"{a}, {d}, 1.0, {b}, {d}, {cb}"]
        if dup:
            a, b = pairs[0]
            lines += ["*EQUATION", "2", f"{a}, 1, 1.0, {b}, 1, -1.0"]
        p = tmp / f"{name}.inp"
        p.write_text("\n".join(lines) + "\n", encoding="ascii")
        return p

    out = []
    for label, name, kw in (("валидный сетап", "ok", {}),
                            ("пропущена пара", "drop", {"drop": 2}),
                            ("незеркальная пара", "asym", {"asym": True}),
                            ("дубликат → FAIL", "dup", {"dup": True})):
        p = build(name, **kw)
        res = audit_pbc(p, parse_equations(p))
        d = res.directions[0] if res.directions else None
        out.append((label,
                    d.coverage if d else 0.0,
                    d.symmetry if d else 0.0,
                    len(res.duplicates)))
    return out


def _val_label(ax, val, yy, inside_from, fmt="{:.2f}", fs=10.5):
    """Подпись значения: у порога — внутрь бара белым, иначе справа чёрным."""
    inside = val >= inside_from
    ax.annotate(fmt.format(val), (val, yy),
                xytext=(-4 if inside else 6, 0), textcoords="offset points",
                va="center", ha="right" if inside else "left", fontsize=fs,
                color="white" if inside else "black",
                fontweight="bold" if inside else "normal", zorder=6)


def make_dashboard(report: Dict, out_dir: Path) -> Path:
    fig = plt.figure(figsize=(18.0, 10.2))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.36, width_ratios=(1, 1, 1.5),
                          left=0.115, right=0.985, top=0.885, bottom=0.065)

    # ---- M: oracle-арифметика ------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    m, thr = report["M"], report["thresholds"]
    items = [("M1  fsum-оракул", m["M1_fsum_headroom"], thr["M1_fsum_headroom"]),
             ("M2  roundtrip", m["M2_file_headroom"], thr["M2_file_headroom"]),
             ("M3  нормировка", m["M3_scale_dev"], thr["M3_scale_dev"]),
             ("M4  worst_eq", m["M4_worst_eq_hitrate"], thr["M4_worst_eq_hitrate"])]
    ratios = [v / t for _n, v, t in items]
    y = np.arange(len(items))[::-1]
    ax.barh(y, ratios, height=0.58, color=C_MEAS, zorder=3)
    ax.axvline(1.0, color=C_THR, ls="--", lw=2.2, zorder=4)
    for yi, (_n, v, _t), r in zip(y, items, ratios):
        if _n.startswith("M4"):
            _val_label(ax, r, yi, inside_from=0.9, fmt="{:.0%}")
        else:
            _val_label(ax, r, yi, inside_from=0.9, fmt="{:.1e}")
    ax.set_yticks(y, [i[0] for i in items], fontsize=10.5)
    ax.set_xscale("log")
    ax.set_xlim(min(ratios) * 0.04, 6.0)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.set_xlabel("измерено / порог, лог-шкала;\nкрасный пунктир — порог PASS (1.0)",
                  fontsize=10)
    m_pass = sum(1 for n_, v, t in items
                 if (v >= t if n_.startswith("M4") else v <= t))
    ax.set_title(f"M: oracle-арифметика — {m_pass}/4 "
                 f"{'PASS' if m_pass == 4 else 'FAIL'}", fontsize=12.5)

    # ---- F: робастность ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    F = report["F"]
    cats = F["categories"]
    names = list(cats)
    runs = [cats[c]["runs"] for c in names]
    viol = [cats[c]["violations"] for c in names]
    y = np.arange(len(names))[::-1]
    ax.barh(y, runs, height=0.62, color=C_RUNS, zorder=3)
    ax.scatter(viol, y, marker="x", s=110, color=C_THR, zorder=5, linewidths=2.5)
    for yi, v in zip(y, viol):
        ax.annotate(str(v), (v, yi), xytext=(8, 0), textcoords="offset points",
                    va="center", color=C_THR, fontsize=10.5, fontweight="bold")
    ax.set_yticks(y, names, fontsize=9.5)
    ax.set_xlim(-6, max(runs) * 1.2)
    ax.tick_params(axis="x", labelsize=9.5)
    total_viol = sum(viol)
    ax.set_xlabel(f"мусорных прогонов; красный × — нарушений контракта\n"
                  f"({total_viol} на {sum(runs)}); медиана "
                  f"{np.median(F['F4_times_ms']):.1f} мс, "
                  f"max {max(F['F4_times_ms']):.0f} мс — порог 5 с", fontsize=10)
    ax.set_title(f"F: контракт кодов 0/1/2 — {total_viol} нарушений", fontsize=12.5)

    # ---- C1: separation --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    C = report["C"]
    sizes = list(C["C1_separations"])
    S = [C["C1_separations"][s] for s in sizes]
    y = np.arange(len(sizes))[::-1]
    ax.barh(y, S, height=0.58, color=C_MEAS, zorder=3)
    ax.axvline(10.0, color=C_THR, ls="--", lw=2.2, zorder=4)
    for yi, s in zip(y, S):
        _val_label(ax, s, yi, inside_from=9e9, fmt="S = {:.0f}×")
    ax.set_yticks(y, [f"{s} пар узлов" for s in sizes], fontsize=10.5)
    ax.set_xscale("log")
    ax.set_xlim(1, max(S) * 8)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.set_xlabel("S = tol / max_rel, лог-шкала;\nкрасный пунктир — порог 10×",
                  fontsize=10)
    s_min = min(S)
    ax.set_title(f"C1: separation — хорошие прогоны у пола, запас {s_min:.0f}×",
                 fontsize=12.5)

    # ---- C2: ловимость --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    cases = C["C2_cases"]
    cnames = {"сдвиг узла 3e-2·max|u|": "сдвиг узла",
              "подмена узла (разрыв)": "подмена узла",
              "удалённая связь (--pbc-audit)": "удалённая связь"}
    ckeys = list(cases)
    margins = [cases[n]["margin"] if cases[n]["margin"] is not None else np.nan
               for n in ckeys]
    y = np.arange(len(ckeys))[::-1]
    vals = [m if m == m else 1.0 for m in margins]
    colors = [C_MEAS if m == m else C_DEF for m in margins]
    ax.barh(y, vals, height=0.58, color=colors, zorder=3)
    ax.axvline(10.0, color=C_THR, ls="--", lw=2.2, zorder=4)
    for yi, m in zip(y, margins):
        if m == m:
            _val_label(ax, m, yi, inside_from=9e9, fmt="{:.0f}×")
        else:
            ax.annotate("невязка чистая → ловит P-аудит", (2.0, yi),
                        va="center", fontsize=9.5, color="#8b0000",
                        fontweight="bold", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="#8b0000", alpha=1.0))
    ax.set_yticks(y, [cnames[k] for k in ckeys], fontsize=10.5)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 20000)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.set_xlabel("margin = rel_дефекта / tol, лог-шкала;\nкрасный пунктир — порог 10×",
                  fontsize=10)
    n_caught = sum(1 for c in cases.values() if c["caught"])
    margin_min = min(c["margin"] for c in cases.values() if c["margin"] is not None)
    ax.set_title(f"C2: ловимость инжект-дефектов — {n_caught}/{len(cases)}",
                 fontsize=12.5)

    # ---- P: аудит сетапа --------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="pbc_dash_") as td:
        variants = _rve_variants(Path(td))
    ax = fig.add_subplot(gs[1, 1])
    vnames = [v[0] for v in variants]
    cov = [v[1] for v in variants]
    sym = [v[2] for v in variants]
    y = np.arange(len(vnames))[::-1]
    h = 0.34
    ax.barh(y + h / 2, cov, height=h, color=C_MEAS, zorder=3)
    ax.barh(y - h / 2, sym, height=h, color=C_SYM, zorder=3)
    ax.axvline(1.0, color=C_THR, ls="--", lw=1.8, zorder=4)
    for yi, c, s in zip(y, cov, sym):
        _val_label(ax, c, yi + h / 2, inside_from=0.90)
        _val_label(ax, s, yi - h / 2, inside_from=0.90)
    ax.set_yticks(y, vnames, fontsize=10)
    ax.set_xlim(0, 1.13)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="x", labelsize=9.5)
    ax.set_xlabel("покрытие (синий) и симметрия пар (оранжевый);\n"
                  "красный пунктир — норма 1.0", fontsize=10)
    n_def = sum(1 for v in variants[1:] if v[1] < 1.0 or v[2] < 1.0 or v[3] > 0)
    ax.set_title(f"P: --pbc-audit — {n_def}/{len(variants) - 1} дефекта сетапа "
                 "детектированы", fontsize=12.5)

    # ---- статус-таблица (ручная отрисовка: mpl-table обрезает текст) -----------
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    rows = [
        ("M oracle", f"{m_pass}/4 против fsum и инвариантов",
         "PASS" if m_pass == 4 else "FAIL"),
        ("F fuzz-контракт", f"{F['n_fuzz']} входов, {total_viol} crash",
         "PASS" if total_viol == 0 and F["F2_exit_code_accuracy"] == 1.0 else "FAIL"),
        ("P аудит сетапа", f"{n_def}/3 дефекта ловятся",
         "PASS" if n_def == 3 else "FAIL"),
        ("C калибровка", f"S = {s_min:.0f}×, margin ≥ {margin_min:.0f}×",
         "PASS" if s_min >= 10 and margin_min >= 10 else "FAIL"),
        ("G golden ccx", "ждёт 2.17 / 2.19 / 2.22", "SKIPPED"),
        ("Баг (M2)", "тихая потеря уравнений", "FIXED"),
    ]
    x0, x1, x2, x3 = 0.0, 0.27, 0.78, 1.0
    ytop, ybot = 0.98, 0.06
    n = len(rows) + 1
    dy = (ytop - ybot) / n
    for i in range(n):
        yy = ytop - (i + 1) * dy
        face = "#eeeeee" if i == 0 else "#ffffff"
        ax.add_patch(Rectangle((x0, yy), x3 - x0, dy, transform=ax.transAxes,
                               facecolor=face, edgecolor="#aaaaaa", lw=0.7, zorder=2))
    for i, row in enumerate([("Серия", "Результат", "Итог")] + rows):
        yy = ytop - (i + 1) * dy + dy / 2
        ax.text(x0 + 0.014, yy, row[0], transform=ax.transAxes, va="center",
                fontsize=10.5, fontweight="bold")
        ax.text(x1 + 0.014, yy, row[1], transform=ax.transAxes, va="center",
                fontsize=10.5)
        if i > 0:
            face = {"PASS": C_OK, "FIXED": C_OK, "SKIPPED": C_WAIT,
                    "FAIL": "#fdecea"}[row[2]]
            ax.add_patch(Rectangle((x2, ytop - (i + 1) * dy), x3 - x2, dy,
                                   transform=ax.transAxes, facecolor=face,
                                   edgecolor="#aaaaaa", lw=0.7, zorder=3))
        ax.text((x2 + x3) / 2, yy, row[2] if i > 0 else "Итог",
                transform=ax.transAxes, va="center", ha="center", fontsize=10.5,
                fontweight="bold")
    ax.set_title("Статус серий доп. валидации", fontsize=12.5)

    all_pass = report["summary"]["all_pass"]
    fig.suptitle("pbc-checker-ccx — дополнительная валидация: порог против результата",
                 fontsize=16, y=0.965)
    fig.text(0.5, 0.918,
             f"oracle-метрики · fuzz-контракт · калибровка порога · аудит сетапа | "
             f"v{report['tool_version']}, {report['generated'][:10]} | "
             f"итог: {'PASS' if all_pass else 'FAIL'}",
             ha="center", fontsize=11.5, color="#444444")
    out = out_dir / "fig_summary_dashboard.png"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    report = json.loads((ROOT / "docs/metrics/metrics.json").read_text(encoding="utf-8"))
    path = make_dashboard(report, ROOT / "docs/metrics")
    print(f"дашборд: {path}")
