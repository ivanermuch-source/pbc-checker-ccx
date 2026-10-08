"""Отчётные графики «порог vs результат» для метрик дополнительной валидации.

fig_m_compass   — M-метрики: измерение против порога PASS (лог-шкала).
fig_f_robustness— F-метрики: контракт CLI на мусорных входах + время.
fig_residual_map— карта невязок: пол / гейт / тревога против распределения.

Вызывается из tools/metrics_report.py; вход — dict отчёта run_all().
"""
from __future__ import annotations

from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

C_MEAS, C_THR, C_RUNS, C_DEF = "#1f77b4", "#d62728", "#b0b0b0", "#d9534f"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})


def fig_m_compass(report: Dict, out) -> None:
    """Измерено (доля порога) против порога 1.0; левее порога — PASS."""
    m, thr = report["M"], report["thresholds"]
    items = [
        ("M1  oracle против fsum", m["M1_fsum_headroom"], thr["M1_fsum_headroom"]),
        ("M2  файловый roundtrip E12.5", m["M2_file_headroom"], thr["M2_file_headroom"]),
        ("M3  инвариант нормировки", m["M3_scale_dev"], thr["M3_scale_dev"]),
        ("M4  точность worst_eq", m["M4_worst_eq_hitrate"], thr["M4_worst_eq_hitrate"]),
    ]
    ratios = [v / t for _n, v, t in items]
    y = np.arange(len(items))[::-1]
    # constrained layout: легенда вынесена за оси (right outside) — в лог-шкале
    # правая декада занимает лишь ~24% ширины, любая внутренняя легенда
    # ложится на бары M4/их аннотации
    fig, ax = plt.subplots(figsize=(10.2, 4.2), layout="constrained")
    ax.barh(y, ratios, height=0.55, color=C_MEAS, zorder=3)
    ax.axvline(1.0, color=C_THR, ls="--", lw=2, zorder=4)
    for yi, (_n, v, _t), r in zip(y, items, ratios):
        label = "100 %" if _n.startswith("M4") else f"{v:.1e}"
        ax.annotate(label, (r, yi), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9)
    ax.set_yticks(y, [i[0] for i in items])
    ax.set_xscale("log")
    ax.set_xlim(min(ratios) * 0.05, 5.0)
    ax.set_xlabel("измерено / порог  (лог-шкала)")
    ax.set_title("Oracle-метрики M: измерено против порога PASS\n"
                 "M1–M3 — PASS левее порога; M4 — hit rate, обязан равняться порогу 100 %")
    fig.legend(handles=[Patch(color=C_MEAS, label="измерено"),
                        Line2D([], [], color=C_THR, ls="--", lw=2,
                               label="порог PASS")],
               loc="outside right upper", fontsize=10)
    fig.savefig(out / "fig_m_compass.png")
    plt.close(fig)


def fig_f_robustness(report: Dict, out) -> None:
    """Слева: прогоны и нарушения контракта по категориям мусора;
    справа: время одного прогона CLI против порога 5 с."""
    F = report["F"]
    cats = F["categories"]
    names = list(cats)
    runs = [cats[c]["runs"] for c in names]
    viol = [cats[c]["violations"] for c in names]
    y = np.arange(len(names))[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax1.barh(y, runs, height=0.6, color=C_RUNS, zorder=3)
    ax1.scatter(viol, y, marker="x", s=90, color=C_THR, zorder=4, linewidths=2)
    for yi, v in zip(y, viol):
        ax1.annotate(str(v), (max(v, 0), yi), xytext=(6, 0),
                     textcoords="offset points", va="center",
                     color=C_THR, fontsize=9, fontweight="bold")
    ax1.set_yticks(y, names)
    ax1.set_xlabel("мусорных прогонов / нарушений")
    ax1.set_title(f"F1: контракт кодов выхода 0/1/2\n"
                  f"{sum(viol)} нарушений на {sum(runs)} прогонов — порог 0")
    ax1.legend(handles=[Patch(color=C_RUNS, label="прогонов категории"),
                        Line2D([], [], color=C_THR, marker="x", ls="",
                               markersize=9, markeredgewidth=2,
                               label="нарушений контракта (порог 0)")],
               loc="lower right", fontsize=9)

    times = np.asarray(F["F4_times_ms"], dtype=float)
    ax2.hist(times, bins=40, color=C_MEAS, zorder=3)
    ax2.axvline(5000.0, color=C_THR, ls="--", lw=2, zorder=4)
    ax2.set_xlabel("время одного прогона CLI, мс (лог-шкала)")
    ax2.set_ylabel("прогонов")
    ax2.set_xscale("log")
    # max идёт в заголовок: любое место внутри осей ловит легенду или столбцы
    ax2.set_title("F4: время прогона на мусорном входе\n"
                  f"медиана {np.median(times):.1f} мс, max {times.max():.1f} мс "
                  "— порог 5000 мс")
    ax2.legend(handles=[Patch(color=C_MEAS, label="прогонов с таким временем"),
                        Line2D([], [], color=C_THR, ls="--", lw=2,
                               label="порог F4 = 5 с")],
               loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_f_robustness.png")
    plt.close(fig)


def fig_residual_map(out) -> None:
    """Пол / гейт / тревога против распределения невязок: синтетический прогон
    из 297 связей с шумом формата E12.5 и 3 дефектных связей (инжект)."""
    rng = np.random.default_rng(7)
    good = 10.0 ** rng.uniform(-8.0, np.log10(3e-6), 297)   # до пола чувствительности
    defects = np.array([2e-3, 8e-3, 3e-2])

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bins = np.logspace(-8.3, -1.3, 46)
    ax.hist(good, bins=bins, color=C_MEAS, label="связи с шумом формата E12.5 (297)")
    ax.hist(defects, bins=bins, color=C_DEF, label="дефектные связи — инжект (3)")
    for x, ls, c, lab in (
            (5e-6, ":", "#666666", "пол чувствительности 5·10⁻⁶ (E12.5)"),
            (1e-3, "--", C_THR, "гейт tol = 10⁻³ (FAIL правее)"),
            (1e-2, "-.", "#8b0000", "тревога 10⁻²")):
        ax.axvline(x, color=c, ls=ls, lw=2, label=lab)
    ax.set_xscale("log")
    ymax = ax.get_ylim()[1]
    ax.set_xlabel("относительная невязка  |Σ aᵢ·uᵢ| / max|u|")
    ax.set_ylabel("уравнений")
    ax.set_title("Карта невязок: реальный результат против уровней допуска\n"
                 "синтетический прогон: max_rel = 3·10⁻² → FAIL (за тревогой)")
    ax.annotate("PASS-зона\n(шум формата)", xy=(2.5e-7, 0.55 * ymax), ha="center",
                fontsize=9, color="#1a5276", va="center",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#1a5276", alpha=0.9, zorder=5))
    ax.annotate("FAIL", xy=(1.6e-2, 0.55 * ymax), ha="center", fontsize=10,
                color="#8b0000", fontweight="bold", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#8b0000", alpha=0.9, zorder=5))
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_residual_map.png")
    plt.close(fig)


def fig_c_calibration(report: Dict, out) -> None:
    """Калибровка порога: separation (запас гейта на хороших прогонах)
    по трём «сеткам» и margin инжект-дефектов против порога 10×."""
    C = report["C"]
    # constrained layout + общая легенда вне осей: внутри лог-осей любая
    # легенда рискует накрыть аннотации длинных нижних баров
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8), layout="constrained")

    sizes = list(C["C1_separations"])
    S = [C["C1_separations"][s] for s in sizes]
    y1 = np.arange(len(sizes))[::-1]
    ax1.barh(y1, S, height=0.55, color=C_MEAS, zorder=3)
    ax1.axvline(10.0, color=C_THR, ls="--", lw=2, zorder=4)
    for yi, s in zip(y1, S):
        ax1.annotate(f"S = {s:.0f}×", (s, yi), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=9)
    ax1.set_yticks(y1, [f"{s} пар узлов" for s in sizes])
    ax1.set_xscale("log")
    ax1.set_xlim(1, max(S) * 4)
    ax1.set_xlabel("S = tol / max_rel  (лог-шкала)")
    ax1.set_title("C1: separation на хороших прогонах\n"
                  "невязки у пола — запас до гейта ≥10×")

    names = list(C["C2_cases"])
    margins = [C["C2_cases"][n_]["margin"] if C["C2_cases"][n_]["margin"] is not None
               else float("nan") for n_ in names]
    caught = [C["C2_cases"][n_]["caught"] for n_ in names]
    y2 = np.arange(len(names))[::-1]
    colors = [C_MEAS if m == m else "#ff9896" for m in margins]  # nan → P-audit
    ax2.barh(y2, [m if m == m else 1.0 for m in margins], height=0.55,
             color=colors, zorder=3)
    ax2.axvline(10.0, color=C_THR, ls="--", lw=2, zorder=4)
    for yi, n_, m in zip(y2, names, margins):
        if m == m:
            ax2.annotate(f"margin = {m:.0f}×", (m, yi), xytext=(6, 0),
                         textcoords="offset points", va="center", fontsize=9)
        else:
            ax2.annotate("ловится P-аудитом (невязка чистая)", (1.2, yi),
                         va="center", fontsize=8.5, color="#8b0000")
    ax2.set_yticks(y2, names, fontsize=8.5)
    ax2.set_xscale("log")
    ax2.set_xlim(0.5, 3000)
    ax2.set_xlabel("margin = rel_дефекта / tol  (лог-шкала)")
    ax2.set_title(f"C2: ловимость инжект-дефектов — "
                  f"{sum(caught)}/{len(caught)} пойманы")
    fig.legend(handles=[Patch(color=C_MEAS, label="измерено (S / margin)"),
                        Line2D([], [], color=C_THR, ls="--", lw=2,
                               label="порог = 10×")],
               loc="outside lower center", ncol=2, fontsize=10)
    fig.savefig(out / "fig_c_calibration.png")
    plt.close(fig)


def make_plots(report: Dict, out) -> None:
    fig_m_compass(report, out)
    fig_f_robustness(report, out)
    fig_residual_map(out)
    fig_c_calibration(report, out)
