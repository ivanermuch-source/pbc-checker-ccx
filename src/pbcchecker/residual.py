"""Векторный расчёт PBC-невязок: r_j = sum_i a_ji * u(node_ji, dof_ji).

Невязка уравнения — число; идеальный прогон даёт r = 0 с точностью формата
файла (.frd ASCII: E12.5 => пол ~5e-6 от max|u|). Всё считается в numpy
и сворачивается через np.add.reduceat: O(термов) операций, без питоновских
циклов по термам.

Узел уравнения без DISP — ошибка данных (KeyError), а не WARN: гейт не должен
молча пропускать связи, по которым нет чем проверять.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .frd_disp import DispBlock
from .inp_equations import Equation

# Полушаг формата E12.5: последняя значащая цифра мантиссы.
FIELD_EPS = 5e-6


@dataclass(frozen=True)
class StepResidual:
    label: str
    max_abs_u: float
    max_abs_residual: float
    max_rel_residual: float
    floor_abs: float          # 0.5*10^(-(digits)) * max_abs_u — предел чувствительности
    worst_eq: int             # индекс уравнения с максимальной |r|
    n_nodes: int


@dataclass(frozen=True)
class ResidualResult:
    steps: List[StepResidual]
    n_equations: int
    eq_max_rel: List[float]   # max по шагам относительной невязки каждого уравнения


def term_arrays(equations: Sequence[Equation]) -> Tuple[np.ndarray, np.ndarray,
                                                         np.ndarray, np.ndarray]:
    """Термы всех уравнений в плоские массивы: (узлы, dof-1, коэф., offsets).

    offsets — стартовые индексы уравнений в плоских массивах (для reduceat),
    len(offsets) = n_equations + 1; parse_equations гарантирует >= 1 терму
    в уравнении, поэтому offsets строго возрастают.
    """
    n_terms = sum(len(eq.terms) for eq in equations)
    nodes = np.fromiter((n for eq in equations for n, _d, _c in eq.terms),
                        dtype=np.int64, count=n_terms)
    dofs = np.fromiter((d - 1 for eq in equations for _n, d, _c in eq.terms),
                       dtype=np.int64, count=n_terms)
    coefs = np.fromiter((c for eq in equations for _n, _d, c in eq.terms),
                        dtype=np.float64, count=n_terms)
    offsets = np.zeros(len(equations) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(eq.terms) for eq in equations])
    return nodes, dofs, coefs, offsets


def compute_residuals(equations: Sequence[Equation], blocks: List[DispBlock]) -> ResidualResult:
    if not equations:
        raise ValueError("нет уравнений *EQUATION")
    if not blocks:
        raise ValueError("нет блоков DISP в .frd (добавьте '*NODE FILE' с 'U' в deck)")
    for blk in blocks:
        if blk.nodes.size == 0:
            raise ValueError(f"пустой блок DISP '{blk.label}' — нет строк узлов")

    union_nodes = np.unique(np.concatenate([b.nodes for b in blocks]))
    term_nodes, dofs, coefs, offsets = term_arrays(equations)

    # глобальная проверка: узел вне объединения всех блоков — ошибки данных
    pos = np.searchsorted(union_nodes, term_nodes)
    inside = pos < len(union_nodes)
    hit = np.zeros(len(term_nodes), dtype=bool)
    hit[inside] = union_nodes[pos[inside]] == term_nodes[inside]
    if not hit.all():
        bad = sorted({int(x) for x in term_nodes[~hit]})
        raise KeyError(f"узлы {bad[:10]}{'…' if len(bad) > 10 else ''} из *EQUATION "
                       f"отсутствуют во всех блоках DISP (всего {len(bad)})")
    rows = pos  # индексы строк в union_nodes

    # узел может отсутствовать в одном блоке, но присутствовать в другом —
    # проверяем по каждому блоку отдельно (векторный searchsorted на блок)
    steps: List[StepResidual] = []
    eq_max_rel = np.zeros(len(equations), dtype=np.float64)
    for blk in blocks:
        so = np.argsort(blk.nodes, kind="stable")
        sorted_b = blk.nodes[so]
        p = np.clip(np.searchsorted(sorted_b, union_nodes), 0, len(sorted_b) - 1)
        found = sorted_b[p] == union_nodes
        mapped = np.where(found, so[p], -1)[rows]
        absent = mapped < 0
        if absent.any():
            bad = sorted({int(x) for x in term_nodes[absent]})
            raise KeyError(f"узлы {bad[:10]}{'…' if len(bad) > 10 else ''} из *EQUATION "
                           f"отсутствуют в блоке '{blk.label}'")
        r = np.add.reduceat(coefs * blk.u[mapped, dofs], offsets[:-1])
        absr = np.abs(r)               # n_equations >= 1, пустые блоки отсечены выше
        k = int(np.argmax(absr))
        max_abs_u = float(np.abs(blk.u).max())
        max_abs_r = float(absr.max())
        denom = max(max_abs_u, 1e-300)
        eq_max_rel = np.maximum(eq_max_rel, absr / denom)
        steps.append(StepResidual(
            label=blk.label, max_abs_u=max_abs_u, max_abs_residual=max_abs_r,
            max_rel_residual=max_abs_r / denom,
            floor_abs=FIELD_EPS * denom, worst_eq=k, n_nodes=int(len(blk.nodes))))
    return ResidualResult(steps=steps, n_equations=len(equations),
                          eq_max_rel=eq_max_rel.tolist())
