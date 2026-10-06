"""Векторный расчёт PBC-невязок: r_j = sum_i a_ji * u(node_ji, dof_ji).

Невязка уравнения — число; идеальный прогон даёт r = 0 с точностью формата
файла (.frd ASCII: E12.5 => пол ~5e-6 от max|u|). Всё считывается в numpy
и сворачивается через np.add.reduceat: O(термов) операций, без питоновских циклов.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

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
    missing_nodes: Sequence[int]  # узлы уравнений, которых нет в DISP


def term_arrays(equations: Sequence[Equation], nodes: np.ndarray):
    """Отображение термов уравнений в индексы строк массива u.

    Возвращает (rows, dofs0, coefs, offsets, missing): offsets — стартовые индексы
    уравнений для reduceat, missing — узлы, отсутствующие в блоке DISP.
    """
    order = np.argsort(nodes)
    sorted_nodes = nodes[order]
    rows_list, dofs_list, coefs_list, offsets = [], [], [], [0]
    missing = set()
    pos = 0
    for eq in equations:
        for node, dof, coef in eq.terms:
            j = np.searchsorted(sorted_nodes, node)
            if j < len(sorted_nodes) and sorted_nodes[j] == node:
                rows_list.append(int(order[j]))
            else:
                missing.add(node)
                rows_list.append(-1)
            dofs_list.append(dof - 1)
            coefs_list.append(coef)
        offsets.append(len(rows_list))
    return (np.asarray(rows_list, dtype=np.int64),
            np.asarray(dofs_list, dtype=np.int8),
            np.asarray(coefs_list, dtype=np.float64),
            np.asarray(offsets, dtype=np.int64),
            sorted(missing))


def compute_residuals(equations: Sequence[Equation], blocks: List[DispBlock]) -> ResidualResult:
    if not equations:
        raise ValueError("нет уравнений *EQUATION")
    if not blocks:
        raise ValueError("нет блоков DISP в .frd (добавьте '*NODE FILE' с 'U' в deck)")

    union_nodes = np.unique(np.concatenate([b.nodes for b in blocks]))
    rows, dofs, coefs, offsets, missing = term_arrays(equations, union_nodes)
    # узел может отсутствовать в одном блоке, но присутствовать в другом — проверяем по каждому
    steps: List[StepResidual] = []
    valid = rows >= 0
    term_nodes = np.asarray([n for eq in equations for n, _d, _c in eq.terms], dtype=np.int64)

    for blk in blocks:
        local = {int(n): i for i, n in enumerate(blk.nodes.tolist())}  # узлы блока -> строка блока
        block_row = np.array([local.get(int(n), -1) for n in union_nodes], dtype=np.int64)
        mapped = np.where(valid, block_row[np.clip(rows, 0, None)], -1)
        absent = mapped < 0
        if absent.any():
            bad = sorted({int(x) for x in term_nodes[absent]})
            raise KeyError(f"узлы {bad[:10]}{'…' if len(bad) > 10 else ''} из *EQUATION "
                           f"отсутствуют в блоке '{blk.label}'")
        r = np.add.reduceat((coefs * blk.u[mapped, dofs]).reshape(-1),
                            offsets[:-1]) if len(offsets) > 1 else np.array([])
        # reduceat по пустым группам (n_terms=0) невозможен — parse_equations их не допускает
        absr = np.abs(r)
        k = int(np.argmax(absr)) if len(absr) else 0
        max_abs_u = float(np.abs(blk.u).max()) if blk.u.size else 0.0
        max_abs_r = float(absr.max()) if len(absr) else 0.0
        denom = max(max_abs_u, 1e-300)
        steps.append(StepResidual(
            label=blk.label, max_abs_u=max_abs_u, max_abs_residual=max_abs_r,
            max_rel_residual=max_abs_r / denom,
            floor_abs=FIELD_EPS * denom, worst_eq=k, n_nodes=int(len(blk.nodes))))
    return ResidualResult(steps=steps, n_equations=len(equations), missing_nodes=missing)
