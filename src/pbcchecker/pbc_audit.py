"""Аудит самих *EQUATION по геометрии узлов (P-метрики: покрытие, симметрия,
дубликаты).

Закрывает слепую зону гейта невязок: связи могут выполняться идеально
(Σ aᵢ·uᵢ ≈ 0), но составлены криво — пропущенные узлы грани, незеркальные
коэффициенты пары, дубликаты уравнений. Невязка этого не видит: она проверяет
исполнение связей солвером, а не их составление.

Источник геометрии — карточки *NODE (с раскрытием *INCLUDE, как и для
*EQUATION). Направление периодичности определяется автоматически: ось, у
экстремальных плоскостей которой есть уравнения с узлами обеих плоскостей.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .inp_equations import MAX_INCLUDE_DEPTH, Equation, to_float

# узел считается лежащим на экстремальной плоскости в пределах доли от span оси
FACE_TOL = 1e-6
# коэффициенты пары зеркальны, если |c⁺ + c⁻| ≤ SYM_TOL·max(|c⁺|, |c⁻|)
SYM_TOL = 1e-9
_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class DirectionAudit:
    """Аудит одного направления периодичности."""
    axis: int                    # 0/1/2 → x/y/z
    n_face_nodes: int            # узлов на обеих гранях
    n_covered: int               # из них встречается в уравнениях направления
    coverage: float              # n_covered / n_face_nodes (P1)
    uncovered: List[int]         # узлы граней без связей (первые 20)
    n_equations: int
    n_symmetric: int
    symmetry: float              # n_symmetric / n_equations (P2)
    asymmetric_eqs: List[int]    # индексы незеркальных уравнений (первые 20)

    @property
    def label(self) -> str:
        """Имя оси: «x»/«y»/«z»."""
        return _AXES[self.axis]


@dataclass(frozen=True)
class PbcAuditResult:
    """Итог аудита: направления периодичности, дубликаты, вердикт PASS/FAIL/N/A."""
    verdict: str                 # PASS | FAIL | N/A (направления не найдены)
    directions: List[DirectionAudit] = field(default_factory=list)
    duplicates: List[Tuple[int, int]] = field(default_factory=list)  # пары индексов
    n_nodes: int = 0
    note: str = ""


def parse_nodes(inp_path: Path, _seen=None, _depth: int = 0) -> Dict[int, Tuple[float, float, float]]:
    """Узлы из карточек *NODE: {узел: (x, y, z)}, с рекурсивным *INCLUDE.

    Повреждённая строка узла или узел с разными координатами — ValueError
    с номером строки (как у parse_equations: испорченный .inp не молчит)."""
    inp_path = Path(inp_path)
    nodes: Dict[int, Tuple[float, float, float]] = {}
    if _seen is None:
        _seen = frozenset()
    key = inp_path.resolve()
    if _depth > MAX_INCLUDE_DEPTH:
        raise ValueError(f"{inp_path}: глубина *INCLUDE больше {MAX_INCLUDE_DEPTH} — цикл?")
    if key in _seen:
        raise ValueError(f"{inp_path}: цикл *INCLUDE (файл включён повторно)")
    with inp_path.open(encoding="utf-8-sig", errors="ignore") as f:
        in_node = False
        for lineno, raw in enumerate(f, 1):
            s = raw.strip()
            if not s:
                continue
            if s.startswith("**"):
                continue
            if s.startswith("*"):
                in_node = s.split(",")[0].strip().upper() == "*NODE"
                continue
            if not in_node:
                continue
            parts = [p.strip() for p in s.split(",")]
            try:
                node = int(parts[0])
                xyz = tuple(to_float(p) for p in parts[1:4]) + (0.0,) * (4 - len(parts))
            except (ValueError, IndexError) as e:
                raise ValueError(f"{inp_path}: *NODE: строка {lineno} не разбирается: "
                                 f"{s!r}") from e
            xyz3 = xyz[:3]
            if node in nodes and nodes[node] != xyz3:
                raise ValueError(f"{inp_path}: *NODE: узел {node} переопределён "
                                 f"(строка {lineno})")
            nodes[node] = xyz3
        # *INCLUDE внутри секции узлов не поддерживаем узлово: включения читаются
        # отдельным проходом ниже
    for line in _include_lines(inp_path):
        sub = (inp_path.parent / line).resolve()
        if not sub.is_file():
            raise FileNotFoundError(f"{inp_path}: *INCLUDE: нет файла {sub}")
        sub_nodes = parse_nodes(sub, _seen | {key}, _depth + 1)
        for n, xyz3 in sub_nodes.items():
            if n in nodes and nodes[n] != xyz3:
                raise ValueError(f"{inp_path}: узел {n} переопределён в {sub}")
        nodes.update(sub_nodes)
    return nodes


def _include_lines(inp_path: Path) -> List[str]:
    """Цели файлов из карточек *INCLUDE верхнего уровня (INPUT= или след. строка)."""
    out: List[str] = []
    with Path(inp_path).open(encoding="utf-8-sig", errors="ignore") as f:
        lines = enumerate(f, 1)
        for _lineno, raw in lines:
            s = raw.strip()
            if not s.startswith("*") or s.startswith("**"):
                continue
            card = s.split(",")[0].strip().upper()
            if card != "*INCLUDE":
                continue
            target = None
            for part in s.split(",")[1:]:
                if part.strip().upper().startswith("INPUT="):
                    target = part.split("=", 1)[1].strip().strip('"').strip()
            if target is None:
                for _l2, raw2 in lines:
                    s2 = raw2.strip()
                    if not s2 or s2.startswith("**"):
                        continue
                    target = s2.strip('"')
                    break
            if target:
                out.append(target)
    return out


def audit_pbc(inp_path: Path, equations: Sequence[Equation]) -> PbcAuditResult:
    """P1/P2/P3 по геометрии: покрытие граней связями, зеркальность пар,
    дубликаты уравнений. N/A — если направления периодичности не найдены."""
    nodes = parse_nodes(inp_path)
    if not nodes:
        return PbcAuditResult("N/A", note=f"{inp_path}: в deck нет карточек *NODE — "
                                          "аудит геометрии невозможен")
    coords = {n: c for n, c in nodes.items()}
    directions: List[DirectionAudit] = []
    for axis in range(3):
        vals = [c[axis] for c in coords.values()]
        vmin, vmax = min(vals), max(vals)
        span = vmax - vmin
        if span <= 0:
            continue
        tol = span * FACE_TOL
        minus = {n for n, c in coords.items() if abs(c[axis] - vmin) <= tol}
        plus = {n for n, c in coords.items() if abs(c[axis] - vmax) <= tol}
        eq_idx = [i for i, eq in enumerate(equations)
                  if any(n in minus for n, _d, _c in eq.terms)
                  and any(n in plus for n, _d, _c in eq.terms)]
        if not eq_idx:
            continue
        covered = {n for i in eq_idx for n, _d, _c in equations[i].terms
                   if n in minus or n in plus}
        face_nodes = minus | plus
        n_sym = 0
        asym: List[int] = []
        for i in eq_idx:
            s_minus = [0.0, 0.0, 0.0]
            s_plus = [0.0, 0.0, 0.0]
            for n, d, c in equations[i].terms:
                if n in minus:
                    s_minus[d - 1] += c
                elif n in plus:
                    s_plus[d - 1] += c
            ok = all(abs(s_minus[k] + s_plus[k]) <= SYM_TOL * max(abs(s_minus[k]),
                                                                 abs(s_plus[k]), 1e-300)
                     for k in range(3) if s_minus[k] or s_plus[k])
            if ok:
                n_sym += 1
            else:
                asym.append(i)
        directions.append(DirectionAudit(
            axis=axis, n_face_nodes=len(face_nodes), n_covered=len(covered & face_nodes),
            coverage=len(covered & face_nodes) / len(face_nodes),
            uncovered=sorted(face_nodes - covered)[:20],
            n_equations=len(eq_idx), n_symmetric=n_sym, symmetry=n_sym / len(eq_idx),
            asymmetric_eqs=asym[:20]))

    # P3: дубликаты — одинаковые множества (узел, dof) в двух уравнениях
    seen: Dict[frozenset, int] = {}
    duplicates: List[Tuple[int, int]] = []
    for i, eq in enumerate(equations):
        key = frozenset((n, d) for n, d, _c in eq.terms)
        if key in seen:
            duplicates.append((seen[key], i))
        else:
            seen[key] = i

    if not directions:
        return PbcAuditResult("N/A", directions=[], duplicates=duplicates,
                              n_nodes=len(nodes),
                              note="направления периодичности не найдены: нет уравнений, "
                                   "связывающих противоположные экстремальные плоскости")
    ok = all(d.coverage == 1.0 and d.symmetry == 1.0 for d in directions) \
        and not duplicates
    return PbcAuditResult("PASS" if ok else "FAIL", directions=directions,
                          duplicates=duplicates, n_nodes=len(nodes))


def audit_summary(audit: PbcAuditResult) -> List[str]:
    """Строки сводки аудита для CLI и Markdown-отчёта."""
    if audit.verdict == "N/A":
        return [f"PBC-аudit: N/A — {audit.note}"]
    lines = [f"PBC-аudit: {audit.verdict}"]
    for d in audit.directions:
        cov = f"покрытие граней {d.n_covered}/{d.n_face_nodes}"
        if d.uncovered:
            cov += f" (без связей: {', '.join(map(str, d.uncovered))})"
        sym = f"симметрия пар {d.n_symmetric}/{d.n_equations}"
        if d.asymmetric_eqs:
            sym += f" (ур. {', '.join(map(str, d.asymmetric_eqs))})"
        lines.append(f"  ось {d.label}: {cov}, {sym}")
    if audit.duplicates:
        lines.append("  дубликаты уравнений: "
                     + ", ".join(f"{a}≡{b}" for a, b in audit.duplicates))
    else:
        lines.append("  дубликаты: 0")
    return lines
