"""Разбор *EQUATION из .inp CalculiX.

Формат: строка "*EQUATION", следующая строка — число термов N, далее
3N значений "узел, dof, коэффициент" с произвольными переносами строк
(ccx пишет термы по 80 колонок). Пример из боевого прогона:

    *EQUATION
    2
    93, 3, 1.0, 49070, 3, -1.0

Строки с '**' — комментарии, имена карточек нечувствительны к регистру.
Файл читается потоково, без загрузки в память (PBC-deck'и бывают сотни MB).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# фортран-экспонента без 'E' ('1.0-100') вставляется перед знаком степени
_EXP_FIX = re.compile(r"(?<=[\d.])([+-])(?=\d+$)")


def to_float(tok: str) -> float:
    """float() с поддержкой фортран-экспонент: '1.0-100', '2.5+3', '1.0D+00'."""
    s = tok.strip().replace("D", "E").replace("d", "E")
    try:
        return float(s)
    except ValueError:
        return float(_EXP_FIX.sub(r"E\1", s))


@dataclass(frozen=True)
class Equation:
    index: int                 # порядковый номер в deck (с 0)
    terms: Tuple[Tuple[int, int, float], ...]  # (узел, dof 1-based, коэф.)

    def render(self) -> str:
        parts = []
        for node, dof, coef in self.terms:
            sign = "+" if coef >= 0 else "-"
            parts.append(f"{sign} {abs(coef):.6g}*u({node},{dof})")
        s = " ".join(parts).lstrip("+ ")
        return f"[{self.index}] {s} = 0"


def parse_equations(inp_path: Path) -> List[Equation]:
    """Все *EQUATION deck'а. Уравнения без термов, с мусором или оборванные
    следующей *-карточкой — ValueError с номером строки, чтобы испорченный
    .inp не превращался в тихий PASS."""
    equations: List[Equation] = []
    with Path(inp_path).open(encoding="utf-8", errors="ignore") as f:
        lines = enumerate(f, 1)
        for lineno, raw in lines:
            s = raw.strip()
            if not s.startswith("*") or s.startswith("**"):
                continue
            if s.split(",")[0].strip().upper() != "*EQUATION":
                continue
            equations.append(_parse_one(inp_path, lines, lineno, len(equations)))
    return equations


def _parse_one(inp_path: Path, lines, card_lineno: int, eq_idx: int) -> Equation:
    """Одно уравнение; lines — общий итератор (номер строки, строка), уже
    позиционирован после карточки *EQUATION в строке card_lineno."""
    # следующая значимая строка — число термов
    n_terms, terms_lineno = None, card_lineno
    for lineno, raw in lines:
        s = raw.strip()
        if not s or s.startswith("**"):
            continue
        try:
            n_terms = int(s.split(",")[0])
        except ValueError as e:
            raise ValueError(f"{inp_path}: *EQUATION #{eq_idx}: число термов не целое: "
                             f"{s!r} (строка {lineno})") from e
        terms_lineno = lineno
        break
    if n_terms is None:
        raise ValueError(f"{inp_path}: *EQUATION #{eq_idx} без строки с числом термов "
                         f"(карточка в строке {card_lineno}, конец файла)")
    if n_terms < 1:
        raise ValueError(f"{inp_path}: *EQUATION #{eq_idx}: число термов {n_terms} < 1 "
                         f"(строка {terms_lineno})")
    # собираем 3N числовых токенов с переносами между строками;
    # *-карточка до набора 3N токенов = оборванное уравнение, а не данные
    tokens: List[str] = []
    for lineno, raw in lines:
        s = raw.strip()
        if not s or s.startswith("**"):
            continue
        if s.startswith("*"):
            raise ValueError(f"{inp_path}: *EQUATION #{eq_idx}: термов {len(tokens) // 3} "
                             f"из {n_terms}, а в строке {lineno} началась карточка "
                             f"{s.split(',')[0]}")
        tokens.extend(t.strip() for t in s.split(","))
        if len(tokens) >= 3 * n_terms:
            break
    if len(tokens) < 3 * n_terms:
        raise ValueError(f"{inp_path}: *EQUATION #{eq_idx}: термов {len(tokens) // 3} "
                         f"из {n_terms} (карточка в строке {card_lineno}, конец файла)")
    terms = []
    for k in range(n_terms):
        node_s, dof_s, coef_s = tokens[3 * k:3 * k + 3]
        try:
            node = int(node_s)
            dof = int(dof_s)
            coef = to_float(coef_s)
        except ValueError as e:
            raise ValueError(f"{inp_path}: *EQUATION #{eq_idx}: терм #{k + 1} "
                             f"«{node_s},{dof_s},{coef_s}» не разбирается "
                             f"(карточка в строке {card_lineno})") from e
        if not (1 <= dof <= 3):
            raise ValueError(
                f"{inp_path}: *EQUATION #{eq_idx}: dof={dof} вне 1..3 "
                "(инструмент проверяет поступательные DOF твёрдого тела)")
        terms.append((node, dof, coef))
    return Equation(eq_idx, tuple(terms))
