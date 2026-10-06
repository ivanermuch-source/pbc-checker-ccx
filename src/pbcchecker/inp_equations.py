"""Разбор *EQUATION из .inp CalculiX.

Формат: строка "*EQUATION", следующая строка — число термов N, далее
3N значений "узел, dof, коэффициент" с произвольными переносами строк
(ccx пишет термы по 80 колонок). Пример из боевого прогона:

    *EQUATION
    2
    93, 3, 1.0, 49070, 3, -1.0

Строки с '**' — комментарии, имена карточек нечувствительны к регистру.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


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
    """Все *EQUATION deck'а. Уравнения без термов или с мусором — ошибка ValueError
    с номером строки, чтобы испорченный .inp не превращался в тихий PASS."""
    lines = Path(inp_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    equations: List[Equation] = []
    i, n = 0, len(lines)
    while i < n:
        stripped = lines[i].strip()
        i += 1
        if not stripped.startswith("*") or stripped.startswith("**"):
            continue
        if stripped.split(",")[0].strip().upper() != "*EQUATION":
            continue
        # следующая значимая строка — число термов
        while i < n and (not lines[i].strip() or lines[i].strip().startswith("**")):
            i += 1
        if i >= n:
            raise ValueError(f"{inp_path}: *EQUATION без строки с числом термов (строка {i})")
        try:
            n_terms = int(lines[i].split(",")[0].strip())
        except ValueError as e:
            raise ValueError(f"{inp_path}: число термов не целое: {lines[i]!r} (строка {i+1})") from e
        i += 1
        # собираем 3N числовых токенов с переносами между строками
        tokens: List[str] = []
        while i < n and len(tokens) < 3 * n_terms:
            s = lines[i].strip()
            i += 1
            if not s or s.startswith("**"):
                continue
            tokens.extend(s.split(","))
        if len(tokens) < 3 * n_terms:
            raise ValueError(
                f"{inp_path}: *EQUATION #{len(equations)}: термов {len(tokens)//3} из {n_terms}")
        terms = []
        for k in range(n_terms):
            node = int(tokens[3 * k]); dof = int(tokens[3 * k + 1]); coef = float(tokens[3 * k + 2])
            if not (1 <= dof <= 3):
                raise ValueError(
                    f"{inp_path}: *EQUATION #{len(equations)}: dof={dof} вне 1..3 "
                    "(инструмент проверяет поступательные DOF твёрдого тела)")
            terms.append((node, dof, coef))
        equations.append(Equation(len(equations), tuple(terms)))
    return equations
