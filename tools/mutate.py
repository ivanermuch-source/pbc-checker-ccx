"""Мутационное тестирование pbcchecker (mutmut на Windows не работает — свой движок).

Мутант — одно текстовое изменение одной строки кода: оператор сравнения,
'and'/'or', снятие 'not', +<->-, числовая константа +1, присваивание -> pass.
AST используется только чтобы отобрать строки кода (без докстрингов/комментариев);
строки с кавычками не мутируются (не ломаем сообщения об ошибках).

Мутант пишется во временную копию пакета, прогоняется pytest:
- тесты упали/зависли => мутант УБИТ;
- тесты прошли => ВЫЖИЛ — дыра в тестах (или эквивалентный мутант — руками).

Запуск:  python tools/mutate.py [фильтр-подстрока-модуля]
Выход:   сводка + tools/mutants_survived.txt

Известные эквивалентные мутанты (проверены вручную, добивать тестами не нужно):
- residual.py np.where(..., -1) -> -2: годится любой отрицательный sentinel;
- residual.py 1e-300 -> 2e-300: 0/любое = 0 (случай всех u=0);
- frd_disp.py probe 1<<16, reshape(-1,3) -> (-2,3): numpy 2.x любую отрицательную
  размерность считает «неизвестной»; начальные cur_* = [] и reset-ы в emit() —
  мёртвые инициализации для корректных потоков; and->or в in_disp/cur_nodes —
  недостижимые состояния; line[3:14] вместо [3:13] — отрабатывает split-fallback;
- inp_equations.py tokens[3k:4k+3] — клэмп среза при ровно 3N токенов;
  init last_tok_lineno — перезаписывается в цикле.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "pbcchecker"
MODULES = ["cli.py", "gate.py", "report.py", "residual.py", "frd_disp.py", "inp_equations.py"]

NUM_RE = re.compile(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b")
CMP_SWAPS = [("<=", "<"), (">=", ">"), ("==", "!="), ("!=", "=="),
             (" is not ", " is "), (" not in ", " in ")]
CMP_SWAPS2 = [("<", "<="), (">", ">="), (" is ", " is not "), (" in ", " not in ")]
ASSIGN_RE = re.compile(r"^(\s*)[\w.\[\]\"']+(?:\s*,\s*[\w.\[\]\"']+)*\s*(?::[^=]+)?=(?!=)")


def code_lines(src: str) -> set:
    """Номера строк с кодом (не докстринги, не комментарии)."""
    tree = ast.parse(src)
    doc = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                d = body[0].value
                doc.update(range(d.lineno, d.end_lineno + 1))
    lines = set()
    for node in ast.walk(tree):
        if hasattr(node, "lineno") and not isinstance(node, ast.Constant):
            end = getattr(node, "end_lineno", node.lineno)
            for ln in range(node.lineno, end + 1):
                lines.add(ln)
    return lines - doc


def mutate_line(line: str):
    """Все одноточные мутации строки: (описание, новая строка)."""
    if line.strip().startswith("#"):
        return []                   # комментарий целиком
    if "'" in line or '"' in line:
        return []                   # литералы/сообщения не трогаем
    line = line.split("#", 1)[0].rstrip()      # отрезать висячий комментарий
    if not line:
        return []
    out = []
    for old, new in CMP_SWAPS + CMP_SWAPS2:
        if old in line:
            out.append((f"{old.strip()}->{new.strip()}", line.replace(old, new, 1)))
    if " and " in line:
        out.append(("and->or", line.replace(" and ", " or ", 1)))
    if " or " in line:
        out.append(("or->and", line.replace(" or ", " and ", 1)))
    m = re.search(r"\bnot\s", line)
    if m and "is not" not in line and "not in" not in line:
        out.append(("снять not", line[:m.start()] + line[m.end():]))
    for op in ("+", "-"):
        m = re.search(rf"(?<![\w.]){re.escape(op)}(?![\w=+*-])", line)
        if m:
            flip = "-" if op == "+" else "+"
            out.append((f"{op}->{flip}",
                        line[:m.start()] + flip + line[m.end():]))
    # числовые константы: bump каждого числа по очереди
    for i, m in enumerate(NUM_RE.finditer(line)):
        old = m.group(0)
        out.append((f"{old}->{_bump(old)}", line[:m.start()] + _bump(old) + line[m.end():]))
    m = ASSIGN_RE.match(line)
    if m and line.strip() != "pass":
        out.append(("присваивание->pass", m.group(1) + "pass"))
    return out


def _bump(tok: str) -> str:
    """'13'->'14', '5e-06'->'6e-06', '0.5'->'0.6' — сохранить форму записи."""
    m = re.match(r"(\d+)(?:\.(\d+))?([eE][+-]?\d+)?$", tok)
    head, frac, exp = m.group(1), m.group(2), m.group(3) or ""
    if frac is None:
        return f"{int(head) + 1}{exp}"
    if set(frac) <= {"0"}:
        frac = "1" + frac[1:]                       # .00 -> .10
    else:
        frac = frac[:-1] + str(int(frac[-1]) + 1)   # .5 -> .6
    return f"{head}.{frac}{exp}"


def run_tests(timeout: float = 90.0) -> str:
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-x", "--no-header",
             "-p", "no:cacheprovider", "-m", "not e2e", "tests"],
            cwd=ROOT, capture_output=True, timeout=timeout, text=True)
    except subprocess.TimeoutExpired:
        return "timeout"
    return "pass" if r.returncode == 0 else "fail"


def main():
    only = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else ""
    mods = [m for m in MODULES if only in m]
    if not mods:
        sys.exit(f"нет модулей по фильтру {only!r}; доступно: {', '.join(MODULES)}")
    killed = survived = 0
    survivors = []
    t0 = time.time()
    # мутация in-place: editable-установка (PEP 660 finder) обходит PYTHONPATH,
    # временную копию pytest не увидит — мутируем настоящий src с восстановлением
    backups = {m: (SRC / m).read_text(encoding="utf-8") for m in mods}
    try:
        if run_tests() != "pass":
            sys.exit("БАЗОВЫЙ ПРОГОН НЕ ПРОШЁЛ — мутации бессмысленны")
        print("база проходит, начинаю мутации", flush=True)
        for mod in mods:
            src_lines = backups[mod].splitlines()
            clines = code_lines(backups[mod])
            n = 0
            for ln in sorted(clines):
                for desc, new_line in mutate_line(src_lines[ln - 1]):
                    if new_line == src_lines[ln - 1] or not new_line.strip():
                        continue
                    n += 1
                    (SRC / mod).write_text(
                        "\n".join(src_lines[:ln - 1] + [new_line] + src_lines[ln:]),
                        encoding="utf-8")
                    v = run_tests()
                    if v == "pass":
                        survived += 1
                        survivors.append(f"{mod}:{ln}: {desc}: {src_lines[ln - 1].strip()}")
                        print(f"  ВЫЖИЛ {mod}:{ln} {desc}", flush=True)
                    else:
                        killed += 1
            (SRC / mod).write_text(backups[mod], encoding="utf-8")
            print(f"{mod}: {n} мутантов, всего убито {killed} / выжило {survived}", flush=True)
    finally:
        for m, text in backups.items():          # гарантированное восстановление
            (SRC / m).write_text(text, encoding="utf-8")
    (ROOT / "tools" / "mutants_survived.txt").write_text(
        "\n".join(survivors) + "\n", encoding="utf-8")
    total = killed + survived
    print(f"\nИтого {total}: убито {killed}, выжило {survived} "
          f"({100 * killed / max(total, 1):.0f}%), {time.time() - t0:.0f} c")
    print("Выжившие: tools/mutants_survived.txt")


if __name__ == "__main__":
    main()
