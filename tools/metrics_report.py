"""Метрики дополнительной валидации и отчётные графики «порог vs результат».

M-серия — oracle-арифметика (MMS-подход): невязка чекера сравнивается с
эталоном math.fsum и аналитическими инвариантами; единственный источник
ответа «числа верны», а не только «гейт чувствителен» (мутации дают второе).

F-серия — робастность CLI на мусорных входах: контракт кодов выхода
0/1/2, детерминизм и время; «0 crash» доказывается прогоном N битых пар
.inp/.frd через pbcchecker.cli.main в процессе.

Запуск:  python tools/metrics_report.py [--out docs/metrics] [--n-fuzz 1000]
Выход:   metrics.json + 4 графика (fig_m_compass, fig_f_robustness,
         fig_residual_map, fig_c_calibration) + сводный дашборд
         fig_summary_dashboard — все из фактических данных отчёта.
         Пороги держат tests/test_metrics.py.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from pbcchecker import __version__
from pbcchecker.cli import main as cli_main
from pbcchecker.frd_disp import DispBlock, iter_disp_blocks
from pbcchecker.gate import check_job
from pbcchecker.inp_equations import Equation, parse_equations
from pbcchecker.residual import compute_residuals, term_arrays

EPS = 2.220446049250313e-16          # eps float64
TINY = 1e-300                        # защита деления на 0 в нормировках

# Пороги PASS; в JSON идут рядом с измерением (график: порог vs результат).
THRESHOLDS = {
    "M1_fsum_headroom": 0.5,      # err/(n+2)·eps·Σ|a·u| против точного fsum
    "M2_file_headroom": 0.5,      # то же через полный файловый конвейер E12.5
    "M3_scale_dev": 1e-12,        # |Δrel| при u→α·u, a→β·a (инвариант нормировки)
    "M4_worst_eq_hitrate": 1.0,   # worst_eq обязан указать на подменённый терм
    "F1_contract_violations": 0,  # код ∉ {0,1,2} / незcaught исключение
    "F2_exit_code_accuracy": 1.0, # доля верных кодов на golden-наборе
    "F3_nondeterminism": 0,       # повтор ≠ (код + stdout)
    "F4_max_seconds": 5.0,        # один мусорный прогон CLI
    "C1_separation_min": 10.0,    # S = tol / max_rel на хороших прогонах
    "C2_detection_rate": 1.0,     # инжект-дефекты пойманы (FAIL/exit≠0)
    "C2_margin_min": 10.0,        # min по дефектам: rel_defect / tol
}

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- M-серия

def _synth(rng: random.Random, n_eq: int, n_nodes: int) -> Tuple[List[Equation], np.ndarray]:
    """Синтетические уравнения и поле u: узлы 1..n_nodes, dof 1..3, коэф ~U(0.1, 2)."""
    eqs = []
    for i in range(n_eq):
        n_t = rng.randint(2, 50)
        terms = tuple((rng.randint(1, n_nodes), rng.randint(1, 3),
                       rng.choice((-1.0, 1.0)) * rng.uniform(0.1, 2.0))
                      for _ in range(n_t))
        eqs.append(Equation(index=i, terms=terms))
    u = np.array([[rng.gauss(0.0, 1.0) for _ in range(3)] for _ in range(n_nodes)])
    return eqs, u


def _fsum_residuals(eqs: List[Equation], u: np.ndarray) -> np.ndarray:
    """Эталонные невязки: точное суммирование fsum, без накопления reduceat."""
    return np.array([
        math.fsum(c * u[n - 1, d - 1] for n, d, c in eq.terms) for eq in eqs])


def m1_oracle_fsum(rng: random.Random, n_eq: int = 200, n_nodes: int = 400) -> float:
    """headroom = max err / ((n_terms+2)·eps·Σ|a·u|) — запас до границы
    накопления последовательного суммирования; PASS ≤ 0.5."""
    eqs, u = _synth(rng, n_eq, n_nodes)
    nodes, dofs, coefs, offs = term_arrays(eqs)
    au = coefs * u[nodes - 1, dofs]
    r_chk = np.add.reduceat(au, offs[:-1])
    r_ref = _fsum_residuals(eqs, u)
    counts = np.array([len(e.terms) for e in eqs], dtype=np.float64)
    sums_abs = np.add.reduceat(np.abs(au), offs[:-1])
    bound = (2.0 * counts + 2.0) * EPS * sums_abs   # γ_2n: умножения + суммирование
    return float((np.abs(r_chk - r_ref) / np.maximum(bound, TINY)).max())


def m2_roundtrip(tmp: Path, rng: random.Random) -> float:
    """Файловый oracle: .inp + .frd пишутся нами, u округляется тем же E12.5;
    невязка чекера (полный конвейер парсеров) обязана совпасть с fsum по
    записанным строкам до границы накопления."""
    n_nodes, n_eq = 60, 30
    eqs, u_raw = _synth(rng, n_eq, n_nodes)
    u = np.vectorize(lambda v: float(f"{v:12.5E}"))(u_raw)   # что лежит в файле

    inp = tmp / "m2.inp"
    lines = ["*HEADING"]
    for eq in eqs:
        lines.append("*EQUATION")
        lines.append(str(len(eq.terms)))
        lines.append(", ".join(f"{n}, {d}, {c!r}" for n, d, c in eq.terms))
    inp.write_text("\n".join(lines) + "\n", encoding="ascii")

    frd = tmp / "m2.frd"
    rows = ["    1C", "    1UVERSION           Version 2.20",
            "  100CL  1  0.1000000E-01", "    -4  DISP        4    1",
            "    -5  D1          1    2    3"]
    for i in range(n_nodes):
        vx, vy, vz = u[i]
        rows.append(f" -1{i + 1:10d}{vx:12.5E}{vy:12.5E}{vz:12.5E}")
    rows.append("    -3")
    frd.write_text("\n".join(rows) + "\n", encoding="ascii")

    res = compute_residuals(parse_equations(inp), list(iter_disp_blocks(frd)))
    blk = list(iter_disp_blocks(frd))[0]
    eqs_read = parse_equations(inp)
    denom = max(float(np.abs(blk.u).max()), TINY)
    r_chk_rel = np.asarray(res.eq_max_rel)          # per-eq rel из самого чекера
    r_ref_rel = np.abs(_fsum_residuals(eqs_read, blk.u)) / denom
    nodes, dofs, coefs, offs = term_arrays(eqs_read)
    so = np.argsort(blk.nodes, kind="stable")
    mapped = so[np.searchsorted(blk.nodes[so], nodes)]
    counts = np.array([len(e.terms) for e in eqs_read], dtype=np.float64)
    sums_abs = np.add.reduceat(np.abs(coefs * blk.u[mapped, dofs]), offs[:-1])
    bound_rel = (2.0 * counts + 2.0) * EPS * sums_abs / denom
    return float((np.abs(r_chk_rel - r_ref_rel) / np.maximum(bound_rel, TINY)).max())


def m3_scale_invariance(rng: random.Random) -> float:
    """rel-невязка инвариантна к u→α·u (нормировка max|u|) и лине́йна по a→β·a
    с точностью округления; отклонение — баг нормировки/суммирования.
    Возврат: max отклонение / base."""
    eqs, u = _synth(rng, 120, 250)
    nodes_arr = np.arange(1, u.shape[0] + 1, dtype=np.int32)

    def rel(eqs_, u_):
        blk = DispBlock(step=1, label="s", nodes=nodes_arr, u=u_)
        return compute_residuals(eqs_, [blk]).steps[0].max_rel_residual

    base = rel(eqs, u)
    dev = 0.0
    for a in np.logspace(-6, 6, 13):
        dev = max(dev, abs(rel(eqs, u * float(a)) - base) / base)
    for b in (1e-6, 1e-3, 1.0, 1e3, 1e6):
        eqs_b = [Equation(e.index, tuple((n, d, c * b) for n, d, c in e.terms))
                 for e in eqs]
        # линейность по a: rel(βa) = β·rel(a) с точностью округления,
        # ошибка масштабирования ~eps·β — потому нормируем на β·base
        dev = max(dev, abs(rel(eqs_b, u) - b * base) / (b * base))
    return dev


def m4_worst_eq(trials: int = 60) -> float:
    """Подмена терма делает уравнение j заведомо худшим — worst_eq обязан
    указать на него. Возврат: доля попаданий."""
    hits = 0
    for t in range(trials):
        rng = random.Random(7000 + t)
        eqs, u = _synth(rng, 50, 200)
        nodes_arr = np.arange(1, u.shape[0] + 1, dtype=np.int32)
        r_base = np.abs(_fsum_residuals(eqs, u))
        j = rng.randrange(len(eqs))
        big = int(np.argmax(np.abs(u[:, 0])))
        k = (10.0 * float(r_base.max()) + 1.0) / max(abs(float(u[big, 0])), TINY)
        terms = list(eqs[j].terms)
        terms[0] = (big + 1, 1, float(k))
        eqs[j] = Equation(j, tuple(terms))
        blk = DispBlock(step=1, label="s", nodes=nodes_arr, u=u)
        if compute_residuals(eqs, [blk]).steps[0].worst_eq == j:
            hits += 1
    return hits / trials


# ---------------------------------------------------------------- F-серия

def _run_cli(inp: Path, frd: Path) -> Tuple[int, str, float]:
    out, err = io.StringIO(), io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli_main([str(inp), "--frd", str(frd)])
        except SystemExit as e:            # ap.error / --version / --help
            code = int(e.code or 0)
    return code, out.getvalue(), time.perf_counter() - t0


def _valid_pair(tmp: Path, name: str, fail: bool = False) -> Tuple[Path, Path]:
    """Маленькая валидная пара: PASS-вариант — связи выполнены до E12.5,
    FAIL-вариант — одна связь нарушена на 100 %."""
    inp = tmp / f"{name}.inp"
    frd = tmp / f"{name}.frd"
    rows = ["    1C", "    1UVERSION           Version 2.20",
            "  100CL  1  0.1000000E-01", "    -4  DISP        4    1",
            "    -5  D1          1    2    3"]
    u = {1: (1.0e-3, 0.0, 0.0), 2: (1.0e-3, 0.0, 0.0), 3: (0.0, 2.0e-3, 0.0),
         4: (0.0, 2.0e-3, 0.0)}
    if fail:
        u[2] = (0.0, 0.0, 0.0)
    for n, (vx, vy, vz) in u.items():
        rows.append(f" -1{n:10d}{vx:12.5E}{vy:12.5E}{vz:12.5E}")
    rows.append("    -3")
    frd.write_text("\n".join(rows) + "\n", encoding="ascii")
    inp.write_text(
        "*HEADING\n*EQUATION\n2\n1, 1, 1.0, 2, 1, -1.0\n"
        "*EQUATION\n2\n3, 2, 1.0, 4, 2, -1.0\n", encoding="ascii")
    return inp, frd


_FUZZ_CATEGORIES = [
    "inp_truncate", "frd_truncate", "inp_bytes", "frd_bytes",
    "inp_struct", "frd_struct", "empty", "swapped",
]

_STRUCT_INP = [
    b"*EQUATION\n0\n", b"*EQUATION\n-2\n1, 1, 1.0\n",
    b"*EQUATION\nabc\n", b"*EQUATION\n1\n1, 4, 1.0\n",
    b"*EQUATION\n2\n1, 1, 1.0\n",           # оборвано после первого терма
    b"*EQUATION\n2\n1, 1, xyz, 2, 1, -1.0\n",
    b"*EQUATION\n2\n0, 1, 1.0, 2, 1, -1.0\n",
    # вторая связь под той же карточкой — раньше парсер молча терял хвост
    b"*EQUATION\n2\n1, 1, 1.0, 2, 1, -1.0\n2\n3, 1, 1.0, 4, 1, -1.0\n",
]
_STRUCT_FRD = [
    b"    -4  DISP        4    1\n    -5  D1\n",                       # нет -3
    b"    -4  DISP        4    1\n    -3\n",                            # пустой блок
    b" -1xx      11 1.0E-04 0.0E+00 0.0E+00\n    -3\n",                 # мусор в I10
    b"  100CL  1  0.1E-01\n    -3\n",                                   # нет -4
]


def _fuzz_case(rng: random.Random, tmp: Path, idx: int
               ) -> Tuple[str, Path, Path]:
    """Один мусорный кейс: (категория, inp, frd)."""
    cat = _FUZZ_CATEGORIES[idx % len(_FUZZ_CATEGORIES)]
    base_inp, base_frd = _valid_pair(tmp, "fz")
    raw_inp, raw_frd = base_inp.read_bytes(), base_frd.read_bytes()

    if cat == "inp_truncate":
        raw_inp = raw_inp[: int(len(raw_inp) * rng.uniform(0.0, 1.0))]
    elif cat == "frd_truncate":
        raw_frd = raw_frd[: int(len(raw_frd) * rng.uniform(0.0, 1.0))]
    elif cat == "inp_bytes":
        b = bytearray(raw_inp)
        for _ in range(rng.randint(1, 20)):
            b[rng.randrange(len(b))] = rng.randrange(32, 127)
        raw_inp = bytes(b)
    elif cat == "frd_bytes":
        b = bytearray(raw_frd)
        for _ in range(rng.randint(1, 20)):
            b[rng.randrange(len(b))] = rng.randrange(32, 127)
        raw_frd = bytes(b)
    elif cat == "inp_struct":
        raw_inp = rng.choice(_STRUCT_INP)
    elif cat == "frd_struct":
        raw_frd = rng.choice(_STRUCT_FRD)
    elif cat == "empty":
        raw_inp = b"" if rng.random() < 0.5 else b"   \n\t\n"
        raw_frd = b"" if rng.random() < 0.5 else b"\n"
    elif cat == "swapped":
        raw_inp, raw_frd = raw_frd, raw_inp

    inp, frd = tmp / f"f{idx}.inp", tmp / f"f{idx}.frd"
    inp.write_bytes(raw_inp)
    frd.write_bytes(raw_frd)
    return cat, inp, frd


def measure_f(n_fuzz: int = 1000, seed: int = 42) -> Dict:
    """F1–F4: контракт кодов, точность golden-кодов, детерминизм, время."""
    rng = random.Random(seed)
    violations: Dict[str, int] = {c: 0 for c in _FUZZ_CATEGORIES}
    violations["missing_frd"] = 0
    runs: Dict[str, int] = {c: 0 for c in violations}
    max_dt = 0.0
    nondet = 0
    times_ms: List[float] = []
    with tempfile.TemporaryDirectory(prefix="pbc_fuzz_") as td:
        tmp = Path(td)
        for i in range(n_fuzz):
            cat, inp, frd = _fuzz_case(rng, tmp, i)
            code, out, dt = _run_cli(inp, frd)
            max_dt = max(max_dt, dt)
            times_ms.append(round(dt * 1000.0, 3))
            runs[cat] += 1
            if code not in (0, 1, 2):     # исключение из main — тоже нарушение
                violations[cat] += 1
            if i % 20 == 0:               # F3: подвыборка на детерминизм
                code2, out2, _ = _run_cli(inp, frd)
                if (code, out) != (code2, out2):
                    nondet += 1
        # missing_frd: путь указывает в никуда
        ghost = tmp / "ghost.frd"
        for i in range(max(5, n_fuzz // 100)):
            _inp, _frd = _valid_pair(tmp, "mf")
            code, out, dt = _run_cli(_inp, ghost)
            max_dt = max(max_dt, dt)
            runs["missing_frd"] += 1
            if code != 2:
                violations["missing_frd"] += 1

        # F2: golden-набор с известными кодами
        good_inp, good_frd = _valid_pair(tmp, "good")
        fail_inp, fail_frd = _valid_pair(tmp, "fail", fail=True)
        na = tmp / "na.inp"
        na.write_text("*HEADING\n*NODE\n1, 0.0, 0.0, 0.0\n", encoding="ascii")
        cases = [(good_inp, good_frd, 0), (fail_inp, fail_frd, 1),
                 # N/A без *EQUATION возвращается раньше обращения к .frd —
                 # даже несуществующий путь даёт 0 (контракт check_job)
                 (na, good_frd, 0), (na, na.with_suffix(".frd"), 0)]
        for i, tpl in enumerate(_STRUCT_INP):
            p = tmp / f"s{i}.inp"
            p.write_bytes(tpl)
            cases.append((p, good_frd, 2))
        for i, tpl in enumerate(_STRUCT_FRD):
            p = tmp / f"t{i}.frd"
            p.write_bytes(tpl)
            cases.append((good_inp, p, 2))
        ok = sum(1 for c, f, want in cases if _run_cli(c, f)[0] == want)
        accuracy = ok / len(cases)

    return {
        "n_fuzz": n_fuzz, "categories": {
            c: {"runs": runs[c], "violations": violations[c]} for c in runs},
        "F1_contract_violations": sum(violations.values()),
        "F2_exit_code_accuracy": accuracy,
        "F3_nondeterminism": nondet,
        "F4_max_seconds": max_dt,
        "F4_times_ms": times_ms,
    }


# ---------------------------------------------------------------- C-серия

def _pbc_case(tmp: Path, name: str, n_pairs: int, rng: random.Random, *,
              shift_eq=None, swap_eq=None, drop_eq=None) -> Tuple[Path, Path]:
    """Периодическая пара .inp/.frd по построению: пары узлов (i, n+i) по dof 1,
    u(n+i) = u(i) + шум квантования E12.5 (~пол). Инжекты дефектов:
    shift_eq — сдвиг u одного узла на 3e-2·max|u|; swap_eq — подмена узла
    терма на узел другой пары (разрыв связи); drop_eq — удалённое уравнение
    (ловится только P-аудитом)."""
    scale = 1.0e-4
    inp_lines = ["*NODE"]
    for i in range(1, n_pairs + 1):
        inp_lines.append(f"{i}, 0.0, 0.5, 0.0")
        inp_lines.append(f"{n_pairs + i}, 1.0, 0.5, 0.0")
    for i in range(1, n_pairs + 1):
        if drop_eq is not None and i == drop_eq:
            continue
        b = n_pairs + i
        if swap_eq is not None and i == swap_eq:
            b = n_pairs + (1 if i > n_pairs // 2 else n_pairs)   # узел чужой пары
        inp_lines += ["*EQUATION", "2", f"{i}, 1, 1.0, {b}, 1, -1.0"]
    inp = tmp / f"{name}.inp"
    inp.write_text("\n".join(inp_lines) + "\n", encoding="ascii")

    out = ["    1C", "    1UVERSION           Version 2.20",
           "  100CL  1  0.1000000E-01", "    -4  DISP        4    1",
           "    -5  D1          1    2    3"]
    for i in range(1, n_pairs + 1):
        ux = scale * (0.5 + i / n_pairs)
        for n, val in ((i, ux), (n_pairs + i, ux + rng.uniform(0.0, 1e-9))):
            if shift_eq is not None and n == n_pairs + shift_eq:
                val += 3.0e-2 * scale
            out.append(f" -1{n:10d}{val:12.5E}{'0.00000E+00':>12}{'0.00000E+00':>12}")
    out.append("    -3")
    frd = tmp / f"{name}.frd"
    frd.write_text("\n".join(out) + "\n", encoding="ascii")
    return inp, frd


def measure_c(tmp: Path, tol: float = 1e-3) -> Dict:
    """C1 separation на 3 «сетках» и C2 ловимость инжект-дефектов.
    Демонстрирует калибровку порога: хорошие прогоны у пола чувствительности,
    дефекты — на порядки выше гейта; удалённая связь ловится P-аудитом."""
    rng = random.Random(11)
    separations = {}
    for n in (25, 100, 400):
        inp, frd = _pbc_case(tmp, f"c1_{n}", n, rng)
        v = check_job(inp, frd, tol=tol)
        separations[n] = tol / max(v.max_rel_residual, 1e-300)

    n = 200
    cases = {}                                    # имя → (rel_defect, пойман ли)
    inp, frd = _pbc_case(tmp, "c2_shift", n, rng, shift_eq=n // 3)
    v = check_job(inp, frd, tol=tol)
    cases["сдвиг узла 3e-2·max|u|"] = (v.max_rel_residual / tol, not v.passed)
    inp, frd = _pbc_case(tmp, "c2_swap", n, rng, swap_eq=n // 2)
    v = check_job(inp, frd, tol=tol)
    cases["подмена узла (разрыв)"] = (v.max_rel_residual / tol, not v.passed)
    inp, frd = _pbc_case(tmp, "c2_drop", n, rng, drop_eq=n // 4)
    v_plain = check_job(inp, frd, tol=tol)        # невязкой не виден
    v_audit = check_job(inp, frd, tol=tol, pbc_audit=True)
    cases["удалённая связь (--pbc-audit)"] = (None,
                                              v_audit.audit.verdict == "FAIL"
                                              and not v_audit.passed)
    detection = sum(1 for _k, (_m, caught) in cases.items() if caught) / len(cases)
    margins = [m for _k, (m, _c) in cases.items() if m is not None]
    return {
        "C1_separations": {str(k): round(s, 1) for k, s in separations.items()},
        "C1_separation_min": min(separations.values()),
        "C2_cases": {k: {"margin": m, "caught": c} for k, (m, c) in cases.items()},
        "C2_detection_rate": detection,
        "C2_margin_min": min(margins),
        "C2_note": "удалённая связь даёт чистую невязку "
                   f"(max_rel={v_plain.max_rel_residual:.1e}) и ловится только "
                   "P-аудитом (--pbc-audit)",
    }



# ---------------------------------------------------------------- прогон

def run_all(n_fuzz: int = 1000, seed: int = 42) -> Dict:
    rng = random.Random(seed)
    m = {
        "M1_fsum_headroom": m1_oracle_fsum(random.Random(seed + 1)),
        "M3_scale_dev": m3_scale_invariance(random.Random(seed + 2)),
        "M4_worst_eq_hitrate": m4_worst_eq(),
    }
    with tempfile.TemporaryDirectory(prefix="pbc_m2_") as td:
        m["M2_file_headroom"] = m2_roundtrip(Path(td), random.Random(seed + 3))
    m = dict(sorted(m.items()))
    res = {"tool_version": __version__, "generated": datetime.now().isoformat(timespec="seconds"),
           "thresholds": THRESHOLDS, "M": m, "F": measure_f(n_fuzz, seed)}
    with tempfile.TemporaryDirectory(prefix="pbc_c_") as td:
        res["C"] = measure_c(Path(td))
    res["summary"] = {
        "M_pass": all(
            v <= THRESHOLDS[k] if k != "M4_worst_eq_hitrate" else v >= THRESHOLDS[k]
            for k, v in m.items()),
        "F_pass": (res["F"]["F1_contract_violations"] == 0
                   and res["F"]["F2_exit_code_accuracy"] == 1.0
                   and res["F"]["F3_nondeterminism"] == 0
                   and res["F"]["F4_max_seconds"] < THRESHOLDS["F4_max_seconds"]),
        "C_pass": (res["C"]["C1_separation_min"] >= THRESHOLDS["C1_separation_min"]
                   and res["C"]["C2_detection_rate"] == 1.0
                   and res["C"]["C2_margin_min"] >= THRESHOLDS["C2_margin_min"]),
    }
    res["summary"]["all_pass"] = all(res["summary"].values())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="docs/metrics", help="каталог артефактов")
    ap.add_argument("--n-fuzz", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    report = run_all(args.n_fuzz, args.seed)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # графики опциональны: измерения и metrics.json пишутся и без matplotlib
    # (extras report = pip install -e ".[report]"); отсутствие — не ошибка
    try:
        from metrics_plots import make_plots          # графики — отдельный модуль
        make_plots(report, out_dir)
        from summary_dashboard import make_dashboard  # сводное изображение-результат
        make_dashboard(report, out_dir)
    except ImportError:
        print("ПРИМЕЧАНИЕ: matplotlib не установлен — графики пропущены "
              '(pip install -e ".[report"]); metrics.json записан полностью.')

    ok = report["summary"]["all_pass"]
    print(json.dumps(report["M"], indent=2))
    print(json.dumps({k: v for k, v in report["F"].items()
                      if k not in ("categories", "F4_times_ms")}, indent=2))
    print(json.dumps({k: v for k, v in report["C"].items()
                      if k not in ("C2_cases",)}, indent=2))
    print(f"{'PASS' if ok else 'FAIL'}; артефакты: {out_dir}/")
    raise SystemExit(0 if ok else 1)
