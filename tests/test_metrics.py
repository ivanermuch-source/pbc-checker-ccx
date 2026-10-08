"""Пороги метрик дополнительной валидации (M/F-серии) в CI.

Измерители живут в tools/metrics_report.py — публичный инструмент репозитория
(не пакет); тесты держат те же пороги, что отчёт docs/metrics/.
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from metrics_report import (  # noqa: E402
    THRESHOLDS,
    m1_oracle_fsum,
    m2_roundtrip,
    m3_scale_invariance,
    m4_worst_eq,
    measure_f,
)


def test_M1_oracle_fsum():
    assert m1_oracle_fsum(random.Random(101)) <= THRESHOLDS["M1_fsum_headroom"]


def test_M2_file_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        headroom = m2_roundtrip(Path(td), random.Random(102))
    assert headroom <= THRESHOLDS["M2_file_headroom"]


def test_M3_scale_invariance():
    assert m3_scale_invariance(random.Random(103)) <= THRESHOLDS["M3_scale_dev"]


def test_M4_worst_eq():
    assert m4_worst_eq(trials=30) >= THRESHOLDS["M4_worst_eq_hitrate"]


def test_F_series_contract():
    f = measure_f(n_fuzz=240, seed=7)
    assert f["F1_contract_violations"] == 0
    assert f["F2_exit_code_accuracy"] == 1.0
    assert f["F3_nondeterminism"] == 0
    assert f["F4_max_seconds"] < THRESHOLDS["F4_max_seconds"]


def test_regression_second_equation_under_same_card():
    """Регрессия тихой потери связей (найдена M2-метрикой): второй блок
    «N / термы» под той же карточкой *EQUATION раньше молча пропускался —
    29 из 30 связей не проверялись, прогон проходил наполовину."""
    from pbcchecker.inp_equations import parse_equations

    body = "*EQUATION\n2\n1, 1, 1.0, 2, 1, -1.0\n2\n3, 1, 1.0, 4, 1, -1.0\n"
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.inp"
        p.write_text(body)
        with pytest.raises(ValueError, match="данные после"):
            parse_equations(p)


def test_regression_two_cards_still_parse():
    from pbcchecker.inp_equations import parse_equations

    body = ("*EQUATION\n2\n1, 1, 1.0, 2, 1, -1.0\n"
            "*EQUATION\n2\n3, 1, 1.0, 4, 1, -1.0\n")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "b.inp"
        p.write_text(body)
        assert len(parse_equations(p)) == 2
