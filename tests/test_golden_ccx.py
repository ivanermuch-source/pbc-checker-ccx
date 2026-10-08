"""G-серия: golden по версиям ccx — расширяемый каркас.

Закрывает риск «форматы DISP-блоков между версиями ccx 2.17/2.19/2.22 слегка
отличаются»: один golden-прогон на версию, паритет вердикта и невязки
проверяется в CI. Пока golden-файлов нет — тест skipped (не падает).

Как добавить: создайте examples/golden_ccx/golden.json вида

    {"ccx_2.17": {"inp": "job.inp", "frd": "job.frd",
                  "verdict": "PASS", "max_rel_residual": 1.2e-06,
                  "ccx_version": "Version 2.17"}}

и положите рядом сами файлы (сгенерированные соответствующей версией ccx).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pbcchecker.frd_disp import read_ccx_version
from pbcchecker.gate import check_job

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "examples" / "golden_ccx"


def _golden_entries():
    manifest = GOLDEN_DIR / "golden.json"
    if not manifest.is_file():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return [(name, entry) for name, entry in data.items()]


def test_G_golden_ccx_versions():
    entries = _golden_entries()
    if not entries:
        pytest.skip("нет examples/golden_ccx/golden.json — положите прогоны "
                    "разных версий ccx (2.17/2.19/2.22), см. docstring модуля")
    for name, e in entries:
        inp, frd = GOLDEN_DIR / e["inp"], GOLDEN_DIR / e["frd"]
        v = check_job(inp, frd, tol=e.get("tol", 1e-3))
        rel_ref = e["max_rel_residual"]
        assert v.verdict == e["verdict"], f"{name}: verdict {v.verdict} ≠ {e['verdict']}"
        assert abs(v.max_rel_residual - rel_ref) <= 1e-9 * max(rel_ref, 1e-300), \
            f"{name}: невязка {v.max_rel_residual} ≠ {rel_ref}"
        assert read_ccx_version(frd) == e["ccx_version"], \
            f"{name}: версия из заголовка не совпала"
