"""E2E: CLI через subprocess — коды выхода, вывод, отчёты, окружение реального пользователя."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import make_frd, make_inp

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.e2e


def run_cli(args, cwd=None, env_extra=None, timeout=120):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "pbcchecker.cli", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(cwd or ROOT), env=env, timeout=timeout)


def test_e2e_pass_fail_flow(tmp_path):
    inp, frd = tmp_path / "job.inp", tmp_path / "job.frd"
    make_inp(inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    make_frd(frd, [(1, "0.1E-01", {1: (5e-4, 0, 0), 2: (5e-4, 0, 0)})])
    r = run_cli([str(inp), "--out-dir", str(tmp_path / "rep")])
    assert r.returncode == 0 and "PASS:" in r.stdout and "отчёт:" in r.stdout
    make_frd(frd, [(1, "0.1E-01", {1: (5e-4, 0, 0), 2: (4e-4, 0, 0)})])
    r = run_cli([str(inp)])
    assert r.returncode == 1 and "FAIL:" in r.stdout
    assert "худшее уравнение" in r.stdout       # виновник виден без отчёта
    assert (tmp_path / "rep").is_dir()          # отчёты созданы


def test_e2e_na_report_and_exit0(tmp_path):
    inp = tmp_path / "job.inp"
    make_inp(inp, [])
    r = run_cli([str(inp), "--out-dir", str(tmp_path)])
    assert r.returncode == 0 and "N/A" in r.stdout
    reports = sorted(tmp_path.glob("pbc_report_job_*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text(encoding="utf-8"))["verdict"] == "N/A"


def test_e2e_data_error_exit2(tmp_path):
    inp = tmp_path / "job.inp"
    make_inp(inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    r = run_cli([str(inp), "--frd", str(tmp_path / "нет_такого.frd")])
    assert r.returncode == 2 and "ОШИБКА ДАННЫХ" in r.stderr
    assert "Traceback" not in r.stderr          # crash не протекает пользователю


def test_e2e_redirect_nonutf8_console(tmp_path):
    """Редирект/консоль с чужой кодировкой (cp1252 в CI) не должен ронять
    прогон: кириллица заменяется, код выхода честный (не 1=FAIL)."""
    inp, frd = tmp_path / "job.inp", tmp_path / "job.frd"
    make_inp(inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    make_frd(frd, [(1, "0.1E-01", {1: (5e-4, 0, 0), 2: (5e-4, 0, 0)})])
    r = run_cli([str(inp)], env_extra={"PYTHONIOENCODING": "cp1252"})
    assert r.returncode == 0 and "Traceback" not in r.stderr + r.stdout


def test_e2e_cyrillic_paths_and_relative_cwd(tmp_path):
    work = tmp_path / "Программы" / "прогон 1"
    work.mkdir(parents=True)
    make_inp(work / "job.inp", [[(1, 1, 1.0), (2, 1, -1.0)]])
    make_frd(work / "job.frd", [(1, "0.1E-01", {1: (5e-4, 0, 0), 2: (5e-4, 0, 0)})])
    r = run_cli(["job.inp"], cwd=work)          # относительный путь из другого cwd
    assert r.returncode == 0 and "PASS:" in r.stdout


def test_e2e_bom_and_include(tmp_path):
    (tmp_path / "pbc.inc").write_text(
        "*EQUATION\n2\n1, 1, 1.0, 2, 1, -1.0\n", encoding="utf-8")
    inp = tmp_path / "job.inp"
    inp.write_bytes("\ufeff*INCLUDE\npbc.inc\n".encode("utf-8"))   # BOM + include
    make_frd(tmp_path / "job.frd", [(1, "0.1E-01", {1: (5e-4, 0, 0), 2: (5e-4, 0, 0)})])
    r = run_cli([str(inp)])
    assert r.returncode == 0 and "PASS:" in r.stdout   # раньше было бы тихое N/A


def test_e2e_big_deck(tmp_path):
    """Репрезентативный RVE: 1500 узлов, 750 уравнений, 3 шага — PASS и адекватное время."""
    n, n_eq, steps = 1500, 750, 3
    # u одинаково внутри пары (j, j+750): период по модулю n//2
    ux = {i: 1.5e-4 * (1 + (i - 1) % (n // 2) % 7) for i in range(1, n + 1)}
    lines = ["*NODE, NSET=ALL"]
    lines += [f"{i}, 0.0, 0.0, 0.0" for i in range(1, n + 1)]
    for j in range(1, n_eq + 1):
        lines += ["*EQUATION", "2", f"{j}, 1, 1.0, {j + n // 2}, 1, -1.0"]
    (tmp_path / "big.inp").write_text("\n".join(lines), encoding="utf-8")
    make_frd(tmp_path / "big.frd",
             [(s, f"0.{s}E-02", {i: (ux[i] * s, 0.0, 0.0) for i in range(1, n + 1)})
              for s in range(1, steps + 1)])
    r = run_cli([str(tmp_path / "big.inp"), "--out-dir", str(tmp_path / "rep")], timeout=180)
    assert r.returncode == 0 and "PASS:" in r.stdout
    data = json.loads(next((tmp_path / "rep").glob("*.json")).read_text(encoding="utf-8"))
    assert data["n_equations"] == n_eq and len(data["worst_equations"]) == 10
    assert len(data["steps"]) == steps


def test_e2e_installed_entry_point(tmp_path):
    """Точ входа pbc-check из установки пакета (если доступна в PATH)."""
    import shutil
    exe = shutil.which("pbc-check")
    if not exe:
        pytest.skip("pbc-check не в PATH")
    inp = tmp_path / "job.inp"
    make_inp(inp, [])
    r = subprocess.run([exe, str(inp)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=60)
    assert r.returncode == 0 and "N/A" in r.stdout
