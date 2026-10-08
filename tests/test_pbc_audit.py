"""P-серия: аудит самих *EQUATION по геометрии *NODE (--pbc-audit).

P1 coverage — узлы противоположных граней покрыты связями; P2 симметрия —
коэффициенты пар противоположны (±); P3 — дубликаты уравнений. Слепая зона
гейта невязок: кривые связи выполняются точно и дают PASS.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pbcchecker.cli import main as cli_main
from pbcchecker.gate import check_job
from pbcchecker.inp_equations import parse_equations
from pbcchecker.pbc_audit import audit_pbc, parse_nodes

# RVE-сетка: период по x (0→1), граневые узлы 101..106 (x=0) ↔ 201..206 (x=1),
# соответствие по (y, z): j → y = j % 3, z = j // 3; 6 пар × 3 dof = 18 связей
PAIRS = [(100 + j, 200 + j) for j in range(1, 7)]


def _rve_files(tmp: Path, *, drop_pair=None, dup=False, asym=False,
               internal=True, no_nodes=False) -> tuple[Path, Path]:
    """Валидный PBC-сетап по построению + опциональные дефекты сетапа."""
    lines = ["*NODE"] if not no_nodes else []
    for a, b in PAIRS:
        j = a - 101
        y, z = j % 3, j // 3
        if not no_nodes:
            lines.append(f"{a}, 0.0, {y}.0, {z}.0")
            lines.append(f"{b}, 1.0, {y}.0, {z}.0")
    if internal and not no_nodes:       # внутренний слой — не должен мешать аудиту
        lines.append("900, 0.5, 1.0, 0.5")
    for i, (a, b) in enumerate(PAIRS):
        if drop_pair is not None and i == drop_pair:
            continue
        for d in (1, 2, 3):
            cb = -0.5 if (asym and i == 1 and d == 1) else -1.0
            lines += ["*EQUATION", "2", f"{a}, {d}, 1.0, {b}, {d}, {cb}"]
    if dup:                             # точная копия первой связи
        a, b = PAIRS[0]
        lines += ["*EQUATION", "2", f"{a}, 1, 1.0, {b}, 1, -1.0"]
    inp = tmp / "rve.inp"
    inp.write_text("\n".join(lines) + "\n", encoding="ascii")

    out = ["    1C", "    1UVERSION           Version 2.20",
           "  100CL  1  0.1000000E-01", "    -4  DISP        4    1",
           "    -5  D1          1    2    3"]
    u = {}
    for k, (a, b) in enumerate(PAIRS):
        ux = 1.0e-4 * (1 + k / 7)
        val = (ux, 2e-4, -1e-4)
        u[a] = u[b] = val
    u[900] = (1.5e-4, 2e-4, -1e-4)
    for n in sorted(u):
        vx, vy, vz = u[n]
        out.append(f" -1{n:10d}{vx:12.5E}{vy:12.5E}{vz:12.5E}")
    out.append("    -3")
    frd = tmp / "rve.frd"
    frd.write_text("\n".join(out) + "\n", encoding="ascii")
    return inp, frd


def test_P_all_pass(tmp_path):
    inp, frd = _rve_files(tmp_path)
    res = audit_pbc(inp, parse_equations(inp))
    assert res.verdict == "PASS"
    assert len(res.directions) == 1
    d = res.directions[0]
    assert d.axis == 0 and d.coverage == 1.0 and d.symmetry == 1.0
    assert not res.duplicates and d.n_face_nodes == 12 and d.n_equations == 18
    v = check_job(inp, frd, pbc_audit=True)
    assert v.verdict == "PASS" and v.passed and v.audit.verdict == "PASS"


def test_P1_uncovered_nodes_fail(tmp_path):
    inp, frd = _rve_files(tmp_path, drop_pair=2)
    res = audit_pbc(inp, parse_equations(inp))
    assert res.verdict == "FAIL"
    d = res.directions[0]
    assert d.coverage == 10 / 12 and set(d.uncovered) == set(PAIRS[2])
    v = check_job(inp, frd, pbc_audit=True)   # невязка чистая, сетап — нет
    assert not v.passed and v.verdict == "FAIL"


def test_P2_asymmetric_pair_fail(tmp_path):
    inp, frd = _rve_files(tmp_path, asym=True)
    res = audit_pbc(inp, parse_equations(inp))
    assert res.verdict == "FAIL"
    assert res.directions[0].symmetry == 17 / 18


def test_P3_duplicate_equation_fail(tmp_path):
    inp, frd = _rve_files(tmp_path, dup=True)
    res = audit_pbc(inp, parse_equations(inp))
    assert res.verdict == "FAIL" and res.duplicates == [(0, 18)]


def test_P_na_no_direction(tmp_path):
    inp, frd = _rve_files(tmp_path, internal=False)
    text = inp.read_text()
    inp.write_text(text.replace("1.0, ", "0.0, "))
    # ВСЕ узлы в одной плоскости x=0 → span=0, ось пропускается целиком
    res = audit_pbc(inp, parse_equations(inp))
    assert res.verdict == "N/A" and "направления периодичности не найдены" in res.note


def test_P_face_tolerance_boundary_node(tmp_path):
    """Узел на расстоянии tol от грани считается лежащим на грани (FACE_TOL):
    RVE со span=1, tol=1e-6; узел на x=1e-6 принадлежит грани x=0."""
    lines = ["*NODE",
             "1, 0.0, 0.0, 0.0", "2, 1.0, 0.0, 0.0",
             "3, 0.0, 1.0, 0.0", "4, 1.0, 1.0, 0.0",
             "5, 1e-06, 0.5, 0.0"]                     # в полосе грани x=0
    for a, b in ((1, 2), (3, 4), (5, 2)):
        lines += ["*EQUATION", "2", f"{a}, 1, 1.0, {b}, 1, -1.0"]
    inp = tmp_path / "tol.inp"
    inp.write_text("\n".join(lines) + "\n", encoding="ascii")
    res = audit_pbc(inp, parse_equations(inp))
    assert res.verdict == "PASS"          # узел 5 покрыт — он на грани, не дыра


def test_P_node_two_fields_z_zero(tmp_path):
    """*NODE с двумя координатами: z = 0 (паддинг), а не мусор."""
    inp = tmp_path / "xy.inp"
    inp.write_text("*NODE\n101, 1.5, 2.5\n", encoding="ascii")
    assert parse_nodes(inp) == {101: (1.5, 2.5, 0.0)}


def test_P_include_coordinate_conflict(tmp_path):
    """Узел в main и в include с разными координатами — ошибка, не тихая
    перезапись (конфликт между файлами, внутри файла ловит другой тест)."""
    main_inp = tmp_path / "main.inp"
    inc = tmp_path / "nodes.inc"
    inc.write_text("*NODE\n101, 1.0, 0.0, 0.0\n", encoding="ascii")
    main_inp.write_text("*NODE\n101, 0.0, 0.0, 0.0\n"
                        "*INCLUDE, INPUT=nodes.inc\n"
                        "*EQUATION\n2\n101, 1, 1.0, 101, 1, -1.0\n", encoding="ascii")
    with pytest.raises(ValueError, match="переопределён"):
        parse_nodes(main_inp)


def test_P_no_nodes_card_is_error_when_requested(tmp_path):
    inp, frd = _rve_files(tmp_path, no_nodes=True)
    with pytest.raises(ValueError, match="нет карточек \\*NODE"):
        check_job(inp, frd, pbc_audit=True)


def test_P_include_nodes(tmp_path):
    main_inp = tmp_path / "main.inp"
    inc = tmp_path / "nodes.inc"
    inc.write_text("*NODE\n101, 0.0, 0.0, 0.0\n201, 1.0, 0.0, 0.0\n",
                   encoding="ascii")
    main_inp.write_text("*INCLUDE, INPUT=nodes.inc\n"
                        "*EQUATION\n2\n101, 1, 1.0, 201, 1, -1.0\n", encoding="ascii")
    nodes = parse_nodes(main_inp)
    assert nodes == {101: (0.0, 0.0, 0.0), 201: (1.0, 0.0, 0.0)}
    res = audit_pbc(main_inp, parse_equations(main_inp))
    assert res.verdict == "PASS" and res.directions[0].coverage == 1.0


def test_P_broken_node_line(tmp_path):
    inp = tmp_path / "bad.inp"
    inp.write_text("*NODE\n101, x, 0, 0\n", encoding="ascii")
    with pytest.raises(ValueError, match="строка 2"):
        parse_nodes(inp)


def test_P_node_redefined(tmp_path):
    inp = tmp_path / "bad.inp"
    inp.write_text("*NODE\n101, 0.0, 0.0, 0.0\n101, 1.0, 0.0, 0.0\n", encoding="ascii")
    with pytest.raises(ValueError, match="переопределён"):
        parse_nodes(inp)


def test_P_cli_e2e(tmp_path, capsys):
    inp, frd = _rve_files(tmp_path)
    assert cli_main([str(inp), "--frd", str(frd), "--pbc-audit"]) == 0
    out = capsys.readouterr().out
    assert "PBC-аudit: PASS" in out and "покрытие граней 12/12" in out

    inp2, frd2 = _rve_files(tmp_path, drop_pair=0)
    capsys.readouterr()
    assert cli_main([str(inp2), "--frd", str(frd2), "--pbc-audit"]) == 1
    out2 = capsys.readouterr().out
    assert "PBC-аudit: FAIL" in out2 and "покрытие граней 10/12" in out2
