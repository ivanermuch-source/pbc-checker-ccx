import json

import pytest

from pbcchecker import check_job
from pbcchecker.cli import main as cli_main
from pbcchecker.frd_disp import iter_disp_blocks, read_ccx_version
from pbcchecker.inp_equations import parse_equations

from conftest import make_frd, make_inp


def test_parse_equations_basic(tmp_job):
    eqs = [ [(93, 3, 1.0), (49070, 3, -1.0)],
            [(197, 1, 1.0), (49071, 1, -1.0)] ]
    make_inp(tmp_job.inp, eqs)
    parsed = parse_equations(tmp_job.inp)
    assert len(parsed) == 2
    assert parsed[0].terms == ((93, 3, 1.0), (49070, 3, -1.0))
    assert parsed[1].index == 1
    assert "u(93,3)" in parsed[0].render()


def test_parse_equations_comments_and_case(tmp_path):
    p = tmp_path / "x.inp"
    p.write_text("** comment\n*node file\nu\n*eQuAtIoN\n1\n5, 2, 2.5\n** trail\n", encoding="utf-8")
    parsed = parse_equations(p)
    assert parsed[0].terms == ((5, 2, 2.5),)


def test_parse_bad_dof_raises(tmp_path):
    p = make_inp(tmp_path / "x.inp", [[(1, 4, 1.0), (2, 1, -1.0)]])
    with pytest.raises(ValueError, match="dof=4"):
        parse_equations(p)


def test_parse_truncated_raises(tmp_path):
    p = tmp_path / "x.inp"
    p.write_text("*EQUATION\n2\n1, 1, 1.0, 2, 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="термов"):
        parse_equations(p)


def test_no_equations_empty(tmp_path):
    p = make_inp(tmp_path / "x.inp", [])
    assert parse_equations(p) == []


def test_frd_blocks_and_version(tmp_job):
    make_frd(tmp_job.frd, [
        (1, "0.1000000E-01", {1: (0.0, 0.0, -1.3e-3), 2: (1.5e-4, 0.0, -1.3e-3)}),
        (2, "0.2000000E-01", {1: (0.0, 0.0, -2.6e-3), 2: (3.0e-4, 0.0, -2.6e-3)}),
    ])
    assert read_ccx_version(tmp_job.frd) == "Version 2.20"
    blocks = list(iter_disp_blocks(tmp_job.frd))
    assert len(blocks) == 2
    assert blocks[0].step == 1 and "0.1000000E-01" in blocks[0].label
    assert blocks[0].nodes.tolist() == [1, 2]
    # E12.5 округляет: -1.3e-3 -> -1.30000E-03 точно
    assert blocks[0].u[0, 2] == pytest.approx(-1.3e-3, abs=1e-12)


def test_frd_skips_dispi(tmp_job):
    text = make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (1e-3, 0, 0)})]).read_text()
    text = text.replace("-4  DISP", "-4  DISPI", 1)  # первый блок — мнимый
    tmp_job.frd.write_text(text, encoding="utf-8")
    blocks = list(iter_disp_blocks(tmp_job.frd))
    assert len(blocks) == 0


def test_frd_split_fallback(tmp_job):
    make_frd(tmp_job.frd, [(1, "0.1E-01", {7: (2e-3, 0, 0)})])
    lines = tmp_job.frd.read_text().splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(" -1") and "7" in ln[:13]:
            lines[i] = " -1         7  2.000000E-03  0.000000E+00  0.000000E+00"  # 13-символьные поля
            break
    tmp_job.frd.write_text("\n".join(lines) + "\n", encoding="utf-8")
    blk = next(iter(iter_disp_blocks(tmp_job.frd)))
    assert blk.nodes.tolist() == [7] and blk.u[0, 0] == pytest.approx(2e-3, abs=1e-9)


def test_pass_zero_residual(tmp_job):
    make_inp(tmp_job.inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (5e-4, 0, 0), 2: (5e-4, 0, 0)})])
    v = check_job(tmp_job.inp, tmp_job.frd, out_dir=tmp_job.root)
    assert v.verdict == "PASS" and v.passed is True
    assert v.max_rel_residual <= 1e-6
    assert len(v.report_paths) == 2  # json + md
    data = json.loads(v.report_paths[0].read_text(encoding="utf-8"))
    assert data["inputs"]["inp_sha256"] and data["ccx_version"] == "Version 2.20"
    assert "pbc_report_job_" in v.report_paths[0].name  # датирование в имени


def test_fail_corrupted(tmp_job):
    make_inp(tmp_job.inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (5e-4, 0, 0), 2: (4.9e-4, 0, 0)})])  # сдвиг 1e-5
    v = check_job(tmp_job.inp, tmp_job.frd)
    assert v.verdict == "FAIL" and v.passed is False
    assert v.worst_eq == 0


def test_na_without_equations(tmp_job):
    make_inp(tmp_job.inp, [])
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (0, 0, 0)})])
    v = check_job(tmp_job.inp, tmp_job.frd)
    assert v.verdict == "N/A" and v.passed is None


def test_multistep_picks_worst(tmp_job):
    make_inp(tmp_job.inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    make_frd(tmp_job.frd, [
        (1, "0.1E-01", {1: (1e-3, 0, 0), 2: (1e-3, 0, 0)}),   # чисто
        (2, "0.2E-01", {1: (2e-3, 0, 0), 2: (1.5e-3, 0, 0)}), # испорчен
    ])
    v = check_job(tmp_job.inp, tmp_job.frd)
    assert v.verdict == "FAIL" and v.worst_step.startswith("step 2")


def test_missing_frd(tmp_job):
    make_inp(tmp_job.inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    with pytest.raises(FileNotFoundError):
        check_job(tmp_job.inp)  # job.frd не создан


def test_missing_disp_block(tmp_job):
    make_inp(tmp_job.inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    make_frd(tmp_job.frd, [])
    with pytest.raises(ValueError, match="DISP"):
        check_job(tmp_job.inp, tmp_job.frd)


def test_equation_node_absent(tmp_job):
    make_inp(tmp_job.inp, [[(99, 1, 1.0), (2, 1, -1.0)]])
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (0, 0, 0), 2: (0, 0, 0)})])
    with pytest.raises(KeyError, match="99"):
        check_job(tmp_job.inp, tmp_job.frd)


def test_cli_exit_codes(tmp_job):
    make_inp(tmp_job.inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (5e-4, 0, 0), 2: (5e-4, 0, 0)})])
    assert cli_main([str(tmp_job.inp)]) == 0
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (5e-4, 0, 0), 2: (4e-4, 0, 0)})])
    assert cli_main([str(tmp_job.inp)]) == 1
    make_inp(tmp_job.inp, [])
    assert cli_main([str(tmp_job.inp)]) == 0  # N/A
    assert cli_main([str(tmp_job.inp), "--frd", str(tmp_job.root / "nope.frd")]) == 0  # N/A раньше поиска
    make_inp(tmp_job.inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    assert cli_main([str(tmp_job.inp), "--frd", str(tmp_job.root / "nope.frd")]) == 2


def test_cli_tol_flag(tmp_job):
    make_inp(tmp_job.inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (5e-4, 0, 0), 2: (4.9e-4, 0, 0)})])  # ~2e-2
    assert cli_main([str(tmp_job.inp)]) == 1                 # порог по умолчанию 1e-3
    assert cli_main([str(tmp_job.inp), "--tol", "1e-1"]) == 0


def test_cli_rejects_nonpositive_tol(tmp_job):
    make_inp(tmp_job.inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    for bad in ("0", "-1e-3", "nan"):
        with pytest.raises(SystemExit) as ei:
            cli_main([str(tmp_job.inp), "--tol", bad])
        assert ei.value.code == 2


def test_cli_unexpected_error_exit2(tmp_job):
    make_inp(tmp_job.inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    # каталог вместо .frd -> OSError (PermissionError/IsADirectoryError) -> код 2, не traceback
    assert cli_main([str(tmp_job.inp), "--frd", str(tmp_job.root)]) == 2


def test_parse_zero_and_negative_terms(tmp_path):
    for n in ("0", "-2"):
        p = tmp_path / "x.inp"
        p.write_text(f"*EQUATION\n{n}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="< 1"):
            parse_equations(p)


def test_parse_equation_interrupted_by_card(tmp_path):
    p = tmp_path / "x.inp"
    p.write_text("*EQUATION\n2\n1, 1, 1.0\n*NSET, N=X\n2, 1, -1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="карточка"):
        parse_equations(p)


def test_parse_garbage_token(tmp_path):
    p = tmp_path / "x.inp"
    p.write_text("*EQUATION\n1\n1, 1, абырвалг\n", encoding="utf-8")
    with pytest.raises(ValueError, match="не разбирается"):
        parse_equations(p)


def test_to_float_fortran_exponents():
    from pbcchecker.inp_equations import to_float
    assert to_float("1.0-100") == pytest.approx(1e-100)   # фортран-экспонента без E
    assert to_float("2.5+3") == pytest.approx(2500.0)
    assert to_float("1.0D+00") == 1.0                      # фортранское D
    assert to_float("-1.3E-03") == pytest.approx(-1.3e-3)


def test_parse_fortran_coef(tmp_path):
    p = tmp_path / "x.inp"
    p.write_text("*EQUATION\n2\n1, 1, 1.0-3, 2, 1, -1.0D+00\n", encoding="utf-8")
    parsed = parse_equations(p)
    assert parsed[0].terms == ((1, 1, 1e-3), (2, 1, -1.0))


def test_frd_unterminated_disp_raises(tmp_job):
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (0, 0, 0)}), (2, "0.2E-01", {1: (0, 0, 0)})])
    text = tmp_job.frd.read_text(encoding="utf-8").replace("    -3\n", "", 1)
    tmp_job.frd.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="не закрыт"):
        list(iter_disp_blocks(tmp_job.frd))


def test_frd_eof_without_close_recovers(tmp_job):
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (1e-3, 0, 0)})])
    text = tmp_job.frd.read_text(encoding="utf-8").replace("    -3\n", "", 1)
    tmp_job.frd.write_text(text, encoding="utf-8")
    blocks = list(iter_disp_blocks(tmp_job.frd))
    assert len(blocks) == 1 and blocks[0].nodes.tolist() == [1]


def test_frd_empty_disp_block_raises(tmp_job):
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (0, 0, 0)})])
    lines = [ln for ln in tmp_job.frd.read_text(encoding="utf-8").splitlines()
             if not ln.startswith(" -1")]
    tmp_job.frd.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="пустой блок"):
        list(iter_disp_blocks(tmp_job.frd))


def test_frd_bad_node_line_raises(tmp_job):
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (0, 0, 0)})])
    lines = [ln.replace(ln, " -1  мусор") if ln.startswith(" -1") else ln
             for ln in tmp_job.frd.read_text(encoding="utf-8").splitlines()]
    tmp_job.frd.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="нечитаемая"):
        list(iter_disp_blocks(tmp_job.frd))


def test_report_worst_equations_sorted(tmp_job):
    make_inp(tmp_job.inp, [
        [(1, 1, 1.0), (2, 1, -1.0)],   # чистое
        [(3, 1, 1.0), (4, 1, -1.0)],   # испорченное — должно быть худшим №1
    ])
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (5e-4, 0, 0), 2: (5e-4, 0, 0),
                                           3: (1e-3, 0, 0), 4: (9.9e-4, 0, 0)})])
    v = check_job(tmp_job.inp, tmp_job.frd, out_dir=tmp_job.root)
    data = json.loads(v.report_paths[0].read_text(encoding="utf-8"))
    assert data["worst_equations"][0]["index"] == 1
    assert data["worst_equations"][0]["max_rel_residual"] == pytest.approx(1e-2, rel=0.01)
    md = v.report_paths[1].read_text(encoding="utf-8")
    assert "u(4,1)" in md.split("## Худшие")[1]   # в топе — термы испорченного уравнения
    assert "u(2,1)" in md.split("## Худшие")[1]


def test_reports_same_day_not_overwritten(tmp_job):
    make_inp(tmp_job.inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (5e-4, 0, 0), 2: (5e-4, 0, 0)})])
    check_job(tmp_job.inp, tmp_job.frd, out_dir=tmp_job.root)
    check_job(tmp_job.inp, tmp_job.frd, out_dir=tmp_job.root)
    reports = sorted(tmp_job.root.glob("pbc_report_job_*.json"))
    assert len(reports) == 2   # append-only: второй прогон не затёр первый
