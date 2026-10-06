"""Юнит-тесты pbcchecker: разбор .inp/.frd, невязки, CLI, отчёты."""
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


# --- покрытие редких путей разбора ---

def test_parse_blank_and_comment_between_card_and_count(tmp_path):
    p = tmp_path / "x.inp"
    p.write_text("*EQUATION\n\n** сколько термов\n1\n1, 1, 2.0\n", encoding="utf-8")
    assert parse_equations(p)[0].terms == ((1, 1, 2.0),)


def test_parse_nonint_term_count_raises(tmp_path):
    p = tmp_path / "x.inp"
    p.write_text("*EQUATION\nдва\n1, 1, 1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="число термов не целое"):
        parse_equations(p)


def test_parse_eos_after_card_raises(tmp_path):
    p = tmp_path / "x.inp"
    p.write_text("*EQUATION\n", encoding="utf-8")
    with pytest.raises(ValueError, match="без строки с числом термов"):
        parse_equations(p)


def test_parse_comment_inside_terms(tmp_path):
    p = tmp_path / "x.inp"
    p.write_text("*EQUATION\n2\n1, 1, 1.0\n** перенос\n2, 1, -1.0\n", encoding="utf-8")
    assert parse_equations(p)[0].terms == ((1, 1, 1.0), (2, 1, -1.0))


def test_frd_version_unknown_when_absent(tmp_job):
    text = make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (0, 0, 0)})]).read_text()
    tmp_job.frd.write_text("\n".join(ln for ln in text.splitlines()
                                     if "1UVERSION" not in ln) + "\n", encoding="utf-8")
    assert read_ccx_version(tmp_job.frd) == "unknown"


def test_frd_malformed_100cl(tmp_job):
    text = make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (1e-3, 0, 0)})]).read_text()
    text = text.replace("  100CL  1  0.1E-01", "  100CL  ?  ?")
    tmp_job.frd.write_text(text, encoding="utf-8")
    blocks = list(iter_disp_blocks(tmp_job.frd))
    assert len(blocks) == 1 and blocks[0].step == 0
    assert blocks[0].label.startswith("block 0")  # время не разобранось


def test_frd_other_dataset_closes_disp(tmp_job):
    text = make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (1e-3, 0, 0)})]).read_text()
    text = text.replace("    -3", "    -4  STRESS       1    1")
    tmp_job.frd.write_text(text, encoding="utf-8")
    blocks = list(iter_disp_blocks(tmp_job.frd))  # нестандартное закрытие, данные не теряем
    assert len(blocks) == 1 and blocks[0].nodes.tolist() == [1]


def test_frd_other_dataset_empty_disp_ignored(tmp_job):
    # пустой незакрытый DISP + другой датасет: данных нет — блок просто забывается
    text = make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (1e-3, 0, 0)})]).read_text()
    lines = []
    for ln in text.splitlines():
        if ln.startswith(" -1"):
            continue                      # убрать строки узлов
        if ln.strip() == "-3":
            ln = "    -4  STRESS       1    1"  # другой датасет вместо закрытия
        lines.append(ln)
    tmp_job.frd.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert list(iter_disp_blocks(tmp_job.frd)) == []


def test_frd_garbage_node_tokens_raise(tmp_job):
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (0, 0, 0)})])
    lines = [ln.replace(ln, " -1 abc def ghi jkl") if ln.startswith(" -1") else ln
             for ln in tmp_job.frd.read_text(encoding="utf-8").splitlines()]
    tmp_job.frd.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="нечитаемая"):
        list(iter_disp_blocks(tmp_job.frd))


def test_compute_residuals_direct_errors(tmp_job):
    from pbcchecker.frd_disp import DispBlock
    from pbcchecker.residual import compute_residuals
    import numpy as np
    eqs = parse_equations(make_inp(tmp_job.inp, [[(1, 1, 1.0), (2, 1, -1.0)]]))
    with pytest.raises(ValueError, match="нет уравнений"):
        compute_residuals([], [DispBlock(1, "b", np.array([1], np.int32), np.zeros((1, 3)))])
    with pytest.raises(ValueError, match="пустой блок"):
        compute_residuals(eqs, [DispBlock(1, "b", np.array([], np.int32),
                                          np.zeros((0, 3)))])


def test_node_missing_in_one_step(tmp_job):
    # узел есть в шаге 1, но исчез в шаге 2 — ошибка, а не тихий пропуск шага
    make_inp(tmp_job.inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (0, 0, 0), 2: (0, 0, 0)}),
                           (2, "0.2E-01", {1: (0, 0, 0)})])
    with pytest.raises(KeyError, match="отсутствуют в блоке"):
        check_job(tmp_job.inp, tmp_job.frd)


def test_cli_out_dir_prints_report_paths(tmp_job, capsys):
    make_inp(tmp_job.inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (5e-4, 0, 0), 2: (5e-4, 0, 0)})])
    assert cli_main([str(tmp_job.inp), "--out-dir", str(tmp_job.root / "rep")]) == 0
    assert "отчёт:" in capsys.readouterr().out


# --- BOM, кодировки, запятые, *INCLUDE, бинарный .frd (аудит 2026-10-06) ---

def test_parse_bom_equation_first_line(tmp_path):
    p = tmp_path / "x.inp"
    p.write_bytes(b"\xef\xbb\xbf*EQUATION\n1\n1, 1, 2.0\n")   # BOM + карточка 1-й строкой
    assert parse_equations(p)[0].terms == ((1, 1, 2.0),)


def test_parse_cp1251_russian_comments(tmp_path):
    p = tmp_path / "x.inp"
    p.write_bytes("*EQUATION\n** связи грань X\n1\n1, 1, 2.0\n".encode("cp1251"))
    assert parse_equations(p)[0].terms == ((1, 1, 2.0),)


def test_parse_trailing_and_double_commas(tmp_path):
    p = tmp_path / "x.inp"
    p.write_text("*EQUATION\n2\n1, 1, 1.0,,\n2, 1, -1.0,\n", encoding="utf-8")
    assert parse_equations(p)[0].terms == ((1, 1, 1.0), (2, 1, -1.0))


def test_parse_extra_tokens_raise(tmp_path):
    p = tmp_path / "x.inp"
    p.write_text("*EQUATION\n1\n1, 1, 1.0, 2, 2, 5.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="лишних"):
        parse_equations(p)


def test_parse_nonfinite_coef_raises(tmp_path):
    p = tmp_path / "x.inp"
    p.write_text("*EQUATION\n1\n1, 1, nan\n", encoding="utf-8")
    with pytest.raises(ValueError, match="не конечен"):
        parse_equations(p)


def test_include_equations_parsed(tmp_path):
    (tmp_path / "pbc.inc").write_text("*EQUATION\n2\n1, 1, 1.0, 2, 1, -1.0\n", encoding="utf-8")
    main = tmp_path / "job.inp"
    main.write_text("*STEP\n*INCLUDE\npbc.inc\n*END STEP\n", encoding="utf-8")
    assert parse_equations(main)[0].terms == ((1, 1, 1.0), (2, 1, -1.0))


def test_include_input_param_and_nested(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "leaf.inc").write_text(
        "*EQUATION\n1\n7, 2, 3.0\n", encoding="utf-8")
    (tmp_path / "sub" / "mid.inc").write_text("*INCLUDE\nleaf.inc\n", encoding="utf-8")
    main = tmp_path / "job.inp"
    main.write_text("*INCLUDE, INPUT=sub/mid.inc\n", encoding="utf-8")
    assert parse_equations(main)[0].terms == ((7, 2, 3.0),)


def test_include_cycle_raises(tmp_path):
    a = tmp_path / "a.inp"
    b = tmp_path / "b.inp"
    a.write_text("*INCLUDE\nb.inp\n", encoding="utf-8")
    b.write_text("*INCLUDE\na.inp\n", encoding="utf-8")
    with pytest.raises(ValueError, match="цикл"):
        parse_equations(a)


def test_include_missing_raises(tmp_path):
    p = tmp_path / "job.inp"
    p.write_text("*INCLUDE\nnope.inc\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="nope.inc"):
        parse_equations(p)


def test_include_without_name_raises(tmp_path):
    p = tmp_path / "job.inp"
    p.write_text("*INCLUDE\n", encoding="utf-8")          # EOF сразу за карточкой
    with pytest.raises(ValueError, match="без имени файла"):
        parse_equations(p)


def test_include_too_deep_raises(tmp_path):
    for i in range(35):                                   # цепочка длиннее лимита
        (tmp_path / f"f{i}.inp").write_text(f"*INCLUDE\nf{i + 1}.inp\n", encoding="utf-8")
    with pytest.raises(ValueError, match="глубина"):
        parse_equations(tmp_path / "f0.inp")


def test_cli_na_out_dir_prints_report(tmp_job, capsys):
    make_inp(tmp_job.inp, [])
    rc = cli_main([str(tmp_job.inp), "--out-dir", str(tmp_job.root)])
    out = capsys.readouterr().out
    assert rc == 0 and "N/A" in out and "отчёт:" in out


def test_gate_default_tol_and_str_paths(tmp_job):
    # rel = 1.5e-3: FAIL при default 1e-3 (мутация default 2e-3 давала бы PASS)
    make_inp(tmp_job.inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (1.0, 0, 0), 2: (0.9985, 0, 0)})])
    assert check_job(str(tmp_job.inp)).verdict == "FAIL"   # str-путь, frd по умолчанию
    assert cli_main([str(tmp_job.inp)]) == 1               # CLI default tol тот же


def test_gate_tol_boundary_equality_pass(tmp_job):
    # rel ровно = tol (0.5, точные степени двойки): нестрогое '<=' — PASS
    make_inp(tmp_job.inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (1.0, 0, 0), 2: (0.5, 0, 0)})])
    assert check_job(tmp_job.inp, tmp_job.frd, tol=0.5).verdict == "PASS"


# --- мутационные киллеры (добивание выживших мутантов) ---

def test_report_top_worst_capped_at_10(tmp_job):
    eqs = [[(i, 1, 1.0), (i + 20, 1, -1.0)] for i in range(1, 13)]   # 12 уравнений
    make_inp(tmp_job.inp, eqs)
    disp = {n: (1e-3, 0, 0) for n in list(range(1, 13)) + list(range(21, 33))}
    disp[25] = (2e-3, 0, 0)                    # уравнение #5 (индекс 4) — худшее
    make_frd(tmp_job.frd, [(1, "0.1E-01", disp)])
    v = check_job(tmp_job.inp, tmp_job.frd, out_dir=tmp_job.root)
    data = json.loads(v.report_paths[0].read_text(encoding="utf-8"))
    assert len(data["worst_equations"]) == 10          # не 11/12
    assert data["worst_equations"][0]["index"] == 4


def test_na_report_zero_equations(tmp_job):
    make_inp(tmp_job.inp, [])
    v = check_job(tmp_job.inp, out_dir=tmp_job.root)
    data = json.loads(v.report_paths[0].read_text(encoding="utf-8"))
    assert data["n_equations"] == 0 and data["worst_equations"] == []


def test_floor_abs_is_field_eps(tmp_job):
    make_inp(tmp_job.inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (1e-3, 0, 0), 2: (1e-3, 0, 0)})])
    v = check_job(tmp_job.inp, tmp_job.frd)
    s = v.result.steps[0]
    assert s.floor_abs == pytest.approx(5e-6 * 1e-3)   # пол = E12.5 * max|u|


def test_frd_fallback_misaligned_node(tmp_job):
    # узел не в I10-колонке: позиционный срез падает, split-fallback спасает
    make_frd(tmp_job.frd, [(1, "0.1E-01", {7: (2e-3, 0, 0)})])
    lines = [ln.replace(ln, " -1 7 2.000000E-03 0.000000E+00 0.000000E+00")
             if ln.startswith(" -1") else ln
             for ln in tmp_job.frd.read_text(encoding="utf-8").splitlines()]
    tmp_job.frd.write_text("\n".join(lines) + "\n", encoding="utf-8")
    blk = next(iter(iter_disp_blocks(tmp_job.frd)))
    assert blk.nodes.tolist() == [7] and blk.u[0, 0] == pytest.approx(2e-3, abs=1e-9)


def test_frd_fallback_extra_token_ignored(tmp_job):
    make_frd(tmp_job.frd, [(1, "0.1E-01", {7: (2e-3, 0, 0)})])
    lines = [ln.replace(ln, " -1 7 2.000000E-03 0.000000E+00 0.000000E+00 extra")
             if ln.startswith(" -1") else ln
             for ln in tmp_job.frd.read_text(encoding="utf-8").splitlines()]
    tmp_job.frd.write_text("\n".join(lines) + "\n", encoding="utf-8")
    blk = next(iter(iter_disp_blocks(tmp_job.frd)))
    assert blk.nodes.tolist() == [7]                    # 6-й токен игнорируется


def test_frd_ten_digit_node(tmp_job):
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1234567890: (1e-3, 0, 0)})])
    make_inp(tmp_job.inp, [[(1234567890, 1, 1.0), (2, 1, 0.0)]])
    # до 3N проверки: сам номер из колонок 3:13 читается целиком
    blocks = list(iter_disp_blocks(tmp_job.frd))
    assert blocks[0].nodes.tolist() == [1234567890]


def test_frd_error_line_number(tmp_job):
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (0, 0, 0)})])
    lines = tmp_job.frd.read_text(encoding="utf-8").splitlines()
    bad = 8                                              # 4 заголовка + 100CL + -4 + -5 + узел
    lines[bad - 1] = " -1 мусор"
    tmp_job.frd.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"строка 8\b"):
        list(iter_disp_blocks(tmp_job.frd))


def test_frd_block_labels_sequential_without_time(tmp_job):
    text = make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (0, 0, 0)}),
                                  (2, "0.2E-01", {1: (0, 0, 0)})]).read_text()
    text = text.replace("  100CL  1", "  100CL  ?").replace("  100CL  2", "  100CL  ?")
    tmp_job.frd.write_text(text, encoding="utf-8")
    blocks = list(iter_disp_blocks(tmp_job.frd))
    assert [b.label for b in blocks] == ["block 0", "block 1"]


def test_include_depth_boundary(tmp_path):
    def chain(n):
        for i in range(n):
            body = f"*INCLUDE\nf{i + 1}.inp\n" if i < n - 1 else "*EQUATION\n1\n7, 1, 1.0\n"
            (tmp_path / f"f{i}.inp").write_text(body, encoding="utf-8")
    chain(33)                                            # глубина 32 — ещё допустимо
    assert parse_equations(tmp_path / "f0.inp")[0].terms == ((7, 1, 1.0),)
    chain(34)                                            # глубина 33 — за пределом
    with pytest.raises(ValueError, match="глубина"):
        parse_equations(tmp_path / "f0.inp")


def test_parse_accepts_str_path(tmp_path):
    p = tmp_path / "x.inp"
    p.write_text("*EQUATION\n1\n1, 1, 2.0\n", encoding="utf-8")
    assert parse_equations(str(p))[0].terms == ((1, 1, 2.0),)


def test_inp_error_line_numbers(tmp_path):
    p = tmp_path / "x.inp"
    p.write_text("*EQUATION\n0\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"строка 2\b"):
        parse_equations(p)
    p.write_text("*EQUATION\n1\n1, 1, 1.0, 9, 9, 9.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"строка 3\b"):
        parse_equations(p)
    p.write_text("*EQUATION\n2\n1, 1, 1.0\n** перенос\n2, 1, -1.0, 7\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"строка 5\b"):
        parse_equations(p)
    p.write_text("*EQUATION\n2\n1, 1, 1.0\n*NSET\n2, 1, -1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"строке 4\b"):
        parse_equations(p)


def test_parse_extra_tokens_7_for_2(tmp_path):
    p = tmp_path / "x.inp"
    p.write_text("*EQUATION\n2\n1, 1, 1.0, 2, 1, -1.0, 7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="лишних"):
        parse_equations(p)


# --- векторный парсер .frd: точность побитово == float() ---

def _write_raw_frd(path, node_lines):
    out = ["    1C", "    1UVERSION           Version 2.20",
           "  100CL  1  0.1E-01", "    -4  DISP        4    1",
           "    -5  D1          1    2    3"] + node_lines + ["    -3"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def test_frd_glued_fields_bitwise(tmp_job):
    # слипшиеся поля реального ccx ("E+00-1.30000E-03"), границы ldexp-схемы
    # (E-18/E+10) и экстремум E+99 (блок уходит в построчный fallback)
    _write_raw_frd(tmp_job.frd, [
        " -1         3 0.00000E+00-1.30000E-03 0.00000E+00",
        " -1        93-1.30000E-03 0.00000E+00 9.99999E+99",
        " -1 123456789 1.00000E+00-2.50000E-01-3.75000E-05",
        " -1         4 3.00000E-04 1.23456E+10-9.99999E-18",
    ])
    blk = next(iter(iter_disp_blocks(tmp_job.frd)))
    assert blk.nodes.tolist() == [3, 93, 123456789, 4]
    # побитовое равенство float() — без approx: сертифицируем разбор
    assert blk.u.tolist() == [
        [0.0, float("-1.30000E-03"), 0.0],
        [float("-1.30000E-03"), 0.0, float("9.99999E+99")],
        [1.0, float("-2.50000E-01"), float("-3.75000E-05")],
        [float("3.00000E-04"), float("1.23456E+10"), float("-9.99999E-18")],
    ]


def test_frd_vector_matches_float_on_fixture(tmp_job):
    make_frd(tmp_job.frd, [(1, "0.1E-01", {n: (n * 1.7e-4, -n * 3.1e-5, 0.0)
                                           for n in range(1, 60)})])
    blk = next(iter(iter_disp_blocks(tmp_job.frd)))
    manual = [(int(l[3:13]), float(l[13:25]), float(l[25:37]))
              for l in tmp_job.frd.read_text(encoding="utf-8").splitlines()
              if l.startswith(" -1") and len(l) == 49]   # splitlines без "\n"
    assert blk.nodes.tolist() == [m[0] for m in manual]
    assert blk.u[:, 0].tolist() == [m[1] for m in manual]   # побитово
    assert blk.u[:, 1].tolist() == [m[2] for m in manual]


def test_frd_nonuniform_block_fallback(tmp_job):
    # блок из строк разной длины — построчный fallback с тем же результатом;
    # 10-значный узел проверяет I10-срез fallback'а
    _write_raw_frd(tmp_job.frd, [
        " -1         5  1.00000E-03  0.00000E+00  0.00000E+00",   # поля по 13
        " -112345678901.00000E-03 0.00000E+00 0.00000E+00",        # узел I10 + поля по 12
        " -1         6  2.00000E-03  0.00000E+00  0.00000E+00",
    ])
    blk = next(iter(iter_disp_blocks(tmp_job.frd)))
    assert blk.nodes.tolist() == [5, 1234567890, 6]
    assert blk.u[:, 0].tolist() == [1e-3, 1e-3, 2e-3]


def test_frd_torture_bitwise(tmp_job):
    """Значения, накрывающие каждый вес мантиссы/экспоненты векторного
    парсера и границы точной ldexp-схемы (k in [-22, 13]): любые мутации
    весов/таблиц дают неверные числа, а не тихий fallback."""
    rows = [
        (1234567890, "1.23456E+01", "9.87654E-02", "1.23456E+12"),
        (1023456789, "9.87654E+18", "1.23456E+10", "-9.87654E-07"),
        (7,          "9.99999E-17", "1.23456E-17", "1.00000E+05"),
    ]
    _write_raw_frd(tmp_job.frd,
                   [" -1%10d%12s%12s%12s" % (n, a, b, c) for n, a, b, c in rows])
    blk = next(iter(iter_disp_blocks(tmp_job.frd)))
    assert blk.nodes.tolist() == [r[0] for r in rows]
    want = [[float(r[1]), float(r[2]), float(r[3])] for r in rows]
    assert blk.u.tolist() == want          # побитово, включая границы k=-22/+13 и k=0


def test_frd_all_exponents_bitwise(tmp_job):
    """Свойство: каждая экспонента E12.5 из точного диапазона векторного
    парсера (e in [-17, 18]) даёт биты float()."""
    rows = []
    for i, e in enumerate(range(-17, 19)):        # 36 экспонент, разные мантиссы
        m = 100000 + (i * 123457) % 900000        # взаимно просто с 10 — без нулей
        s = "%c%d.%05dE%c%02d" % (" -"[i % 2], m // 100000, m % 100000,
                                  "+-"[e < 0], abs(e))
        rows.append((i + 1, s))
    _write_raw_frd(tmp_job.frd,
                   [" -1%10d%12s%12s%12s" % (n, a, "0.00000E+00", "0.00000E+00")
                    for n, a in rows])
    blk = next(iter(iter_disp_blocks(tmp_job.frd)))
    for i, (n, s) in enumerate(rows):
        assert blk.u[i, 0].hex() == float(s).hex(), (s, blk.u[i, 0].hex())
        assert blk.u[i, 1].hex() == float("0.00000E+00").hex()   # +0.0
        if i % 2:
            assert blk.u[i, 2].hex() == float("0.00000E+00").hex()


def test_frd_negative_zero_and_plus_sign(tmp_job):
    # -0.00000E+00: биты знака совпадают со strtod; '+' в мантиссе — нестандарт,
    # векторный парсер отвергает, fallback разбирает
    _write_raw_frd(tmp_job.frd, [
        " -1         1-0.00000E+00 0.00000E+00 0.00000E+00",
        " -1         2+1.00000E+00 0.00000E+00 0.00000E+00",
    ])
    blk = next(iter(iter_disp_blocks(tmp_job.frd)))
    assert blk.u[0, 0].hex() == float("-0.00000E+00").hex()      # -0.0, не +0.0
    assert blk.u[1, 0] == 1.0


def test_frd_binary_raises(tmp_job):
    tmp_job.frd.write_bytes(b"\x00\x01\x02\x00binary junk")
    with pytest.raises(ValueError, match="бинарный"):
        list(iter_disp_blocks(tmp_job.frd))


def test_na_writes_report(tmp_job):
    make_inp(tmp_job.inp, [])
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (0, 0, 0)})])
    v = check_job(tmp_job.inp, tmp_job.frd, out_dir=tmp_job.root)
    assert v.verdict == "N/A" and len(v.report_paths) == 2
    data = json.loads(v.report_paths[0].read_text(encoding="utf-8"))
    assert data["verdict"] == "N/A" and data["steps"] == []
    assert data["inputs"]["frd_sha256"]           # frd был — хэш в audit_trace


def test_na_report_without_frd(tmp_job):
    make_inp(tmp_job.inp, [])                      # job.frd не создаём
    v = check_job(tmp_job.inp, out_dir=tmp_job.root)
    assert v.verdict == "N/A" and len(v.report_paths) == 2
    data = json.loads(v.report_paths[0].read_text(encoding="utf-8"))
    assert data["inputs"]["frd"] is None and data["inputs"]["frd_sha256"] is None


def test_cli_fail_prints_worst_equation(tmp_job, capsys):
    make_inp(tmp_job.inp, [[(1, 1, 1.0), (2, 1, -1.0)]])
    make_frd(tmp_job.frd, [(1, "0.1E-01", {1: (5e-4, 0, 0), 2: (4e-4, 0, 0)})])
    assert cli_main([str(tmp_job.inp)]) == 1
    out = capsys.readouterr().out
    assert "худшее уравнение" in out and "u(2,1)" in out
