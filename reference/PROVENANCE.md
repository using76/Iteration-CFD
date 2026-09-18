# reference/ - the answer keys `ofgpu-validate` is held against

Every published number a gate in `rust/src/bin/validate.rs` (or a test in
`rust/src`) compares against is a row of the table below, whether it lives in
a file under this directory (`kind = file`) or is still typed into a source
file as a constant (`kind = literal`). The table is read by machine:
`rust/src/bin/validate_key/mod.rs` parses it at run time, and its test
`every_answer_key_is_named_in_the_manifest_and_every_row_has_its_marker` holds
it against the `// answer-key: <id>` markers in the source, in both directions.

Rules:

- one row per key; `id` is the marker text and is unique;
- for `kind = file`, `file` is a path relative to this directory; for
  `kind = literal` it names the source file and the constant(s);
- `sha256` is the lower-case hex SHA-256 of the file's bytes exactly as
  tracked (LF line endings, see `.gitattributes`), as `sha256sum` prints it;
  `-` for a literal;
- a gate whose file is absent, or whose digest is not this table's, reports
  its verdict as open by name and is never passed (SPEC-LIT section 10);
- a key file is the paper's numbers as printed, never edited: a station a
  comparison leaves out is flagged in a column, not removed;
- everything else under `reference/` (clones such as `reference/fds`) is
  git-ignored and not distributed: see `.gitignore` and `NOTICE`.

| id | kind | file | source | DOI/URL | licence | sha256 |
|---|---|---|---|---|---|---|
| ghia1982-table-I | file | ghia1982/table_I_u.csv | U. Ghia, K. N. Ghia and C. T. Shin, "High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method", J. Comput. Phys. 48 (1982) 387-411, Table I (u on the vertical centreline x = 0.5), columns Re = 100 and Re = 400 only | no DOI is quoted: not in the programme's DOI ledger | the paper is (c) 1982 Academic Press; the tabulated values are transcribed as published numerical data, as SPEC-LIT section 10 has done since the first release | 391011116f4f3533527486772470f220df73b471328f8424dec89f0d7286a082 |
| ghia1982-table-II | file | ghia1982/table_II_v.csv | the same paper, Table II (v on the horizontal centreline y = 0.5), columns Re = 100 and Re = 400 only; the Re = 400 entry at x = 0.9063 is reproduced as printed and flagged excluded_re400 = 1 (it breaks the monotone run its neighbours show; Nilsson and Wallin, Uppsala University report 22015 (2022) section 5.2 exclude the same station) | no DOI is quoted: not in the programme's DOI ledger | as above | 1a18def0ff1a375dd6f4c532c5f75f1c6443132d1383969fd759c0e33045ce05 |
| devahldavis1983 | literal | rust/src/bin/validate.rs PUBLISHED = 2.243 (Gate 59-A) | G. de Vahl Davis, Int. J. Numer. Meth. Fluids 3 (1983) 249-264, Ra = 1e4 mean Nu, quoted from Qi et al., Nanoscale Research Letters 8 (2013) 56, Table 3 | DOI 10.1002/fld.1650030305 (primary, paywalled); DOI 10.1186/1556-276X-8-56 (the secondary the number is read from, open access) | primary paywalled; the secondary is open access, one number quoted | - |
| belazizia2012 | literal | rust/src/bin/validate.rs PUBLISHED: [(Scalar, Scalar); 3] (Gate 5, SPEC-LIT section 60) | A. Belazizia, S. Benissaad and S. Abboudi, Adv. Theor. Appl. Mech. 5 (2012) no. 4, 179-190, Fig. 6 digitised at Ra = 1e4, D = 0.2, Pr = 0.7 - a SECONDARY source for Kaminski and Prakash, Int. J. Heat Mass Transfer 29 (1986) 1979-1988 | primary DOI 10.1016/0017-9310(86)90017-7 (paywalled); secondary at m-hikari.com/atam/atam2012/atam1-4-2012/ | secondary open access; three digitised points | - |
| qu-mudawar2002 | literal | rust/src/bin/validate.rs QM_L, QM_Y, QM_Z, QM_RE, QM_TIN, QM_QPP, QM_KF, QM_KS (Gate 6); QM_RHO, QM_CP, QM_MU are SPEC-LIT section 79 Disclosure 2 (water at 20 C), not the paper's | W. Qu and I. Mudawar, Int. J. Heat Mass Transfer 45 (2002), Tables 1-2 (geometry and operating conditions) | DOI 10.1016/S0017-9310(02)00101-1 | the paper is (c) Elsevier; geometry and operating conditions transcribed as data | - |
| qu-mudawar2002-fig4 | literal | rust/src/bin/validate.rs QM_FIG4 (Gate 6) | the same paper, Fig. 4(b) and 4(c) at Re ~ 140, digitised: Kawano et al.'s marker, its error-bar ends, Qu and Mudawar's curve - SPEC-LIT section 79 Disclosure 1 | DOI 10.1016/S0017-9310(02)00101-1 | as above; two digitised rows | - |
| fds-fan-test | literal | rust/src/fan/tests.rs FanCurve::quadratic(10.0, 0.16), q_fds = 0.0498253, dp_fds = 2 * 4.51513 (Gate 52-B) | NIST Fire Dynamics Simulator verification suite, Verification/HVAC/fan_test.fds (MAX_FLOW = 0.16, MAX_PRESSURE = 10) and its published fan_test.csv (vflow1 = -0.0498253, pres_1 = 4.51513) | github.com/firemodels/fds | US-government work (NIST), public domain; the deck and CSV are data and are not distributed here | - |
| ghia1982-simple | literal | rust/src/simple.rs GHIA_U_RE100, GHIA_V_RE100 (the lib regression simple::tests::lid_driven_cavity_matches_ghia_ghia_and_shin) | a second transcription of the Re = 100 columns of the two file rows above; the record of truth is ghia1982/table_I_u.csv and ghia1982/table_II_v.csv | as the two file rows above | as the two file rows above | - |
