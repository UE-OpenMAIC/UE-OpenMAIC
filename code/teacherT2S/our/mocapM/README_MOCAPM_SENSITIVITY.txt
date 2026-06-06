MoCapM Top-K fusion-branch sensitivity scripts, 2-4-8 version

Copy all files in this folder to:
  D:\code\teacherT2S\our\mocapM

Main command:
  RUN_MOCAP_TOPK_SENSITIVITY.cmd

Purpose:
  Run Top-K fusion-branch sensitivity for ER-MSSF on MoCap.
  The default Top-K list is now:
    2,4,8

Default P/I/S reliability weights are kept unchanged:
  P=0.45, I=0.35, S=0.20

Usage:
  1) Smoke test / first 1 MoCap case only:
     RUN_MOCAP_TOPK_SENSITIVITY.cmd
     RUN_MOCAP_TOPK_SENSITIVITY.cmd quick

  2) Full MoCap Top-K sensitivity for paper:
     RUN_MOCAP_TOPK_SENSITIVITY.cmd full

  3) Custom Top-K list, first 1 case:
     RUN_MOCAP_TOPK_SENSITIVITY.cmd quick 2,4,8

  4) Custom Top-K list, full dataset:
     RUN_MOCAP_TOPK_SENSITIVITY.cmd full 2,4,8

Generated variants:
  topk_02
  topk_04
  topk_08

Outputs:
  D:\code\teacherT2S\our\mocapM\_topk_sensitivity\topk_sensitivity_summary.csv
  D:\code\teacherT2S\our\mocapM\_topk_sensitivity\topk_sensitivity_summary.xlsx
  D:\code\teacherT2S\our\mocapM\_topk_sensitivity\topk_sensitivity_latex_rows.tex

Notes:
  - Default double-click mode still runs only the first 1 case for smoke testing.
  - For the paper parameter sensitivity table, use:
      RUN_MOCAP_TOPK_SENSITIVITY.cmd full
  - Do not combine --max-series 1 with --only-case-ids, because max-series is applied before case-id filtering.
