# Time2State baseline for ActRecTut

Put this folder under:

```text
D:\code\teacherT2S\baseline\t2s\actrectut\
```

Also copy `_shard/t2s_runner_selected.py` from this package into:

```text
D:\code\teacherT2S\baseline\t2s\_shard\t2s_runner_selected.py
```

Run:

```text
RUN_T2S_ACTRECTUT_NO_FLASH_LOGGER.cmd
```

Output:

```text
results_t2s_actrectut_paper_grid
```

This baseline uses one Time2State model per paper-grid setting and does not use multi-branch selection, PEER/PID, top-k branches, or meta clustering.
