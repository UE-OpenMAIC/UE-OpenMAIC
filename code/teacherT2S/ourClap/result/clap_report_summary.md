# CLaP Baseline and PID-Meta Report

Formal runs included: HAS2, MITBIH2, SKAB2, TSSB2, UTSA2. Runs under old/ and smoke-only runs are excluded from the formal table.

## Baseline vs PID-Meta

| Dataset | Base ARI | PID ARI | Delta ARI | Base NMI | PID NMI | Delta NMI | Base Covering | PID Covering | Delta Covering | Base AMI | PID AMI | Delta AMI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAS2 | 0.5191 | 0.5205 | 0.0015 | 0.5936 | 0.5806 | -0.0130 | 0.6969 | 0.7046 | 0.0077 | 0.5837 | 0.5724 | -0.0113 |
| MITBIH2 | 0.3517 | 0.2746 | -0.0771 | 0.4209 | 0.4014 | -0.0195 | 0.7191 | 0.6849 | -0.0342 | 0.4039 | 0.3619 | -0.0421 |
| SKAB2 | 0.2946 | 0.3326 | 0.0380 | 0.3648 | 0.3851 | 0.0203 | 0.6335 | 0.6494 | 0.0159 | 0.3543 | 0.3799 | 0.0256 |
| TSSB2 | 0.7545 | 0.7600 | 0.0055 | 0.7744 | 0.7806 | 0.0062 | 0.8551 | 0.8537 | -0.0014 | 0.7706 | 0.7761 | 0.0055 |
| UTSA2 | 0.7662 | 0.7252 | -0.0410 | 0.7938 | 0.7752 | -0.0186 | 0.8244 | 0.7964 | -0.0280 | 0.7783 | 0.7580 | -0.0203 |
| MacroMean | 0.5372 | 0.5226 | -0.0146 | 0.5895 | 0.5845 | -0.0049 | 0.7458 | 0.7378 | -0.0080 | 0.5782 | 0.5697 | -0.0085 |

## PID Selected vs Unselected Branch Means

Branch means are computed per case first, then averaged across cases.

| Dataset | Sel ARI | Unsel ARI | Sel-Unsel ARI | Sel NMI | Unsel NMI | Sel-Unsel NMI | Sel Covering | Unsel Covering | Sel-Unsel Covering | Sel AMI | Unsel AMI | Sel-Unsel AMI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAS2 | 0.5208 | 0.4740 | 0.0467 | 0.5882 | 0.5412 | 0.0470 | 0.7008 | 0.6872 | 0.0137 | 0.5785 | 0.5317 | 0.0469 |
| MITBIH2 | 0.2668 | 0.3050 | -0.0382 | 0.4059 | 0.4251 | -0.0192 | 0.6774 | 0.7069 | -0.0295 | 0.3604 | 0.3894 | -0.0290 |
| SKAB2 | 0.3294 | 0.1511 | 0.1783 | 0.3858 | 0.1720 | 0.2138 | 0.6479 | 0.5257 | 0.1222 | 0.3788 | 0.1690 | 0.2097 |
| TSSB2 | 0.7553 | 0.7340 | 0.0213 | 0.7778 | 0.7623 | 0.0155 | 0.8516 | 0.8500 | 0.0017 | 0.7732 | 0.7577 | 0.0155 |
| UTSA2 | 0.7170 | 0.7251 | -0.0081 | 0.7688 | 0.7585 | 0.0103 | 0.7902 | 0.8170 | -0.0268 | 0.7505 | 0.7426 | 0.0079 |
| MacroMean | 0.5179 | 0.4779 | 0.0400 | 0.5853 | 0.5318 | 0.0535 | 0.7336 | 0.7174 | 0.0163 | 0.5683 | 0.5181 | 0.0502 |

## Output Files

- clap_report_summary_table.csv

- clap_baseline_pid_summary.csv

- clap_selected_unselected_branch_means.csv

- clap_report_summary.png
