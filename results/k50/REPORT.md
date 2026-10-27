# Global and local k50 results

Local: k=50 including the center, p=1,2, alpha=0.1,...,0.6. Global: p=1,2, alpha=0.3,...,0.8. Task1 uses 12 geometry features; Task2 adds the same 37 auxiliary inputs.

## Task1: within-protein fitting

| subset | method | proteins | mean_pcc | rmse | mae |
| --- | --- | --- | --- | --- | --- |
| Small | global12 | 33 | 0.894266 | 3.302132 | 1.850156 |
| Small | local12 | 33 | 0.888401 | 3.348688 | 1.936195 |
| Medium | global12 | 37 | 0.728678 | 3.766913 | 2.570109 |
| Medium | local12 | 37 | 0.729114 | 3.760469 | 2.560077 |
| Large | global12 | 35 | 0.679065 | 5.434901 | 2.967949 |
| Large | local12 | 35 | 0.685221 | 5.318175 | 2.894705 |
| Superset | global12 | 364 | 0.724949 | 5.241842 | 3.321010 |
| Superset | local12 | 364 | 0.728132 | 5.181232 | 3.276474 |

## Task2: held-out prediction

| protocol | model | global_pcc | local_pcc | delta_pcc | global_rmse | local_rmse | global_mae | local_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| atom_cv | GBDT | 0.835684 | 0.838890 | 0.003206 | 6.666271 | 6.619929 | 4.626997 | 4.593290 |
| atom_cv | RF | 0.841858 | 0.843171 | 0.001312 | 6.300996 | 6.276774 | 4.317805 | 4.297696 |
| lopo | GBDT | 0.570050 | 0.575309 | 0.005258 | 12.436307 | 12.432088 | 7.512206 | 7.506346 |
| lopo | RF | 0.531420 | 0.537818 | 0.006399 | 13.164799 | 13.148232 | 7.745018 | 7.721240 |
| protein_cv | GBDT | 0.432621 | 0.440608 | 0.007987 | 12.424644 | 12.403655 | 7.499089 | 7.470121 |
| protein_cv | RF | 0.382863 | 0.388193 | 0.005330 | 13.199768 | 13.177761 | 7.824446 | 7.799224 |

PCC averages test folds for ten-fold protocols and proteins for LOPO. RMSE/MAE pool all target rows. Atom CV allows other residues of the same protein in training. Protein CV and LOPO separate proteins, not homology clusters.

## Calibration check

| protocol | case | model | rmse | baseline_rmse |
| --- | --- | --- | --- | --- |
| atom_cv | global49 | GBDT | 6.666271 | 11.662933 |
| atom_cv | global49 | RF | 6.300996 | 11.662933 |
| atom_cv | local49 | GBDT | 6.619929 | 11.662933 |
| atom_cv | local49 | RF | 6.276774 | 11.662933 |
| lopo | global49 | GBDT | 12.436307 | 11.937970 |
| lopo | global49 | RF | 13.164799 | 11.937970 |
| lopo | local49 | GBDT | 12.432088 | 11.937970 |
| lopo | local49 | RF | 13.148232 | 11.937970 |
| protein_cv | global49 | GBDT | 12.424644 | 11.948702 |
| protein_cv | global49 | RF | 13.199768 | 11.948702 |
| protein_cv | local49 | GBDT | 12.403655 | 11.948702 |
| protein_cv | local49 | RF | 13.177761 | 11.948702 |

One seed is evaluated. The configuration was selected in earlier exploration on overlapping proteins; these results are not independent external validation. A higher PCC does not guarantee lower error on every protein or useful absolute calibration. The two alpha banks differ, so changes are not solely attributable to locality.

Detailed numeric outputs are in this directory. Resumable predictions are in cache/k50/task2. Set FORCE_RETRAIN=True in spyder_train.py for new fits; valid existing checkpoints otherwise resume without training. Audit verifies saved predictions rather than proving a fresh training run occurred.
