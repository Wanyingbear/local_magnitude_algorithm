# Local Magnitude

[English version](README.md)

独立的Cα点云global/local weighting与B-factor实验包。整个目录可移动；不需要外部仓库、原文脚本或原工作区outputs。包内只保留运行代码、固定输入、缓存和当前结果，稿件与来源说明不入包。

## Spyder运行顺序

1. spyder_prepare.py：从坐标计算两种特征，完成Task1并准备Task2。
2. spyder_train.py：训练global49/local49的RF/GBDT。
3. spyder_audit.py：独立检查预测、划分与评分。

打开文件按F5。已有特征时可直接train→audit。当前解释器需安装numpy、pandas、scipy、scikit-learn、threadpoolctl。

spyder_train.py设置WORKERS=4（普通笔记本可用2）、PROTOCOLS=['protein_cv','atom_cv','lopo']、AGGREGATE_ONLY=False、FORCE_RETRAIN=False。

- FORCE_RETRAIN=False：复用本包有效缓存，打印New fits、cached、imported。
- FORCE_RETRAIN=True：选中协议的global和local全部重新拟合，替换检查点；不再从外部整体基线复用。要保留旧运行请先备份cache/k50。
- AGGREGATE_ONLY=True：只汇总完整有效缓存；缺失则报错。
- 全套1460个模型，RF/GBDT各1000棵树，seed=0。先试十折可只选protein_cv、atom_cv。导入缓存不等于新训练。

## 输出位置

**只需查看results/k50/**：

| 文件 | 用途 |
| --- | --- |
| task1_summary.csv | Task1四个子集评分 |
| task1_metrics.csv | Task1逐蛋白结果和散点图数据 |
| task2_metrics.csv | 主PCC、pooled PCC、RMSE、MAE和均值基线 |
| comparison.csv | global/local六项直接比较，不混入文献值 |
| task2_per_protein.csv | Task2逐蛋白评分 |
| lopo_subsets.csv | LOPO分组 |
| feature_diagnostics.csv | 特征范围和饱和度 |
| validation.json | 本次验证状态 |
| run.json | 最近训练的新拟合和缓存数量 |
| REPORT.md | 简洁结果说明 |

完整audit应为passed=true、complete=true、fold_prediction_files_checked=1460、测试行各覆盖一次。部分协议audit标记scope和complete=false。prepare/train改动输出前会使旧验证失效。

只保留k50实验。不生成半径12Å等参数搜索、重复逐原子CSV、阶段性报告和全数据计时。用于断点续跑及核验的1460个压缩预测文件位于cache/k50/task2；每个含test_indices、prediction、training_mean、metadata。按test_indices连接data/task2.npz即可定位原子与目标。

## 配置与口径

- local：包含中心的最近50个Cα，p=1,2，alpha=0.1,...,0.6；不足50用全部。
- global：p=1,2，alpha=0.3,...,0.8，维持论文原有实验设置。
- K=exp(-alpha*d**p)，Cholesky解Kw=1；局部取中心weighting，不是邻域总和。
- Task1：364蛋白、78403行，自身标签上的12维OLS拟合，不是盲测。
- Task2：345蛋白、74527行，12维加相同的37维辅助输入，目标不作为特征。
- 几何计算float64；树模型输入缓存float32，与模型内部表示一致。
- 十折primary_pcc为折平均，LOPO为逐蛋白平均；pooled PCC不能直接替换。
- 原子十折允许同蛋白其他残基训练。单seed、历史选参涉及重叠蛋白，不是独立外部验证。
- 软件版本可能影响新训练结果；导入缓存保留来源，audit不会重新训练。

## 目录

~~~text
algorithms/          local_ca_regular.py、global_ca.py
pipeline/            prepare、train、audit、common
data/task1/          xyz、y、source_rows
data/task2.npz       xyz、y、groups、auxiliary、source_rows
data/datasets.json   名单、子集和辅助列定义
cache/k50/           特征、拟合和预测检查点
results/k50/         当前运行结果
experiment.json      统一配置
~~~

包内只保留Task1/Task2链路的代码、固定输入、缓存与结果；计时脚本、示例数据和字节码缓存不入包。

results/k50为当前运行结果，包内不含LaTeX稿件及固定evidence快照，论文数字需在外部稿件目录单独维护，不随本包运行自动改写。

## 算法直接调用与校验

函数调用 `local_ca_regular(xyz, k=50, n_jobs=4)` 可按邻域并行；默认 `n_jobs=1`。多工作线程时，建议外层使用 `threadpoolctl.threadpool_limits(limits=1)` 限制 BLAS 线程。

verify_bundle.py：固定输入/算法哈希和独立LU验证。主动修改源码后可加--skip-hashes。manifest不锁定可变结果。

~~~python
from algorithms import local_ca_regular, global_ca
local = local_ca_regular(xyz, k=50)
global_weights = global_ca(xyz)
~~~

~~~text
python -m pip install numpy pandas scipy scikit-learn threadpoolctl
python verify_bundle.py
python run_experiment.py prepare --workers 4
python run_experiment.py train --workers 4
python run_experiment.py train --protocols protein_cv --force-retrain --workers 2
python run_experiment.py audit
~~~
