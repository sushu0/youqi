# ML/DL Integration Plan

## Suggested reply

可以。我们建议在现有预处理成果基础上先统一生成样本库，然后三条线并行推进：一条做聚类，研究曲线和储层类型分群；一条做识别，完成油层/砂体/岩性等分类；一条做拟合，完成孔隙度、渗透率、含水饱和度等连续参数回归。同时把常用机器学习方法和神经网络基线统一集成到一个实验框架里，做到统一输入、统一评估、统一导出，便于横向对比和后续扩展。

## How it is implemented here

- Input data comes from `_local_run/07_interpretation`.
- The integrated runner is `ml_dl_suite.py`.
- Task outputs are written to `_local_run/09_ml_suite`.

## Included methods

- Clustering: `KMeans`, `Agglomerative`, `Birch`, `GaussianMixture`, `DBSCAN`
- Classification: `LogisticRegression`, `RandomForest`, `HistGradientBoosting`, `LinearSVM`, `KNN`, `DNN-MLP`
- Regression: `LinearRegression`, `Ridge`, `RandomForest`, `HistGradientBoosting`, `LinearSVR`, `DNN-MLP`

## Example runs

```powershell
Set-Location 'd:\文件夹\油气'
& '.\.venv311\Scripts\python.exe' '.\ml_dl_suite.py' --task all
```

Only clustering:

```powershell
& '.\.venv311\Scripts\python.exe' '.\ml_dl_suite.py' --task cluster
```

Oil recognition:

```powershell
& '.\.venv311\Scripts\python.exe' '.\ml_dl_suite.py' --task classify --classification-target OIL_PRE
```

Permeability fitting:

```powershell
& '.\.venv311\Scripts\python.exe' '.\ml_dl_suite.py' --task regress --regression-target PERM_PRE
```

## Notes

- The current DL baseline is a tabular DNN based on `MLPClassifier` and `MLPRegressor`.
- If the team later installs `torch`, this layer can be extended to `1D-CNN`, `LSTM`, or `AutoEncoder + KMeans`.
- For formal comparison, it is better to customize features to avoid label leakage. For example, when predicting `OIL_PRE`, you may want to remove `SW_PERCENT_PRE` and `PERM_PRE` from the feature list and compare raw-log-based models separately.
