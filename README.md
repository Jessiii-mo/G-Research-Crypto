# G-Research Crypto

> A leakage-aware machine-learning benchmark for short-horizon crypto return residuals using the Kaggle G-Research Crypto Forecasting dataset.

本项目基于 Kaggle G-Research Crypto Forecasting 的分钟级聚合数据，对多种加密资产的短期收益残差 `Target` 进行预测。项目重点是可复现的数据处理、严格的时间切分、统一模型比较以及对弱预测信号的诚实评估。

## 项目概览

正式 180 天实验已经完成。源窗口包含 3,627,788 条分钟记录；在连续序列上生成特征后，每隔 10 个时间点抽样，共使用 362,573 条记录建模，并在 54,409 条时间外测试记录上统一评估。

## 方法

- 数据窗口：2021-03-25 至 2021-09-21，覆盖 14 种资产。
- 特征：19 个仅使用当前及历史信息的收益、波动率、动量和量价特征。
- 抽样：先在连续分钟序列上计算滚动特征，再每隔 10 个时间点抽样。
- 切分：按时间顺序划分训练、验证和测试集，预处理仅在训练集拟合。
- 模型：零预测基线、线性回归、Ridge、决策树、KNN、小型神经网络。
- 指标：RMSE、MAE、Pearson IC、方向准确率和预测头尾分位的实际 Target 差值。
- 计算约束：KNN 训练集按时间等距限制为 20,000 条，其余模型使用 308,164 条训练与验证记录。

## 运行

从 [Kaggle competition data page](https://www.kaggle.com/competitions/g-research-crypto-forecasting/data) 下载 `train.csv`。原始数据不会包含在仓库中。

```bash
python -m pip install -r requirements.txt
python run_experiment.py --data-path /path/to/train.csv
```

结果写入 `artifacts/`：

- `metrics.csv`：整体模型比较；
- `per_asset_metrics.csv`：分资产指标；
- `model_comparison.png`：RMSE 与 Pearson IC 对比。

## 结果

| 模型 | 测试 RMSE | MAE | Pearson IC | 方向准确率 | 头尾十分位 Target 差 |
|---|---:|---:|---:|---:|---:|
| 零预测基线 | 0.00363039 | 0.00209183 | — | — | — |
| 线性回归 | **0.00363006** | 0.00209299 | **0.0209** | 50.28% | 0.000130 |
| Ridge | 0.00363150 | 0.00209280 | 0.0090 | 50.42% | 0.000180 |
| 决策树 | 0.00364210 | 0.00209556 | -0.0013 | 51.48% | 0.000009 |
| KNN | 0.00371624 | 0.00219412 | 0.0093 | 50.11% | 0.000128 |
| 神经网络 | 0.00363717 | 0.00209584 | 0.0122 | 51.08% | **0.000216** |

![Model comparison](artifacts/model_comparison.png)

线性回归取得最高 Pearson IC，但其 RMSE 仅比零预测基线低约 0.009%。复杂模型没有形成稳定优势，说明仅依赖历史 OHLCV 与成交笔数构造的量价特征，对这一高度噪声化目标的增量预测力有限。相比追求更复杂的模型，后续研究更应关注跨资产、订单簿与市场状态特征，并采用滚动窗口验证。

## 说明

`Target` 是经市场因子残差化后的未来短窗口收益，而非可直接实现的交易收益。本项目用于展示时间序列分析与模型评估流程，不宣称形成可交易策略或获得 Kaggle 排名。
