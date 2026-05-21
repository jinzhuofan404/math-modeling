# 术语表 (Terminology) — B题

本文档用于确保数学建模全过程的术语一致性。

## 一、核心概念术语

| 中文名称 | 英文名称 | 缩写符号 | 定义/说明 |
|---------|---------|---------|----------|
| 用户留存率 | User Retention Rate | $R(t)$ | 注册后第t天仍活跃的用户比例 |
| 次日留存率 | Day-1 Retention Rate | $R_1$ | 注册后第1天仍活跃的用户比例 |
| 7日留存率 | Day-7 Retention Rate | $R_7$ | 注册后第7天仍活跃的用户比例 |
| 30日留存率 | Day-30 Retention Rate | $R_{30}$ | 注册后第30天仍活跃的用户比例 |
| 用户流失 | User Churn | — | 用户连续2天无登录记录 |
| 用户生命周期价值 | Lifetime Value | LTV | 用户从注册到流失期间的总付费金额 |
| 每用户平均收入 | Average Revenue Per User | ARPU | 总营收 / 总用户数 |
| 首充 | First Purchase | — | 用户的首次付费行为 |

## 二、数学符号术语

| 符号 | 含义 | 英文名称 |
|-----|------|----------|
| $S(t)$ | t时刻的生存函数（留存率） | Survival Function |
| $h(t)$ | t时刻的风险函数 | Hazard Function |
| $h_0(t)$ | 基准风险函数 | Baseline Hazard |
| $\beta_k$ | 第k个协变量的回归系数 | Regression Coefficient |
| $X_k$ | 第k个特征变量 | Covariate |
| $N_c$ | 聚类c的用户数 | Cluster Size |
| $r_c$ | 聚类c的留存率 | Cluster Retention Rate |
| $E$ | 资源-等级弹性系数 | Resource-Level Elasticity |
| $\hat{y}_i$ | 对第i个样本的预测值 | Predicted Value |

## 三、模型相关术语

| 中文术语 | 英文术语 | 缩写 |
|---------|---------|------|
| 生存分析 | Survival Analysis | SA |
| Kaplan-Meier估计 | Kaplan-Meier Estimator | KM |
| Cox比例风险模型 | Cox Proportional Hazards Model | Cox PH |
| 对数秩检验 | Log-Rank Test | — |
| K均值聚类 | K-Means Clustering | K-Means |
| 轮廓系数 | Silhouette Score | — |
| 多目标优化 | Multi-Objective Optimization | MOO |
| 蒙特卡洛模拟 | Monte Carlo Simulation | MC |
| XGBoost回归 | Extreme Gradient Boosting Regression | XGB |
| SHAP值 | SHapley Additive exPlanations | SHAP |

## 四、数据字段术语

| 字段 | 中文含义 | 数据类型 | 来源 |
|------|---------|---------|------|
| account_id | 用户账号ID | int64 | 两表共有 |
| resource_name | 资源名称 | string | pickdata1 |
| change_type | 资源变更类型(get/reduce) | string | pickdata1 |
| change_num | 变更数量 | float64 | pickdata1 |
| change_reason | 变更原因(数字编码) | int64 | pickdata1 |
| dt | 日期 | string | 两表共有 |
| current_level | 当前等级 | int64 | 两表共有 |
| total_pay | 累计付费总额(元) | float64 | pickdata2 |
| first_pay_time | 首次付费时间 | string | pickdata2 |
| league_name | 联盟名称 | string | pickdata1 |
| event | 行为事件类型 | string | pickdata2 |
| current_diamond | 当前钻石数量 | float64 | pickdata2 |
| current_gold | 当前金币数量 | float64 | pickdata2 |

## 五、资源名称术语

| 英文名称 | 中文名称 | 说明 |
|---------|---------|------|
| food | 粮食 | 基础资源，训练士兵 |
| wood | 木材 | 基础资源，建造升级 |
| stone | 矿石 | 基础资源，建造升级 |
| coins / gold | 金币 | 可购买缺失的基础资源 |
| diamond | 钻石 | 稀缺货币，购买加速/高级装备 |
| fund | 资金 | 游戏内通用货币 |
| heal_point | 治疗点数 | 治疗伤兵 |
| pvp_point | PVP点数 | PVP竞技积分 |
