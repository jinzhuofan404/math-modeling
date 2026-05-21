# B题：策略类游戏的用户留存与付费策略优化 — 建模分析报告

## 一、问题分析

### 1.1 问题背景

数字娱乐行业中，策略类（SLG）游戏"代号K"即将开启新服。运营团队需要在保证用户留存的前提下，通过资源投放与付费设计，最大化开服首月总收入。核心挑战：留存率与付费率之间的最优平衡。

### 1.2 数据实际情况与题目描述的差异

题目描述了4张表（玩家基础信息、登录日志、付费流水、资源消耗），但实际提供的是两组文件：

| 实际文件 | 内容 | 规模 |
|----------|------|------|
| pickdata1.csv | 资源变更记录（get/reduce），33字段 | ~193万行，2000用户 |
| pickdata2.csv | 用户行为事件记录，52字段 | ~416万行，2000用户 |
| pick_stat1/2.csv | 各用户在两类数据中的记录条数 | 2000行 |
| other_pickdata1/2.csv | 测试集（1000用户，互斥） | ~7GB |

**关键字段映射**（从实际数据到题目需求）：

| 题目需求 | 实际来源 | 提取方式 |
|----------|----------|----------|
| 玩家基础信息 | pickdata2: register_time, total_pay, current_level, first_login_time | 按用户聚合取最新值 |
| 登录日志 | pickdata2: dt, account_id | 按用户+日期聚合，生成每日登录状态 |
| 付费流水 | pickdata1: change_dollar(全为0) + pickdata2: total_pay, first_pay_time | 以pickdata2的total_pay为准 |
| 资源消耗 | pickdata1: resource_name, change_type, change_num | 筛选reduce类型，按资源名+用户+日期聚合 |
| 每日在线时长 | pickdata2: event=client_scene_*, dt | 间接推算 |
| 每日等级 | pickdata1/pickdata2: current_level | 按用户+日期取最新值 |

### 1.3 问题类型判断

| 问题 | 类型 | 子类型 |
|------|------|--------|
| 问题1 | 预测 + 描述性分析 | 生存分析、分类预测 |
| 问题2 | 关联分析 + 回归 | 相关性分析、因果推断、多元回归 |
| 问题3 | 优化 + 仿真 | 聚类、多目标优化、蒙特卡洛模拟 |
| 问题4 | 定性建议 | 数据策略 |

---

## 二、模型选择

### 2.1 问题1：玩家留存规律与流失预测模型

**（1）留存率曲线：Kaplan-Meier 生存估计**

使用 Kaplan-Meier (KM) 估计量计算留存率曲线：

$$
\hat{S}(t) = \prod_{i: t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)
$$

其中 $d_i$ 为 $t_i$ 时刻流失的玩家数，$n_i$ 为 $t_i$ 时刻仍在游戏中的玩家数。

定义"流失" = 玩家连续2天无登录记录。计算次日(1天)、7日、14日、30日留存率。

**（2）流失关键节点识别：风险函数分析**

绘制风险函数（Hazard Rate）曲线：

$$
h(t) = \frac{\text{在}t\text{时刻流失的人数}}{\text{在}t\text{时刻仍在游戏的人数}}
$$

识别 $h(t)$ 的局部极值点作为流失高峰。

**（3）留存天数预测模型：Cox比例风险模型（主模型）**

以玩家前3天行为特征作为协变量，建立Cox回归：

$$
h(t|X) = h_0(t) \exp(\beta_1 X_1 + \beta_2 X_2 + ... + \beta_k X_k)
$$

特征变量 $X$：
- 前3天日均在线时长（从行为记录推算）
- 前3天等级提升速度 = (Day3等级 - Day1等级) / 3
- 前3天资源获取总量（food, wood, stone, gold, diamond）
- 首日是否加入联盟（从 league_name 判断）
- 注册渠道（media_source）
- 首日是否有付费行为

备选模型：**LightGBM 生存回归**，处理非线性关系和特征交互。

评价指标：C-index（一致性指数）、时间依赖的AUC。

**参考文献信息**：

| 论文名称 | 作者 | 年份 | 来源 |
|---------|------|------|------|
| Nonparametric Estimation from Incomplete Observations | Kaplan & Meier | 1958 | Journal of the American Statistical Association |
| Regression Models and Life-Tables | Cox D.R. | 1972 | Journal of the Royal Statistical Society |
| Survival Analysis: A Self-Learning Text | Kleinbaum & Klein | 2012 | Springer |

### 2.2 问题2：资源消耗、成长速度与付费行为的关联分析

**（1）资源-等级量化关系分析**

计算各等级的日均资源消耗量，识别"卡点"：

- 定义"等级停滞"：连续3天等级未变化
- 计算停滞用户与正常用户的资源差异（t检验）
- 构建资源消耗对等级增长的弹性系数：

$$
E = \frac{\Delta \text{Level}}{\Delta \text{Resource}} \cdot \frac{\text{Resource}}{\text{Level}}
$$

钻石存量阈值分析：使用 logistic 回归建模 $P(\text{流失}|\text{钻石存量})$，通过 ROC 曲线确定最优阈值。

**（2）首次付费分析**

描述性统计 + 分组对比：
- 首付时间分布（Kaplan-Meier 以"首付"为事件）
- 首付金额分组的留存时长对比（KM分层 + Log-rank检验）
- 首付等级分布直方图

**（3）总付费金额回归模型**

使用 **XGBoost 回归 + SHAP 值解释**（非线性关系为主），同时辅以多元线性回归做可解释性验证。

候选特征：
- 是否加入联盟（从 league_name 非空判断）
- 日均在线时长（在线天数 × 日均时长）
- 资源缺口 = (消费量 - 获取量) 的高峰程度
- 首付金额、首付时间
- 等级增长速度
- 国家/地区（宏观付费习惯）
- 渠道来源

评价：$R^2$、MAE，以及SHAP特征重要性排序。

**参考文献信息**：

| 论文名称 | 作者 | 年份 | 来源 |
|---------|------|------|------|
| XGBoost: A Scalable Tree Boosting System | Chen & Guestrin | 2016 | KDD |
| A Unified Approach to Interpreting Model Predictions | Lundberg & Lee | 2017 | NeurIPS |
| Econometric Analysis | Greene W.H. | 2018 | Pearson |

### 2.3 问题3：新服首月付费策略设计与收益最大化模型

**（1）玩家聚类**

使用 **K-Means 聚类**（标准化后的特征），聚类数由轮廓系数 + 肘部法则确定。

聚类特征：
- 日均在线时长
- 总付费金额
- 等级增长速度
- 资源消耗强度
- 是否加入联盟
- 生命周期天数

预期聚类（3-5类）：零氪休闲党、微氪月卡党、中氪战力党、高氪核心党。

**（2）多目标优化模型**

**决策变量**：对每个玩家聚类 $c$，在时段 $t$（以天为单位），推送礼包组合 $p$ 的概率或规则。

**约束条件**：

1. 留存率约束：30日总留存率 ≥ 10%
   - 每个聚类 $c$ 的留存率 $r_c$ 是礼包策略的函数，通过历史数据估计弹性
2. 逻辑约束：同一玩家每天最多触发1-2次推送
3. 同类玩家在同时段接收相同策略

**优化目标**：

$$
\max \sum_{c} N_c \cdot \left[ \sum_{t=1}^{30} \sum_{p} P(\text{purchase}|c, t, p) \cdot \text{price}_p \cdot \text{trigger\_prob}_{c,t,p} \right]
$$

其中：
- $N_c$ = 聚类 $c$ 的玩家数（总计 10,000）
- $P(\text{purchase}|c, t, p)$ = 聚类 $c$ 在时段 $t$ 购买礼包 $p$ 的概率（从历史数据估计）
- $\text{price}_p$ = 礼包定价

**（3）蒙特卡洛模拟验证**

基于历史数据的概率分布，进行 10,000 次蒙特卡洛模拟：
- 每次模拟抽样 10,000 名玩家（按其聚类比例 + 行为分布）
- 给定策略方案，模拟30天内的付费行为和留存
- 输出总营收分布和留存率分布
- 计算 $P(\text{总营收} \geq 70,000 | \text{策略})$ 和 $P(\text{留存率} \geq 10\% | \text{策略})$

**解法思路**：
- 策略优化是离散组合问题，使用**遗传算法**或**网格搜索+模拟择优**
- 对每种策略候选，运行蒙特卡洛模拟评估期望收益和约束满足概率

**参考文献信息**：

| 论文名称 | 作者 | 年份 | 来源 |
|---------|------|------|------|
| Monte Carlo Methods | Hammersley & Handscomb | 1964 | Methuen |
| K-Means Clustering | MacQueen J. | 1967 | Berkeley Symposium |
| Multi-Objective Optimization Using Evolutionary Algorithms | Deb K. | 2001 | Wiley |

### 2.4 问题4：数据采集建议与策略闭环

定性分析题。建议优先采集的两类数据：

1. **社交行为数据**（好友互动、联盟聊天频次、组队次数）
   - 理由：社交关系是SLG游戏留存的核心驱动力，可以精确量化社交黏性
   - 修正定价策略：高社交黏性玩家可减少推送频率，低社交黏性玩家靶向社交激励

2. **实时战斗力/进度数据**（PVP胜负记录、建筑升级进度、科技树进度）
   - 理由：准确刻画每个玩家的"进度焦虑"，精确触发资源缺口推送
   - 修正定价策略：在玩家升级受阻（连续N次PVP失败、资源缺口>阈值）时精准推送

---

## 三、算法流程

### 3.1 总体工作流

```
原始数据 (pickdata1/2)
    │
    ├─[数据预处理]─→ 清洗后数据
    │                  │
    ├─[特征工程]───→ 玩家级特征表 (每玩家一行, 30天特征)
    │                  │
    ├─[问题1]─────→ 生存分析: KM曲线 + Cox回归 + 留存预测
    │
    ├─[问题2]─────→ 关联分析: 资源弹性 + 付费回归 + XGBoost/SHAP
    │
    ├─[问题3]─────→ 聚类 + 策略优化 + 蒙特卡洛验证
    │
    └─[问题4]─────→ 数据建议
```

### 3.2 特征工程流程

```
pickdata1 (资源变更):
  → 按 user×date 聚合: 各资源的日获取量、日消耗量
  → 按 user 聚合: 各资源总量、获取/消耗比

pickdata2 (行为事件):
  → 按 user 聚合: total_pay(取max), current_level(取最新)
  → 从 event 列: 在线天数、日均事件数
  → first_pay_time, register_time, last_login_time
  → league_name 非空 → is_in_league

合并 → 玩家级特征表 (N=2000, ~50+特征)
```

---

## 四、数据预处理要点

1. **缺失值处理**：
   - pickdata2 中大量字段有缺失（device-related、OS等），按场景填充或删除
   - change_dollar 在 pickdata1 全为0，付费信息完全从 pickdata2 提取

2. **日期解析**：
   - dt 格式为 "2026-02-16"，统一转为日期类型
   - 定义每个用户的 Day1 = 其 register_time（或首次出现日期）

3. **资源名称映射**：
   - 中文资源名（food=粮食, wood=木材, stone=矿石, coins=金币, diamond=钻石, fund=资金）需统一

4. **流失定义**：
   - 连续2天无任何记录 → 判定为流失
   - 流失日期 = 最后一条记录的日期

5. **大数据处理**：
   - 对 ~2.8GB 文件使用 `pd.read_csv(chunksize=200000)` 分块处理
   - 分块聚合后再合并，避免内存爆炸

---

## 五、实现要求

### 5.1 编程语言
Python 3.x

### 5.2 核心依赖
- pandas, numpy — 数据处理
- scikit-learn — 聚类、回归、预处理
- lifelines — 生存分析（Kaplan-Meier, Cox）
- xgboost — 梯度提升回归 + SHAP
- matplotlib, seaborn — 可视化

### 5.3 输出要求
- 各问题的数值结果表格
- 留存率曲线图、风险函数图、聚类可视化图、营收分布直方图
- 策略方案表（问题3的核心产出）
- 代码按问题分文件：`问题1_留存分析.py`, `问题2_关联分析.py`, `问题3_策略优化.py`

---

## 六、注意事项

1. 题目描述的数据结构与实际文件不完全一致，建模时以实际数据为准
2. pickdata2 的 total_pay 是累计值，需要取每个用户的最后状态
3. 资源消耗数据在 pickdata1 的 change_type='reduce' 记录中
4. 问题3的策略优化需要从历史数据估计购买概率，样本量有限（2000用户中~9%付费），估计可能有偏差
5. 蒙特卡洛模拟的抽样需保留各特征的联合分布（非独立抽样）
6. 测试集（other_pickdata）可用于问题1的模型验证
