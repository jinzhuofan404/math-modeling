# B题建模与论文优化设计

> 基于研究报告.txt的结构性反馈，对现有四问建模链路做五个横切面改进。

**目标：** 统一标签口径、补齐统计验证、引入可部署分群，提升模型可信度与论文自洽性。

**约束：** 3-5天工期；不改题目要求（30日留存≥10%目标不变）；全周期K-Means聚类保留作为画像工具。

---

## 架构总览

不改动四问框架，只补"标签层—验证层—部署层"三个横切面：

```
原始日志 (pickdata1 + pickdata2)
  │
  ├─→ 统一标签: duration = min(lifecycle_days, 30), event = lifecycle_days < 30
  │
  ├─→ Day1-3特征表 (问题1生存分析 + 早期分群学生模型)
  │
  └─→ 全周期特征表 (问题2/3的事后解释 + 聚类教师模型)
```

五个改进点嵌入位置：

| # | 改进点 | 嵌入位置 | 产出 |
|---|--------|----------|------|
| 1 | 统一标签口径 | `特征工程.py` + `问题1_求解.py` | Q1改用lifecycle_days做生存标签，全篇统一 |
| 2 | PH诊断+RSF对照 | `问题1_求解.py` | Schoenfeld残差图、分层Cox、RSF C-index |
| 3 | Bootstrap区间 | `问题2_求解.py` | 阈值/AUC/GLM系数/Deviance的95%CI |
| 4 | 早期可部署分群 | `问题3_求解.py` | 全周期标签→Day1-3 XGBoost分类器 |
| 5 | K稳定性比较 | `问题3_求解.py` | K=2..8多指标比较表 |

---

## 改进①：统一标签口径

### 问题

Q1用"连续2天无pickdata2记录"定义流失（30日留存0.85%），Q2/Q3用`lifecycle_days`（首末日期跨度，30日留存~7.1%）。两套口径导致论文前后逻辑断裂。

### 方案

**全篇统一为`lifecycle_days`定义**（Q3的运营口径），Q1重跑，Q2/Q3不变。

### Q1代码改动（`问题1_求解.py`）

- 数据源：从`player_features_day3.csv`切换为`player_features.csv`（取统一标签）
- 特征仍严格使用Day1-3的14个特征列（不改特征体系，仅标签列替换）
- 新增标签计算：
  ```
  duration = min(lifecycle_days, 30)
  event = 1 if lifecycle_days < 30 else 0
  ```
- KM估计、风险率曲线、Cox比例风险模型全部用新标签
- 付费/入盟分层KM曲线同步更新

### Q2改动（`问题2_求解.py`）

- `lifecycle_days`作为协变量的位置改为统一`duration`/`event`
- 其余不变

### Q3改动

- 无。`lifecycle_days >= 30`判定保持原有逻辑。

### 预期数值变化

| 指标 | 旧值 | 新值方向 |
|------|------|---------|
| 次日留存率 | 33.55% | 可能微调 |
| 30日留存率 | 0.85% | ~7.1%附近 |
| 中位生存时间 | 1天 | 可能延长至3-5天 |
| Cox C-index | 0.899 | 待重跑 |

### 论文改动

sec5_problem1.tex：全部留存率数字、KM曲线描述、Cox结果更新

---

## 改进②：PH假设诊断 + 替代模型对照

### PH诊断（`问题1_求解.py`新增）

1. 调用`CoxPHFitter.check_assumptions()`输出每个协变量p值
2. 对违背PH的变量（p<0.05）画Schoenfeld残差散点图
3. 补救：分层Cox（对违背PH的分类变量）或加`var:time`交互项
4. 论文sec5新增"比例风险假设检验"段落

### RSF对照模型

- 用`scikit-survival`的`RandomSurvivalForest`作为非参数对照
- 比较RSF与Cox的C-index
- 论文中讨论"为什么保留Cox为主模型"（解释性优先）

---

## 改进③：Bootstrap不确定性量化

### Q2 Bootstrap（`问题2_求解.py`新增）

| 估计对象 | 方法 | 输出 |
|----------|------|------|
| 钻石阈值566 | Bootstrap 1000轮→每轮Logistic回归→阈值分布 | 95% CI |
| 阶段1 AUC | Bootstrap 1000轮→每轮重新划分7:3训练评估 | AUC的95% CI |
| 阶段2 GLM系数 | Bootstrap 1000轮→每轮重采样付费者→重拟Gamma GLM | 各exp(β)的95% CI |
| 阶段2 Deviance | 同上 | Deviance的95% CI |

### Q3营收Bootstrap（`问题3_求解.py`新增）

- Bootstrap 500轮，每轮MC迭代250次（×10000玩家）
- 输出优化方案期望营收和留存率的95% CI

### 论文改动

sec6新增"不确定性分析"段落：关键系数和指标的区间估计；sec7营收数字附带CI

---

## 改进④：早期可部署分群（教师-学生架构）

### 动机

当前Q3 K-Means使用9个全周期特征（`level_end`、`total_pay`、`vip_level_max`、`diamond_median`等），新服玩家上线时无法获取。需要构建一个仅用Day1-3特征的分类器来预测玩家策略类型。

### 实现（`问题3_求解.py`新增模块）

```
阶段1（教师）：全周期K-Means(K=6) → cluster_label（已有，保留）
阶段2（学生）：XGBoost多分类器(6类), 仅用Day1-3特征预测cluster_label
  - 特征14个：days_logged_d3, level_d3, level_change_d3, avg_duration_d3,
    food/wood/stone/diamond/coins_reduce_d3, diamond_d3, gold_d3,
    is_pay_d3, is_league_d3, n_event_types_d3
  - 评估：分类准确率、混淆矩阵、每类F1
  - 预期准确率低于全周期聚类（诚实报告）
```

### 策略触发改动

- 策略触发不再依赖全周期聚类标签，改用学生分类器的预测类别
- MC模拟仍用全周期标签做ground truth验证

### 论文改动

sec7新增"早期可部署分群"小节，讨论教师-学生架构及准确率trade-off

---

## 改进⑤：聚类数K稳定性比较

### 实现（`问题3_求解.py`修改）

- `K_range`从`range(6,7)`改为`range(2,9)`
- 每个K输出：Silhouette、Calinski-Harabasz、Davies-Bouldin、最小簇占比

### 输出表格

| K | Silhouette | CH Index | DB Index | 最小簇占比 |
|---|-----------|----------|----------|-----------|
| 2 | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |
| 8 | ... | ... | ... | ... |

- 如果K≠6是最优，调整聚类命名和论文叙述
- 如果K=6仍最优，论文中给出选择依据（不再"看起来在选，实际没选"）

### 论文改动

sec7新增"聚类数选择"段落，引用多指标比较表

---

## 论文全局改动

| 章节 | 改动 |
|------|------|
| sec1 摘要 | 更新留存率数字（如次日/7日/30日） |
| sec5 问题1 | 全量更新留存率、KM、Cox数字；新增PH诊断段落(~6行)；新增RSF对照(~4行) |
| sec6 问题2 | Hurdle模型数字可能微调；新增强度不确定性段(~8行) |
| sec7 问题3 | K值选择表格(~6行)；早期分群段落(~8行)；留存数字统一 |
| sec9 模型评价 | 更新数值；新增局限性（两种留存定义统一说明，~4行） |
| 图表 | fig1-4, fig6, fig10重新生成（数值更新）；可能新增Schoenfeld残差图、K比较图 |

---

## 执行顺序

```
步骤1: 统一标签 — 特征工程 + Q1重跑 (→ 新Q1结果)
步骤2: PH诊断 + RSF (基于新Q1结果)
步骤3: K比较 (Q3独立可并行)
步骤4: 早期分群 (Q3, 依赖步骤3的K确定)
步骤5: Bootstrap (Q2 + Q3 MC, 可并行)
步骤6: 图表更新 (nature_figures.py, 基于新数值)
步骤7: 论文全量更新
步骤8: XeLaTeX编译验证 (2遍)
```

## 验证

- `python 问题1_求解.py` — 输出新留存率、C-index，PH诊断结果
- `python 问题2_求解.py` — 输出带CI的bootstrap结果
- `python 问题3_求解.py` — K比较表、早期分群准确率、MC营收CI
- `xelatex main.tex`×2 — 零错误零警告
- 三个问题间的留存数字交叉校验一致
