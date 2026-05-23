# B题建模与论文优化 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 统一标签口径、补齐统计验证、引入可部署分群，提升模型可信度与论文自洽性。

**Architecture:** 不改动现有四问框架。Q1数据源切换为`player_features.csv`取`lifecycle_days`统一标签；Q1新增PH诊断+RSF对照；Q2新增Bootstrap区间；Q3 K比较+早期分群教师-学生架构。

**Tech Stack:** Python (pandas, numpy, lifelines, scikit-survival, xgboost, statsmodels, scipy), XeLaTeX

---

## 文件修改清单

| 文件 | 改动类型 | 改什么 |
|------|----------|--------|
| `B题/问题1_求解.py` | 重写 | 数据源切换+标签统一+PH诊断+RSF |
| `B题/问题2_求解.py` | 新增模块 | bootstrap_metrics()函数 |
| `B题/问题3_求解.py` | 修改+新增 | K_range扩展+K比较表+train_day3_classifier() |
| `B题/nature_figures.py` | 修改 | 更新受影响图的数据 |
| `论文/sections/sec5_problem1.tex` | 修改 | 全部数字更新+PH+RSF段落 |
| `论文/sections/sec6_problem2.tex` | 修改 | 不确定性量化段落 |
| `论文/sections/sec7_problem3.tex` | 修改 | K比较+早期分群段落 |
| `论文/sections/sec9_evaluation.tex` | 修改 | 数值更新+标签统一说明 |

---

### Task 1: Q1标签统一 — 切换数据源+重跑生存分析

**Files:**
- Modify: `B题/问题1_求解.py:50-58`

- [ ] **Step 1: 修改`load_data()`函数，从全周期特征表取标签**

将第50-58行替换为：

```python
def load_data():
    """Load full-period feature table for unified lifecycle_days labels,
    but use ONLY Day1-3 features for Cox modeling (no future info leakage)."""
    path = os.path.join(DATA_DIR, 'player_features.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Feature table not found: {path}. Run 特征工程.py first.')
    df = pd.read_csv(path)
    
    # Unified label: duration = min(lifecycle_days, 30), event = lifecycle_days < 30
    df['duration'] = df['lifecycle_days'].clip(upper=30)
    df['event_churned'] = (df['lifecycle_days'] < 30).astype(int)
    
    print(f'Loaded {len(df)} players (unified lifecycle_days label)')
    print(f'  Churn rate (event=1): {df["event_churned"].mean()*100:.1f}%')
    print(f'  30-day retention: {(df["lifecycle_days"]>=30).mean()*100:.1f}%')
    return df
```

- [ ] **Step 2: 保持Day1-3特征列不变**

`cox_prediction_model()`中的`feature_cols`不变（第201-207行），因为`player_features.csv`中也有对应的Day1-3特征列。检查列名是否匹配：`days_logged_d3`, `level_d3`, `level_change_d3`, `avg_duration_d3`, `food_reduce_d3`, `wood_reduce_d3`, `stone_reduce_d3`, `diamond_reduce_d3`, `coins_reduce_d3`, `diamond_d3`, `gold_d3`, `is_pay_d3`, `is_league_d3`, `n_event_types_d3`。

需要确认这些列存在于`player_features.csv`中。如果不存在，运行`特征工程.py`重新构建时加入这些列。

- [ ] **Step 3: 分层留存分析的is_pay和is_league字段适配**

`segmented_retention()`函数（第280-339行）使用`is_pay_d3`和`is_league_d3`。`player_features.csv`中是`is_paying`和`is_in_league`的全周期版本。需要改为使用全周期版本（全周期payment/league状态与Q3口径一致）：

```python
# 将 df['is_pay_d3'] → df['is_paying']
# 将 df['is_league_d3'] → df['is_in_league']
```

- [ ] **Step 4: 运行Q1并检查输出**

Run: `cd "c:/Users/Nuo/Desktop/数学建模/B题" && python 问题1_求解.py`

Expected: 新的留存率数字（30日留存应接近~7.1%而非0.85%），中位生存时间可能延长，C-index可能变化。

- [ ] **Step 5: 更新结果文件并提交**

```bash
git add B题/问题1_求解.py B题/results/问题1_results.txt
git commit -m "改进①: Q1统一标签口径, 改用lifecycle_days做生存标签"
```

---

### Task 2: PH诊断 + RSF对照

**Files:**
- Modify: `B题/问题1_求解.py:218` (Cox fit之后新增)
- Install: `pip install scikit-survival`

- [ ] **Step 1: 在Cox拟合后添加PH假设检验**

在`cox_prediction_model()`中，第219行`cph.fit(...)`之后、第220行`print(cph.print_summary())`之前，插入：

```python
    # ── PH Assumption Diagnostics ──
    print('\n  --- PH Assumption Check ---')
    try:
        ph_check = cph.check_assumptions(train_data, p_value_threshold=0.05, show_plots=False)
        # Summarize violations
        if ph_check is None or len(ph_check) == 0:
            print('  All covariates satisfy PH assumption (p>=0.05).')
        else:
            violated_vars = list(ph_check['variable'].unique()) if 'variable' in ph_check.columns else []
            print(f'  PH violated for {len(violated_vars)} covariate(s): {violated_vars}')
    except Exception as e:
        print(f'  PH check failed: {e}')
```

- [ ] **Step 2: 对显著违背PH的变量画Schoenfeld残差图**

在PH检验之后插入：

```python
    # ── Schoenfeld Residuals for worst violator ──
    try:
        from lifelines.plotting import plot_schoenfeld_residuals
        schoenfeld_residuals = cph.compute_residuals(train_data, 'schoenfeld')
        # Plot for each variable with p<0.05 from check_assumptions
        ph_check = cph.check_assumptions(train_data, p_value_threshold=0.05, show_plots=False)
        if ph_check is not None and len(ph_check) > 0:
            worst_var = ph_check.iloc[0]['variable']
            fig, ax = plt.subplots(figsize=(8, 4))
            var_idx = list(train_data.columns).index(worst_var)
            ax.scatter(train_data.index, schoenfeld_residuals[:, var_idx], alpha=0.3, s=10)
            ax.axhline(y=0, color='red', linestyle='--')
            ax.set_xlabel('Rank of duration'); ax.set_ylabel(f'Schoenfeld Residual: {worst_var}')
            ax.set_title(f'Schoenfeld Residuals: {worst_var}')
            plt.tight_layout()
            for lang, fig_dir, _ in LANGS:
                plt.savefig(os.path.join(fig_dir, 'figure_schoenfeld.png'), dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f'  Schoenfeld plot failed: {e}')
```

- [ ] **Step 3: 添加RSF对照模型**

在Cox评估之后插入：

```python
    # ── Random Survival Forest Comparison ──
    print('\n  --- RSF Comparison ---')
    try:
        from sksurv.ensemble import RandomSurvivalForest
        import numpy as np
        
        # Prepare structured array for sksurv
        y_train_struct = np.array(
            [(bool(e), d) for e, d in zip(train_data['event_churned'], train_data['duration'])],
            dtype=[('event', bool), ('time', float)]
        )
        rsf = RandomSurvivalForest(n_estimators=100, min_samples_leaf=5, random_state=RANDOM_SEED)
        rsf.fit(X_train.values, y_train_struct)
        
        # Predict risk scores on test set
        rsf_risk = rsf.predict(X_test.values)
        rsf_c_index = concordance_index(
            df_clean.loc[X_test.index, 'duration'].values,
            -rsf_risk,  # negate because higher risk = lower survival
            df_clean.loc[X_test.index, 'event_churned'].values
        )
        print(f'  RSF C-index: {rsf_c_index:.4f} (Cox: {c_index:.4f})')
    except ImportError:
        print('  scikit-survival not installed; skipping RSF.')
    except Exception as e:
        print(f'  RSF failed: {e}')
```

- [ ] **Step 4: 运行验证**

Run: `cd "c:/Users/Nuo/Desktop/数学建模/B题" && python 问题1_求解.py`

Expected: PH诊断输出（哪些变量违背PH），Schoenfeld图生成，RSF C-index输出。

- [ ] **Step 5: 提交**

```bash
git add B题/问题1_求解.py B题/figures/
git commit -m "改进②: 新增PH假设诊断+Schoenfeld残差+RSF对照"
```

---

### Task 3: K稳定性比较

**Files:**
- Modify: `B题/问题3_求解.py:93`

- [ ] **Step 1: 扩展K范围和仪表板指标**

将第93行改为：

```python
    K_range = range(2, 9)  # compare K=2..8 for stability
    inertias, silhouettes, ch_scores, db_scores, min_cluster_pcts = [], [], [], [], []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        ch_scores.append(calinski_harabasz_score(X_scaled, labels))
        db_scores.append(davies_bouldin_score(X_scaled, labels))
        # min cluster %
        _, counts = np.unique(labels, return_counts=True)
        min_cluster_pcts.append(counts.min() / counts.sum() * 100)
```

需要添加sklearn导入：
```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
```

- [ ] **Step 2: 打印K比较表**

在K循环后插入：

```python
    print(f'\n  K Comparison Table:')
    print(f'  {"K":>4} {"Silhouette":>12} {"CH Index":>12} {"DB Index":>12} {"MinCluster%":>12}')
    print(f'  {"-"*52}')
    for i, k in enumerate(K_range):
        print(f'  {k:>4} {silhouettes[i]:>12.4f} {ch_scores[i]:>12.1f} {db_scores[i]:>12.4f} {min_cluster_pcts[i]:>11.1f}%')
```

- [ ] **Step 3: 更新K选择逻辑**

```python
    # Select K: prioritize silhouette, penalize tiny clusters (<3%)
    valid_k = [(i, k, silhouettes[i]) for i, k in enumerate(K_range) if min_cluster_pcts[i] >= 3.0]
    if valid_k:
        best_i, best_k, _ = max(valid_k, key=lambda x: x[2])
    else:
        best_i = np.argmax(silhouettes)
        best_k = list(K_range)[best_i]
    print(f'  Optimal K = {best_k} (Silhouette={silhouettes[best_i]:.3f}, CH={ch_scores[best_i]:.1f}, DB={db_scores[best_i]:.3f}, MinCluster={min_cluster_pcts[best_i]:.1f}%)')
```

- [ ] **Step 4: 运行验证**

Run: `cd "c:/Users/Nuo/Desktop/数学建模/B题" && python 问题3_求解.py`

Expected: K=2..8的比较表，最优K可能仍是6或变成其他值。

- [ ] **Step 5: 提交**

```bash
git add B题/问题3_求解.py B题/results/问题3_results.txt
git commit -m "改进⑤: K=2..8多指标比较, 修正硬编码K=6"
```

---

### Task 4: 早期可部署分群（教师-学生）

**Files:**
- Modify: `B题/问题3_求解.py` (在`player_clustering()`之后新增函数)

- [ ] **Step 1: 新增`train_day3_classifier()`函数**

在`player_clustering()`和`estimate_demand_curves()`之间插入：

```python
def train_day3_classifier(df, cluster_profiles):
    """Train Day1-3 classifier to predict full-period cluster labels (Teacher-Student).
    
    Teacher: full-period K-Means cluster labels (ground truth)
    Student: XGBoost multi-class classifier using only Day1-3 features
    """
    print('\n' + '=' * 50)
    print('3.2 Early Deployable Segmentation (Teacher-Student)')
    print('=' * 50)
    
    # Load Day1-3 features
    day3_path = os.path.join(DATA_DIR, 'player_features_day3.csv')
    if not os.path.exists(day3_path):
        print(f'  Day3 feature table not found: {day3_path}. Skipping.')
        return None, None
    
    df_day3 = pd.read_csv(day3_path)
    
    # Day1-3 features for the student classifier
    student_feats = [
        'days_logged_d3', 'level_d3', 'level_change_d3', 'avg_duration_d3',
        'food_reduce_d3', 'wood_reduce_d3', 'stone_reduce_d3',
        'diamond_reduce_d3', 'coins_reduce_d3',
        'diamond_d3', 'gold_d3',
        'is_pay_d3', 'is_league_d3', 'n_event_types_d3',
    ]
    
    # Ensure all features present
    missing = [f for f in student_feats if f not in df_day3.columns]
    if missing:
        print(f'  Missing Day3 features: {missing}')
        # Fall back to available columns
        student_feats = [f for f in student_feats if f in df_day3.columns]
    
    # Merge cluster labels from full-period df onto Day3 df
    label_map = df[['account_id', 'cluster']].copy()
    df_day3_m = df_day3.merge(label_map, on='account_id', how='inner')
    
    if len(df_day3_m) == 0:
        print('  No matching account_ids between Day3 and full-period tables.')
        return None, None
    
    X_student = df_day3_m[student_feats].fillna(0)
    y_student = df_day3_m['cluster'].values  # K classes, 0..K-1
    
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Cross-validate
    clf = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                        subsample=0.8, random_state=RANDOM_SEED, verbosity=0)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(clf, X_student, y_student, cv=cv, scoring='accuracy')
    print(f'  Student classifier: 5-fold CV accuracy = {cv_scores.mean():.3f} (+/-{cv_scores.std():.3f})')
    
    # Fit on full data
    clf.fit(X_student, y_student)
    
    # Per-class F1
    y_pred = clf.predict(X_student)
    n_classes = len(np.unique(y_student))
    target_names = [cp['name_cn'] for cp in sorted(cluster_profiles, key=lambda x: x['cluster'])]
    print(f'\n  Classification Report (on full Day3 data):')
    print(classification_report(y_student, y_pred, target_names=target_names[:n_classes], digits=3))
    
    # Save feature importance
    feat_imp = pd.DataFrame({'feature': student_feats, 'importance': clf.feature_importances_})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    print(f'\n  Top-5 Day3 features for cluster prediction:')
    for _, row in feat_imp.head(5).iterrows():
        print(f'    {row["feature"]}: {row["importance"]:.4f}')
    
    return clf, student_feats
```

- [ ] **Step 2: 在`main()`中调用新函数**

在第610行`cluster_profiles, df = player_clustering(df)`之后插入：

```python
    # Early deployable segmentation (Teacher-Student)
    day3_classifier, day3_feats = train_day3_classifier(df, cluster_profiles)
```

- [ ] **Step 3: 运行验证**

Run: `cd "c:/Users/Nuo/Desktop/数学建模/B题" && python 问题3_求解.py`

Expected: 输出Day1-3分类器CV准确率、每类F1、特征重要性。准确率可能50-70%，这是诚实的。

- [ ] **Step 4: 提交**

```bash
git add B题/问题3_求解.py B题/results/问题3_results.txt
git commit -m "改进④: 早期可部署分群, 全周期教师→Day1-3学生分类器"
```

---

### Task 5: Bootstrap置信区间

**Files:**
- Modify: `B题/问题2_求解.py` (新增`bootstrap_q2()`函数)
- Modify: `B题/问题3_求解.py` (新增`bootstrap_q3_revenue()`函数)

- [ ] **Step 1: Q2 Bootstrap — 钻石阈值、AUC、GLM系数**

在`问题2_求解.py`的`main()`之前新增：

```python
def bootstrap_q2(df, n_bootstrap=1000):
    """Bootstrap 95% CI for diamond threshold, stage1 AUC, stage2 GLM coefficients."""
    print('\n' + '=' * 50)
    print('Bootstrap Uncertainty Quantification')
    print('=' * 50)
    
    np.random.seed(RANDOM_SEED)
    n = len(df)
    
    # Storage
    dia_thresholds = []
    aucs = []
    glm_coefs = {name: [] for name in glm_feat_names}
    
    for b in range(n_bootstrap):
        if b % 200 == 0:
            print(f'  Bootstrap: {b}/{n_bootstrap}')
        
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        df_boot = df.iloc[idx].copy()
        
        # 1) Diamond threshold
        df_boot['churned'] = (df_boot['lifecycle_days'] < 7).astype(int)
        df_boot['log_diamond'] = np.log1p(df_boot['diamond_median'])
        dia_data_b = df_boot[['log_diamond', 'diamond_median', 'churned']].dropna()
        
        if len(dia_data_b) > 50:
            lr = LogisticRegression()
            lr.fit(dia_data_b[['log_diamond']].values, dia_data_b['churned'].values)
            dia_range = np.linspace(dia_data_b['log_diamond'].min(), dia_data_b['log_diamond'].max(), 100)
            probs = lr.predict_proba(dia_range.reshape(-1, 1))[:, 1]
            thresh_idx = np.argmin(np.abs(probs - 0.5))
            dia_thresholds.append(np.expm1(dia_range[thresh_idx]))
        
        # 2) Stage1 AUC (on 30% holdout, within bootstrap)
        X_b = df_boot[all_feats].fillna(0)
        y_b = df_boot['is_paying'].values
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_b, y_b, test_size=0.3, random_state=b, stratify=y_b)
            clf_b = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                  scale_pos_weight=(len(y_tr)-y_tr.sum())/max(1,y_tr.sum()),
                                  random_state=RANDOM_SEED, verbosity=0)
            clf_b.fit(X_tr, y_tr)
            y_prob = clf_b.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y_te, y_prob))
        except:
            continue
    
    # Summarize CIs
    def ci95(vals, name):
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f'  {name}: median={np.median(vals):.3f}, 95%CI=[{lo:.3f}, {hi:.3f}]')
        return lo, hi
    
    if dia_thresholds:
        ci95(dia_thresholds, 'Diamond Threshold')
    if aucs:
        ci95(aucs, 'Stage1 AUC')
    
    # Stage2 GLM coefficients bootstrap (on payers only)
    payers_df = df[df['total_pay'] > 0]
    n_payers = len(payers_df)
    if n_payers > 10:
        glm_coef_names = []
        glm_coef_vals = []
        for b in range(min(n_bootstrap, 500)):
            idx_p = np.random.choice(n_payers, size=n_payers, replace=True)
            pay_boot = payers_df.iloc[idx_p]
            # Fit GLM on bootstrap sample (simplified: top-3 features)
            X_p = pay_boot[['diamond_median', 'total_get', 'level_end']].fillna(0)
            X_p = sm.add_constant(StandardScaler().fit_transform(X_p))
            try:
                glm_b = sm.GLM(pay_boot['total_pay'].values, X_p,
                              family=sm.families.Gamma(link=sm.families.links.Log()))
                res_b = glm_b.fit()
                if b == 0:
                    glm_coef_names = ['intercept', 'diamond_median', 'total_get', 'level_end']
                for j, name in enumerate(glm_coef_names[:len(res_b.params)]):
                    if name not in glm_coefs:
                        glm_coefs[name] = []
                    glm_coefs[name].append(res_b.params[j])
            except:
                continue
        
        print('\n  Stage2 GLM Coefficient Bootstrap (95% CI):')
        for name, vals in glm_coefs.items():
            if len(vals) > 10:
                lo, hi = ci95(vals, f'  {name}')
    
    return {'dia_thresholds': dia_thresholds, 'aucs': aucs}
```

这份代码需要放在`xgboost_total_pay()`返回metrics之后调用。`all_feats`和`glm_feat_names`需要从`xgboost_total_pay()`的返回值中获取。

实际实现时，将该函数拆为两部分：`bootstrap_diamond_threshold(df)`和`bootstrap_hurdle_metrics(df, all_feats)`。具体实现细节在代码编辑阶段调整，确保与现有变量名一致。

- [ ] **Step 2: Q3 Bootstrap — 营收CI**

在`问题3_求解.py`的MC之后新增：

```python
def bootstrap_q3_mc(cluster_profiles, df, cl_sizes, cl_dists, beta_tuned, n_bootstrap=200):
    """Bootstrap 95% CI for optimized revenue via resampling players."""
    print('\n' + '=' * 50)
    print('Q3 Bootstrap Revenue CI')
    print('=' * 50)
    
    np.random.seed(RANDOM_SEED)
    n_players = len(df)
    
    revenues = []
    for b in range(n_bootstrap):
        if b % 50 == 0:
            print(f'  Bootstrap: {b}/{n_bootstrap}')
        idx = np.random.choice(n_players, size=n_players, replace=True)
        df_boot = df.iloc[idx].copy()
        
        # Re-run clustering and MC on bootstrap sample (simplified)
        # Use reduced MC iterations for speed
        # Store mean revenue for this bootstrap sample
        # (Implementation simplified — use existing cluster assignment
        #  and demand curves with bootstrap sample sizes)
        
    # Print CI
    # lo, hi = np.percentile(revenues, [2.5, 97.5])
```

实际的简化版bootstrap：对2000个玩家重采样200次，每次用已有的cluster分布和需求曲线做简化的营收计算（不重跑完整MC）。

- [ ] **Step 3: 运行Q2验证**

Run: `cd "c:/Users/Nuo/Desktop/数学建模/B题" && python 问题2_求解.py`

Expected: 输出钻石阈值CI、AUC CI、GLM系数CI。

- [ ] **Step 4: 提交**

```bash
git add B题/问题2_求解.py B题/问题3_求解.py B题/results/
git commit -m "改进③: Q2 Bootstrap区间(阈值/AUC/GLM系数), Q3 Bootstrap营收CI"
```

---

### Task 6: 图表更新

**Files:**
- Modify: `B题/nature_figures.py`

- [ ] **Step 1: 更新受影响图的数值**

需要更新的图：
- fig1 (KM曲线): 留存率数字变化
- fig2 (风险率): 基于新duration
- fig4 (Cox系数): 可能变化
- fig5 (预测vs实际): MAE可能变化
- fig6 (分层留存): 数字变化
- fig8 (钻石流失): 如添加CI标注
- fig12 (聚类): K比较图改为多K曲线
- fig10 (MC结果): 营收CI

具体更新数值在运行各求解脚本后从results文件读取。

- [ ] **Step 2: 重新生成所有受影响的图**

Run: `cd "c:/Users/Nuo/Desktop/数学建模/B题" && python nature_figures.py`

- [ ] **Step 3: 提交**

```bash
git add B题/figures/
git commit -m "图表: 更新受影响图以反映新数值"
```

---

### Task 7: 论文同步更新

**Files:**
- Modify: `论文/sections/sec5_problem1.tex`
- Modify: `论文/sections/sec6_problem2.tex`
- Modify: `论文/sections/sec7_problem3.tex`
- Modify: `论文/sections/sec9_evaluation.tex`
- Modify: `论文/sections/sec1_abstract.tex`

- [ ] **Step 1: 更新sec5 (问题1)**

更新所有留存率数字（从新results文件读取）；新增PH诊断段落（~6行）：说明哪些变量满足PH假设、哪些违背、采取了什么补救措施；新增RSF对照段落（~4行）：RSF C-index vs Cox C-index，说明保留Cox的原因（解释性）。

- [ ] **Step 2: 更新sec6 (问题2)**

新增强度不确定性段（~8行）：报告关键指标bootstrap 95% CI。

- [ ] **Step 3: 更新sec7 (问题3)**

K值选择段（~6行）：多指标比较表，最终选择K=N的理由；早期分群段（~8行）：教师-学生架构，Day1-3分类器准确率。

- [ ] **Step 4: 更新sec9 (评价)**

更新数值；新增局限性说明标签统一过程。

- [ ] **Step 5: 更新sec1 (摘要)**

更新留存率数字为统一后的值。

- [ ] **Step 6: 编译验证**

Run: `cd "c:/Users/Nuo/Desktop/数学建模/论文" && xelatex main.tex && xelatex main.tex`

Expected: 零错误零警告。

- [ ] **Step 7: 最终提交**

```bash
git add 论文/
git commit -m "论文: 全量同步五改进数值+新增PH/Bootstrap/K比较/早期分群段落"
```

---

## 执行顺序

```
Task 1 (统一标签+Q1重跑) → Task 2 (PH+RSF, 依赖1的结果)
Task 1 → Task 4 (早期分群, 依赖1的cluster label)
Task 3 (K比较, 独立)
Task 1 → Task 2 → Task 5 (Bootstrap, 在验证层之后)
Task 1-5 → Task 6 (图表, 依赖所有数值)
Task 6 → Task 7 (论文, 最后同步)
```

Task 3和Task 4可并行执行（均在Q3文件但互不依赖）。Task 5在Task 2之后执行。

## 验证

- `python B题/问题1_求解.py` — 新留存率、C-index、PH诊断输出、RSF C-index
- `python B题/问题2_求解.py` — 带bootstrap CI的结果
- `python B题/问题3_求解.py` — K=2..8比较表、早期分群准确率、营收
- `cd 论文 && xelatex main.tex && xelatex main.tex` — 零错误零警告
- 三个问题间留存数字交叉一致
