"""
问题2：资源消耗、成长速度与付费行为的关联分析
方法：量化分析 + 首次付费分析 + XGBoost回归 + SHAP解释
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
import shap

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['lines.linewidth'] = 2

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_CN = os.path.join(BASE_DIR, 'figures', '中文版')
FIG_EN = os.path.join(BASE_DIR, 'figures', '英文版')
RES_DIR = os.path.join(BASE_DIR, 'results')
for d in [FIG_CN, FIG_EN, RES_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_SEED = 42
LANGS = [('cn', FIG_CN), ('en', FIG_EN)]


def set_font(lang):
    if lang == 'cn':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei', 'Arial']
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Arial Unicode MS', 'SimHei', 'Microsoft YaHei']


def load_features():
    path = os.path.join(DATA_DIR, 'player_features.csv')
    return pd.read_csv(path)


def resource_level_elasticity(df):
    """(1) Resource consumption vs level progression."""
    print('\n' + '=' * 50)
    print('2.1 Resource-Level Elasticity Analysis')
    print('=' * 50)

    df['daily_food'] = df['food_reduce'] / df['lifecycle_days'].clip(lower=1)
    df['daily_wood'] = df['wood_reduce'] / df['lifecycle_days'].clip(lower=1)
    df['daily_stone'] = df['stone_reduce'] / df['lifecycle_days'].clip(lower=1)
    df['daily_coins'] = df['coins_reduce'] / df['lifecycle_days'].clip(lower=1)

    res_cols = ['daily_food', 'daily_wood', 'daily_stone', 'diamond_median', 'diamond_reduce']
    res_labels_cn = ['粮食', '木材', '矿石', '钻石(中位数)', '钻石(消耗)']
    res_labels_en = ['Food', 'Wood', 'Stone', 'Diamond (Med)', 'Diamond (Used)']

    print('\n  Correlations with Level Growth:')
    cors = []
    for col, lb_cn, lb_en in zip(res_cols, res_labels_cn, res_labels_en):
        mask = df[col] > 0
        if mask.sum() > 10:
            corr, pval = stats.pearsonr(df.loc[mask, col], df.loc[mask, 'level_growth'])
            cors.append({'Resource_CN': lb_cn, 'Resource_EN': lb_en, 'r': corr, 'p': pval})
            print(f'    {lb_cn}: r={corr:.3f}, p={pval:.4f}')

    # Level statistics
    level_stats = df.groupby('level_end').agg(
        n=('account_id', 'count'),
        mean_growth=('level_growth', 'mean'),
        mean_daily_food=('daily_food', 'mean'),
        mean_daily_wood=('daily_wood', 'mean'),
        mean_daily_stone=('daily_stone', 'mean'),
    ).reset_index()
    level_stats = level_stats[(level_stats['level_end'] > 0) & (level_stats['n'] >= 3)]

    # Bottlenecks: levels where mean growth < previous
    level_stats['growth_change'] = level_stats['mean_growth'].diff()
    bottlenecks = level_stats[level_stats['growth_change'] < -0.1]
    if len(bottlenecks) > 0:
        print(f'\n  Bottleneck levels: {list(bottlenecks["level_end"].values)}')

    # Bilingual plot
    for lang, fig_dir in LANGS:
        set_font(lang)
        if lang == 'cn':
            t1, t2 = '各等级玩家日均资源消耗量', '各等级玩家等级增长速度'
            x1, y1 = '玩家等级', '日均资源消耗量'
            x2, y2 = '玩家等级', '平均等级增长速度'
            fl, wl, sl = '粮食', '木材', '矿石'
        else:
            t1, t2 = 'Resource Consumption by Player Level', 'Level Growth Rate by Player Level'
            x1, y1 = 'Player Level', 'Mean Daily Resource Consumption'
            x2, y2 = 'Player Level', 'Mean Level Growth'
            fl, wl, sl = 'Food', 'Wood', 'Stone'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        levels = level_stats['level_end'].values
        ax1.plot(levels, level_stats['mean_daily_food'], 'o-', color='#2196F3', label=fl, markersize=5)
        ax1.plot(levels, level_stats['mean_daily_wood'], 's-', color='#4CAF50', label=wl, markersize=5)
        ax1.plot(levels, level_stats['mean_daily_stone'], '^-', color='#FF9800', label=sl, markersize=5)
        ax1.set_xlabel(x1, fontsize=12); ax1.set_ylabel(y1, fontsize=12)
        ax1.set_title(t1, fontsize=14); ax1.legend(fontsize=9); ax1.grid(True, linestyle='--', alpha=0.4)

        ax2.plot(levels, level_stats['mean_growth'], 'D-', color='#E91E63', markersize=5, linewidth=2)
        ax2.set_xlabel(x2, fontsize=12); ax2.set_ylabel(y2, fontsize=12)
        ax2.set_title(t2, fontsize=14); ax2.grid(True, linestyle='--', alpha=0.4)
        for _, row in bottlenecks.iterrows():
            ax2.axvline(x=row['level_end'], color='red', linestyle=':', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'figure7_resource_level.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return cors, level_stats


def diamond_churn_threshold(df):
    """Diamond-churn risk analysis."""
    print('\n' + '=' * 50)
    print('2.1b Diamond-Churn Risk')
    print('=' * 50)

    df['churned'] = (df['lifecycle_days'] < 7).astype(int)
    df['log_diamond'] = np.log1p(df['diamond_median'])
    dia_data = df[['log_diamond', 'diamond_median', 'churned']].dropna()
    dia_data = dia_data[dia_data['log_diamond'] > -np.inf]

    threshold = 0
    if len(dia_data) > 50:
        X_dia = dia_data[['log_diamond']].values
        y_dia = dia_data['churned'].values
        lr = LogisticRegression()
        lr.fit(X_dia, y_dia)
        dia_range = np.linspace(X_dia.min(), X_dia.max(), 100)
        probs = lr.predict_proba(dia_range.reshape(-1, 1))[:, 1]
        thresh_idx = np.argmin(np.abs(probs - 0.5))
        threshold = np.expm1(dia_range[thresh_idx])
        print(f'  Diamond threshold (50% churn): {threshold:.0f}')

    # Bilingual plot
    for lang, fig_dir in LANGS:
        set_font(lang)
        if lang == 'cn':
            t1, t2 = '钻石分布：留存 vs 流失玩家', f'流失概率 vs 钻石 (阈值~{threshold:.0f})'
            x1, y1 = 'log(1+钻石)', '密度'
            x2, y2 = '钻石中位数', '7天流失概率'
            sv, ch = '留存(>=7天)', '流失(<7天)'
        else:
            t1, t2 = 'Diamond Distribution: Survived vs Churned', f'Churn Probability vs Diamond (Threshold~{threshold:.0f})'
            x1, y1 = 'log(1+Diamond)', 'Density'
            x2, y2 = 'Median Diamond Count', 'P(Churn in 7 days)'
            sv, ch = 'Survived (>=7d)', 'Churned (<7d)'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for mask, label, color in [(df['churned']==0, sv, '#4CAF50'), (df['churned']==1, ch, '#F44336')]:
            sub = df[mask]
            ax1.hist(np.log1p(sub['diamond_median'].clip(lower=0)), bins=40, alpha=0.6, color=color, label=label, density=True)
        ax1.set_xlabel(x1, fontsize=12); ax1.set_ylabel(y1, fontsize=12)
        ax1.set_title(t1, fontsize=14); ax1.legend(fontsize=10); ax1.grid(True, linestyle='--', alpha=0.4)

        ax2.plot(np.expm1(dia_range), probs, 'b-', linewidth=2)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax2.axvline(x=threshold, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel(x2, fontsize=12); ax2.set_ylabel(y2, fontsize=12)
        ax2.set_title(t2, fontsize=14); ax2.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'figure8_diamond_churn.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return threshold


def recovery_pack_analysis(df, threshold):
    """Analyze optimal recovery pack for players below diamond churn threshold."""
    print('\n' + '=' * 50)
    print('2.1c Optimal Recovery Pack for Diamond-Depleted Players')
    print('=' * 50)

    low_dia = df[df['diamond_median'] < threshold].copy()
    high_dia = df[df['diamond_median'] >= threshold].copy()
    n_low = len(low_dia)
    print(f'  Players below diamond threshold ({threshold:.0f}): {n_low} ({n_low/len(df)*100:.1f}%)')
    churn_rate_low = (low_dia['lifecycle_days'] < 7).mean()
    churn_rate_high = (high_dia['lifecycle_days'] < 7).mean()
    print(f'  7-day churn rate: below={churn_rate_low:.1%}, above={churn_rate_high:.1%}')
    print(f'  Mean lifecycle: below={low_dia["lifecycle_days"].mean():.1f}d, above={high_dia["lifecycle_days"].mean():.1f}d')

    # Resource profile of diamond-depleted players
    print(f'\n  Resource profile (mean daily, below vs above threshold):')
    for res, label in [('daily_food', 'Food'), ('daily_wood', 'Wood'),
                       ('daily_stone', 'Stone'), ('daily_coins', 'Coins')]:
        dia_col = f'diamond_{"median" if "median" in res else "reduce"}'
        val_low = low_dia[res].mean() if res in low_dia.columns else 0
        val_high = high_dia[res].mean() if res in high_dia.columns else 0
        ratio = val_low / val_high if val_high > 0 else float('inf')
        print(f'    {label}: {val_low:.1f} vs {val_high:.1f} (ratio={ratio:.2f})')

    # Paying behavior
    pay_rate_low = low_dia['is_paying'].mean()
    pay_rate_high = high_dia['is_paying'].mean()
    print(f'\n  Pay rate: below={pay_rate_low:.1%}, above={pay_rate_high:.1%}')

    # Key conclusion: based on stagnation analysis (diamond as bottleneck breaker)
    # and churn threshold, the optimal recovery pack is a low-price diamond补给包
    print(f'\n  Recommendation: Low-price diamond补给包 (e.g., 6-30 CNY)')
    print(f'  Rationale: Diamond is the bottleneck breaker (stagnation beta=-0.355)')
    print(f'  {n_low} players ({n_low/len(df)*100:.1f}%) are at elevated churn risk')
    print(f'  Diamond replenishment directly addresses the resource scarcity causing churn')

    return {
        'n_below_threshold': n_low,
        'churn_rate_below': churn_rate_low,
        'churn_rate_above': churn_rate_high,
        'mean_lifecycle_below': low_dia['lifecycle_days'].mean(),
    }


def first_pay_analysis(df):
    """First payment analysis: timing, level, and retention impact."""
    print('\n' + '=' * 50)
    print('2.2 First Payment Analysis')
    print('=' * 50)

    payers = df[df['is_paying'] == 1].copy()
    print(f'  Paying users: {len(payers)} ({len(payers)/len(df)*100:.1f}%)')
    if len(payers) == 0:
        return
    print(f'  Mean pay: {payers["total_pay"].mean():.2f}, Median: {payers["total_pay"].median():.2f}')

    # First payment timing distribution
    if 'first_pay_time' in payers.columns:
        payers['fpt_day'] = pd.to_numeric(payers['first_pay_time'], errors='coerce')
        valid_fpt = payers[payers['fpt_day'].notna() & (payers['fpt_day'] >= 0)]
        if len(valid_fpt) > 0:
            print(f'\n  First Pay Timing:')
            print(f'    Mean day: {valid_fpt["fpt_day"].mean():.2f}, Median: {valid_fpt["fpt_day"].median():.0f}')
            # Day buckets
            day_bins = [0, 1, 3, 7, 14, 30, 100]
            day_labels = ['Day0', 'Day1-2', 'Day3-6', 'Day7-13', 'Day14-29', 'Day30+']
            valid_fpt['day_bucket'] = pd.cut(valid_fpt['fpt_day'], bins=day_bins, labels=day_labels)
            bucket_counts = valid_fpt['day_bucket'].value_counts().sort_index()
            for bucket, count in bucket_counts.items():
                print(f'      {bucket}: {count} payers ({count/len(valid_fpt)*100:.1f}%)')
        else:
            print('  No valid first_pay_time data')

    # First payment level
    if 'first_pay_level' in payers.columns:
        fpl = payers['first_pay_level'].value_counts().sort_index()
        print(f'\n  First Pay Level Distribution:')
        for level, count in fpl.items():
            print(f'    Level {level}: {count} payers ({count/len(payers)*100:.1f}%)')

    payers['pay_group'] = pd.cut(payers['total_pay'], bins=[0, 6, 30, 100, 500], labels=['<6', '6-30', '30-100', '100+'])

    # Bilingual plot
    for lang, fig_dir in LANGS:
        set_font(lang)
        if lang == 'cn':
            t1, t2 = '付费金额分布', '不同付费水平下的留存时长'
            x1, y1 = '总付费(美元)', '玩家数'
            x2, y2 = '付费分组(美元)', '平均生命周期(天)'
        else:
            t1, t2 = 'Payment Amount Distribution', 'Retention by Payment Level'
            x1, y1 = 'Total Pay (USD)', 'Number of Players'
            x2, y2 = 'Payment Group (USD)', 'Mean Lifecycle (days)'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.hist(payers['total_pay'].clip(upper=100), bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        ax1.set_xlabel(x1, fontsize=12); ax1.set_ylabel(y1, fontsize=12)
        ax1.set_title(t1, fontsize=14); ax1.grid(True, linestyle='--', alpha=0.4, axis='y')

        groups = payers.groupby('pay_group')['lifecycle_days'].agg(['mean', 'std', 'count']).reset_index()
        ax2.bar(range(len(groups)), groups['mean'], yerr=groups['std'],
                color=['#90CAF9', '#42A5F5', '#1E88E5', '#0D47A1'], capsize=5, alpha=0.9, edgecolor='white')
        ax2.set_xticks(range(len(groups))); ax2.set_xticklabels(groups['pay_group'], fontsize=10)
        ax2.set_xlabel(x2, fontsize=12); ax2.set_ylabel(y2, fontsize=12)
        ax2.set_title(t2, fontsize=14); ax2.grid(True, linestyle='--', alpha=0.4, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'figure9_first_pay.png'), dpi=300, bbox_inches='tight')
        plt.close()


def xgboost_total_pay(df):
    """Two-stage Hurdle model: Stage1 XGBoost binary classifier (is_paying), Stage2 XGBoost regressor (total_pay | payers)."""
    print('\n' + '=' * 50)
    print('2.3 XGBoost Hurdle Model: Payment Incidence + Payment Amount')
    print('=' * 50)

    feat_cols = [
        'days_active', 'lifecycle_days', 'level_end', 'level_growth', 'level_growth_rate',
        'current_level_max', 'is_in_league', 'vip_level_max', 'n_event_types',
        'food_get', 'food_reduce', 'wood_get', 'wood_reduce', 'stone_get', 'stone_reduce',
        'coins_get', 'coins_reduce', 'diamond_get', 'diamond_reduce', 'diamond_median',
        'total_get', 'total_reduce', 'duration_times',
    ]
    df['resource_intensity'] = df['total_reduce'] / df['lifecycle_days'].clip(lower=1)
    df['diamond_flow'] = df['diamond_get'] - df['diamond_reduce']
    df['coins_flow'] = df['coins_get'] - df['coins_reduce']
    df['log_total_get'] = np.log1p(df['total_get'])
    df['log_total_reduce'] = np.log1p(df['total_reduce'])

    extra = ['resource_intensity', 'diamond_flow', 'coins_flow', 'log_total_get', 'log_total_reduce']
    all_feats = feat_cols + extra

    df_model = df[all_feats + ['total_pay', 'is_paying']].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    X = df_model[all_feats]
    y_clf = df_model['is_paying'].values
    y_reg_raw = df_model['total_pay'].values

    X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg = train_test_split(
        X, y_clf, y_reg_raw, test_size=0.3, random_state=RANDOM_SEED, stratify=y_clf)

    # ---- Stage 1: Binary Classifier ----
    n_pos = y_train_clf.sum()
    scale_weight = (len(y_train_clf) - n_pos) / n_pos if n_pos > 0 else 1
    print(f'\n  --- Stage 1: Payment Incidence Classifier ---')
    print(f'  Train: {len(y_train_clf)} samples, {n_pos} payers ({(n_pos/len(y_train_clf))*100:.1f}%)')
    print(f'  scale_pos_weight: {scale_weight:.1f}')

    clf = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_weight,
                        random_state=RANDOM_SEED, verbosity=0, eval_metric='logloss')
    clf.fit(X_train, y_train_clf)

    y_prob = clf.predict_proba(X_test)[:, 1]
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
    auc = roc_auc_score(y_test_clf, y_prob)
    y_pred_clf = clf.predict(X_test)
    print(f'  AUC-ROC: {auc:.4f}')
    print(f'\n  Classification Report:')
    print(classification_report(y_test_clf, y_pred_clf, target_names=['Non-payer', 'Payer'], digits=4))

    # Stage 1 SHAP
    print('  Computing Stage 1 SHAP...')
    explainer_clf = shap.TreeExplainer(clf)
    shap_subset = X_test.sample(min(200, len(X_test)), random_state=RANDOM_SEED)
    shap_clf_vals = explainer_clf.shap_values(shap_subset)
    shap_clf_mean = np.abs(shap_clf_vals).mean(axis=0)

    # ---- Stage 2: Gamma GLM with Log Link (Belotti et al. 2015) ----
    print(f'\n  --- Stage 2: Gamma GLM Log-Link (Payers Only) ---')
    import statsmodels.api as sm
    from scipy import stats as scipy_stats
    from sklearn.feature_selection import SelectKBest, f_regression

    train_payer_mask = y_train_reg > 0
    test_payer_mask = y_test_reg > 0
    X_train_payers_df = X_train[train_payer_mask]
    y_train_payers_raw = y_train_reg[train_payer_mask]
    X_test_payers_df = X_test[test_payer_mask]
    y_test_payers_raw = y_test_reg[test_payer_mask]

    n_payers_train = len(X_train_payers_df)
    n_payers_test = len(X_test_payers_df)
    print(f'  Training payers: {n_payers_train}, Test payers: {n_payers_test}')

    # Feature pre-screen: top-5 by univariate F-score, then drop collinear pairs
    selector = SelectKBest(f_regression, k=min(5, n_payers_train // 8))
    X_train_sel_raw = selector.fit_transform(X_train_payers_df, y_train_payers_raw)
    X_test_sel_raw = selector.transform(X_test_payers_df)
    selected_idx = np.where(selector.get_support())[0]
    selected_names = [all_feats[i] for i in selected_idx]
    print(f'  Pre-screened to {len(selected_idx)} features: {selected_names}')

    # Drop highly collinear features (corr > 0.9) from screened set
    keep_mask = np.ones(len(selected_names), dtype=bool)
    corr_mat = np.corrcoef(X_train_sel_raw.T)
    for i in range(len(selected_names)):
        if not keep_mask[i]:
            continue
        for j in range(i+1, len(selected_names)):
            if abs(corr_mat[i, j]) > 0.9:
                keep_mask[j] = False
    X_train_sel = X_train_sel_raw[:, keep_mask]
    X_test_sel = X_test_sel_raw[:, keep_mask]
    selected_names = [n for n, k in zip(selected_names, keep_mask) if k]
    print(f'  After collinearity filter: {selected_names}')

    # Standardize features for GLM (raw features have extreme scale differences)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    # Gamma GLM with log link: E[Y|X] = exp(X*beta), Y ~ Gamma
    X_train_glm = sm.add_constant(X_train_scaled)
    X_test_glm = sm.add_constant(X_test_scaled)
    glm_feat_names = ['intercept'] + selected_names

    glm = sm.GLM(y_train_payers_raw, X_train_glm,
                 family=sm.families.Gamma(link=sm.families.links.Log()))
    glm_result = glm.fit()

    # Deviance explained
    null_dev = glm_result.null_deviance
    res_dev = glm_result.deviance
    dev_explained = 1 - res_dev / null_dev if null_dev > 0 else 0

    print(f'\n  Gamma GLM Results:')
    print(f'    Null deviance: {null_dev:.2f}, Residual deviance: {res_dev:.2f}')
    print(f'    Deviance explained: {dev_explained:.3f}')
    print(f'    Log-likelihood: {glm_result.llf:.2f}')

    # Coefficients and p-values
    print(f'\n  {"Feature":30s} {"Coeff":>8s} {"exp(Coeff)":>10s} {"p-value":>8s}')
    print(f'  {"-"*56}')
    sig_coefs = []
    for name, coef, pval in zip(glm_feat_names, glm_result.params, glm_result.pvalues):
        exp_coef = np.exp(coef)
        sig = '*' if pval < 0.1 else ''
        print(f'  {name:30s} {coef:>+8.4f} {exp_coef:>10.4f} {pval:>8.4f} {sig}')
        if pval < 0.2:
            sig_coefs.append((name, coef, exp_coef, pval))
    print(f'  * p<0.1')
    print(f'\n  exp(Coeff) interpretation: factor change in E[total_pay] per unit increase in X')
    print(f'    e.g., exp(Coeff)=1.5 means feature +1 unit -> expected pay x1.5')

    # Predictions on test set (original scale)
    y_pred_glm = glm_result.predict(X_test_glm)
    mae = mean_absolute_error(y_test_payers_raw, y_pred_glm)
    # RMSE on original scale
    rmse = np.sqrt(np.mean((y_test_payers_raw - y_pred_glm)**2))

    print(f'\n  Test Set Performance:')
    print(f'    MAE:  {mae:.4f}')
    print(f'    RMSE: {rmse:.4f}')

    # ---- Combined Results ----
    top_clf = np.argsort(shap_clf_mean)[::-1]  # all features, ascending |SHAP|
    print('\n  Top 10 by |SHAP| (Stage 1: Pay Probability):')
    for i in top_clf:
        print(f'    {all_feats[i]}: {shap_clf_mean[i]:.4f}')

    # ---- Bilingual Figure 1: Stage 1 SHAP (Payment Probability) ----
    for lang, fig_dir in LANGS:
        set_font(lang)
        if lang == 'cn':
            title = 'XGBoost阶段1: 影响付费概率的全部特征 (|SHAP|)'
            xlabel = '|SHAP| 平均边际贡献'
        else:
            title = 'XGBoost Stage 1: All Features for Payment Probability (|SHAP|)'
            xlabel = 'Mean |SHAP| Contribution'

        fig, ax = plt.subplots(figsize=(10, 11))
        names = [all_feats[i] for i in top_clf][::-1]
        vals = shap_clf_mean[top_clf][::-1]
        ax.barh(range(len(names)), vals, color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.invert_yaxis()
        ax.grid(True, linestyle='--', alpha=0.4, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'figure10_xgb_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ---- Bilingual Figure 2: GLM coefficients (Payment Amount | Payer) ----
    # Show only p<0.2 coefficients (exclude intercept)
    sig_names = [n for n, c, e, p in sig_coefs if n != 'intercept']
    sig_vals = [e for n, c, e, p in sig_coefs if n != 'intercept']  # exp(coeff)
    sig_p = [p for n, c, e, p in sig_coefs if n != 'intercept']

    for lang, fig_dir in LANGS:
        set_font(lang)
        if lang == 'cn':
            title = f'阶段2: Gamma GLM 付费金额因子 (exp(β), p<0.2)'
            xlabel = 'exp(β) 乘数效应 (每单位X变化, 期望付费×exp(β))'
        else:
            title = f'Stage 2: Gamma GLM Payment Amount Factors (exp(β), p<0.2)'
            xlabel = 'exp(β) Multiplier (per unit X, expected pay ×exp(β))'

        if len(sig_names) > 0:
            fig, ax = plt.subplots(figsize=(10, max(4, len(sig_names) * 0.5)))
            names = sig_names[::-1]
            vals = [v - 1 for v in sig_vals[::-1]]  # show as % change (exp(beta)-1)
            colors = ['#B64342' if v > 0 else '#4A90D9' for v in vals]
            ax.barh(range(len(names)), vals, color=colors, alpha=0.85)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel('% change in expected pay (exp(β)-1)', fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.axvline(x=0, color='black', linewidth=0.8)
            ax.grid(True, linestyle='--', alpha=0.4, axis='x')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'figure11_xgb_reg_stage2.png'), dpi=300, bbox_inches='tight')
            plt.close()

    metrics = {
        'AUC': auc, 'MAE_payers': mae, 'RMSE_payers': rmse,
        'dev_explained': dev_explained, 'n_payers_train': n_payers_train,
        'glm_params': dict(zip(glm_feat_names, zip(glm_result.params, glm_result.pvalues))),
    }
    return clf, glm_result, metrics


def main():
    print('=' * 60)
    print('Problem 2: Resource & Payment Correlation Analysis')
    print('=' * 60)

    df = load_features()
    print(f'Loaded {len(df)} players')

    corr_df, level_stats = resource_level_elasticity(df)
    dia_threshold = diamond_churn_threshold(df)
    recovery_info = recovery_pack_analysis(df, dia_threshold)
    first_pay_analysis(df)
    model, reg, metrics = xgboost_total_pay(df)

    results = [
        '=== 问题2 结果 ===',
        f'资源-等级增长相关系数: ' + ', '.join([f'{c["Resource_CN"]} r={c["r"]:.3f}' for c in corr_df]),
        f'钻石50%流失阈值: {dia_threshold:.0f}',
        f'钻石阈值以下玩家: {recovery_info["n_below_threshold"]}人, 流失率{recovery_info["churn_rate_below"]:.1%}',
        f'XGBoost Hurdle模型:',
        f'  阶段1(付费概率) AUC-ROC: {metrics["AUC"]:.4f}',
        f'  阶段2(Gamma GLM) Deviance explained: {metrics["dev_explained"]:.3f}, RMSE: {metrics["RMSE_payers"]:.4f}',
        f'  阶段2训练付费者数: {metrics["n_payers_train"]}',
    ]
    with open(os.path.join(RES_DIR, '问题2_results.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    print('\nResults saved.')


if __name__ == '__main__':
    main()
