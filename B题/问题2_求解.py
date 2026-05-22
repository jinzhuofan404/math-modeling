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
from xgboost import XGBRegressor
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


def first_pay_analysis(df):
    """First payment analysis."""
    print('\n' + '=' * 50)
    print('2.2 First Payment Analysis')
    print('=' * 50)

    payers = df[df['is_paying'] == 1].copy()
    print(f'  Paying users: {len(payers)} ({len(payers)/len(df)*100:.1f}%)')
    if len(payers) == 0:
        return
    print(f'  Mean pay: {payers["total_pay"].mean():.2f}, Median: {payers["total_pay"].median():.2f}')
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
    """XGBoost regression for total_pay with SHAP. Log-transform target for skewed distribution."""
    print('\n' + '=' * 50)
    print('2.3 XGBoost + SHAP: Key Factors of Total Pay')
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
    y_raw = df_model['total_pay'].values
    y_log = np.log1p(y_raw)  # log-transform target

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y_raw, test_size=0.3, random_state=RANDOM_SEED)
    _, _, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.3, random_state=RANDOM_SEED)

    # Train on log-transformed target
    model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED, verbosity=0)
    model.fit(X_train, y_train_log)
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)  # transform back

    mae = mean_absolute_error(y_test_raw, y_pred)
    # R2 on non-zero pay only for meaningful metric
    mask_nonzero = y_test_raw > 0
    r2_nonzero = r2_score(y_test_raw[mask_nonzero], y_pred[mask_nonzero]) if mask_nonzero.sum() > 5 else 0
    print(f'  XGBoost Performance (log-target):')
    print(f'    MAE: {mae:.4f}')
    print(f'    R2 (payers only): {r2_nonzero:.4f}')
    print(f'    Non-zero payers in test: {mask_nonzero.sum()}')

    # Feature importance (SHAP)
    print('\n  Computing SHAP...')
    explainer = shap.TreeExplainer(model)
    shap_subset = X_test.sample(min(200, len(X_test)), random_state=RANDOM_SEED)
    shap_values = explainer.shap_values(shap_subset)

    importance = model.feature_importances_
    top_idx = np.argsort(importance)[-15:][::-1]
    shap_mean = np.abs(shap_values).mean(axis=0)
    top_shap = np.argsort(shap_mean)[-10:][::-1]
    print('\n  Top 10 by |SHAP|:')
    for i in top_shap:
        print(f'    {all_feats[i]}: {shap_mean[i]:.4f}')

    # Bilingual feature importance
    for lang, fig_dir in LANGS:
        set_font(lang)
        if lang == 'cn':
            title = 'XGBoost: 影响总付费金额的关键特征'
            xlabel = '特征重要性'
        else:
            title = 'XGBoost: Top Features for Total Pay'
            xlabel = 'Feature Importance'

        fig, ax = plt.subplots(figsize=(10, 8))
        top_feats_disp = [all_feats[i] for i in top_idx]
        ax.barh(range(len(top_feats_disp)), importance[top_idx], color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(top_feats_disp)))
        ax.set_yticklabels(top_feats_disp, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.invert_yaxis()
        ax.grid(True, linestyle='--', alpha=0.4, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'figure10_xgb_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Bilingual pred vs actual
    for lang, fig_dir in LANGS:
        set_font(lang)
        if lang == 'cn':
            title = f'XGBoost: 预测 vs 实际 (MAE={mae:.2f})'
            xl, yl = '实际总付费(美元)', '预测总付费(美元)'
        else:
            title = f'XGBoost: Predicted vs Actual (MAE={mae:.2f})'
            xl, yl = 'Actual Total Pay (USD)', 'Predicted Total Pay (USD)'

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test_raw, y_pred, alpha=0.3, s=20, c='steelblue', edgecolors='none')
        ax.plot([0, y_test_raw.max()], [0, y_test_raw.max()], 'r--', linewidth=1)
        ax.set_xlabel(xl, fontsize=12); ax.set_ylabel(yl, fontsize=12)
        ax.set_title(title, fontsize=14); ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'figure11_xgb_pred.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return model, {'MAE': mae, 'R2_payers': r2_nonzero}


def main():
    print('=' * 60)
    print('Problem 2: Resource & Payment Correlation Analysis')
    print('=' * 60)

    df = load_features()
    print(f'Loaded {len(df)} players')

    corr_df, level_stats = resource_level_elasticity(df)
    dia_threshold = diamond_churn_threshold(df)
    first_pay_analysis(df)
    model, metrics = xgboost_total_pay(df)

    results = [
        '=== 问题2 结果 ===',
        f'资源-等级增长相关系数: ' + ', '.join([f'{c["Resource_CN"]} r={c["r"]:.3f}' for c in corr_df]),
        f'钻石50%流失阈值: {dia_threshold:.0f}',
        f'XGBoost MAE: {metrics["MAE"]:.4f}, R2(付费者): {metrics["R2_payers"]:.4f}',
    ]
    with open(os.path.join(RES_DIR, '问题2_results.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    print('\nResults saved.')


if __name__ == '__main__':
    main()
