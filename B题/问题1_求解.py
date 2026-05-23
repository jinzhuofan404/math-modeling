"""
问题1：玩家留存规律与流失预测模型
方法：Kaplan-Meier生存估计 + Cox比例风险模型 + 流失预测
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Plot config
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
LANGS = [
    ('cn', FIG_CN, 'SimHei'),
    ('en', FIG_EN, 'DejaVu Sans'),
]


def set_font(lang):
    if lang == 'cn':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei', 'Arial']
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Arial Unicode MS', 'SimHei', 'Microsoft YaHei']


def load_data():
    """Load full-period feature table for unified lifecycle_days labels,
    but use ONLY Day1-3 features for Cox modeling (no future info leakage)."""
    # Load full-period table for unified labels
    path_full = os.path.join(DATA_DIR, 'player_features.csv')
    if not os.path.exists(path_full):
        raise FileNotFoundError(f'Feature table not found: {path_full}. Run 特征工程.py first.')
    df_full = pd.read_csv(path_full)

    # Load Day3-only table for Day1-3 features
    path_d3 = os.path.join(DATA_DIR, 'player_features_day3.csv')
    if not os.path.exists(path_d3):
        raise FileNotFoundError(f'Day3 feature table not found: {path_d3}. Run 特征工程.py first.')
    df_d3 = pd.read_csv(path_d3)

    # Merge: labels (lifecycle_days, is_paying, is_in_league) from full table,
    # Day1-3 features from Day3 table
    d3_feature_cols = [c for c in df_d3.columns if c not in ('duration', 'event_churned', 'country', 'platform', 'channel_id', 'media_source')]
    df = df_full[['account_id', 'lifecycle_days', 'is_paying', 'is_in_league']].merge(
        df_d3[d3_feature_cols], on='account_id', how='inner'
    )
    print(f'  Merge: {len(df_full)} full + {len(df_d3)} d3 -> {len(df)} joined')

    # Unified label: duration = min(lifecycle_days, 30), event = lifecycle_days < 30
    df['duration'] = df['lifecycle_days'].clip(upper=30)
    df['event_churned'] = (df['lifecycle_days'] < 30).astype(int)

    print(f'Loaded {len(df)} players (unified lifecycle_days label)')
    print(f'  Churn rate (event=1): {df["event_churned"].mean()*100:.1f}%')
    print(f'  30-day retention: {(df["lifecycle_days"]>=30).mean()*100:.1f}%')
    return df


def km_survival_curve(df):
    """(1) KM survival curve with bilingual output."""
    print('\n' + '=' * 50)
    print('1.1 Kaplan-Meier Survival Analysis')
    print('=' * 50)

    kmf = KaplanMeierFitter()
    T = df['duration'].values
    E = df['event_churned'].values
    kmf.fit(T, E, label='All Players')

    for day in [1, 7, 14, 30]:
        surv = kmf.survival_function_at_times(day).values[0]
        print(f'  Day-{day} retention: {surv*100:.2f}%')
    median_surv = kmf.median_survival_time_
    print(f'  Median survival: {median_surv:.1f} days')

    # Bilingual KM curve
    label_dicts = {
        'cn': {
            'title': 'Kaplan-Meier 玩家留存曲线',
            'xlabel': '注册后天数', 'ylabel': '留存概率',
            'd1': '第1天', 'd7': '第7天', 'd14': '第14天', 'd30': '第30天',
            'line_label': '所有玩家',
        },
        'en': {
            'title': 'Kaplan-Meier Player Retention Curve',
            'xlabel': 'Days Since Registration', 'ylabel': 'Survival Probability',
            'd1': 'Day 1', 'd7': 'Day 7', 'd14': 'Day 14', 'd30': 'Day 30',
            'line_label': 'All Players',
        }
    }
    for lang, fig_dir, _ in LANGS:
        set_font(lang)
        lb = label_dicts[lang]
        fig, ax = plt.subplots(figsize=(8, 6))
        kmf.plot_survival_function(ax=ax, ci_show=True, label=lb['line_label'])
        for day_val, lb_key, color in [(1, 'd1', 'red'), (7, 'd7', 'orange'), (14, 'd14', 'green'), (30, 'd30', 'blue')]:
            ax.axvline(x=day_val, color=color, linestyle='--', alpha=0.4, label=lb[lb_key])
        ax.set_xlabel(lb['xlabel'], fontsize=12)
        ax.set_ylabel(lb['ylabel'], fontsize=12)
        ax.set_title(lb['title'], fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'figure1_km_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Bilingual hazard rate
    naf = NelsonAalenFitter()
    naf.fit(T, E)
    haz = naf.cumulative_hazard_.diff().fillna(0)
    haz_smooth = haz.rolling(window=3, center=True).mean()
    top_peaks = haz_smooth.nlargest(5, haz_smooth.columns[0])

    label_dicts_h = {
        'cn': {'title': '玩家流失风险率随时间变化', 'xlabel': '注册后天数', 'ylabel': '风险率'},
        'en': {'title': 'Player Churn Hazard Rate Over Time', 'xlabel': 'Days Since Registration', 'ylabel': 'Hazard Rate'},
    }
    for lang, fig_dir, _ in LANGS:
        set_font(lang)
        lb = label_dicts_h[lang]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(haz_smooth.index, haz_smooth.values, 'b-', linewidth=1.5)
        ax.set_xlabel(lb['xlabel'], fontsize=12)
        ax.set_ylabel(lb['ylabel'], fontsize=12)
        ax.set_title(lb['title'], fontsize=14)
        for day in top_peaks.index:
            ax.axvline(x=day, color='red', linestyle=':', alpha=0.5)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(0, min(60, T.max()))
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'figure2_hazard_rate.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f'  Top churn risk days: {list(top_peaks.index[:5])}')
    return kmf


def churn_key_moments(df):
    """(2) Survival span distribution with bilingual output."""
    print('\n' + '=' * 50)
    print('1.2 Survival Span Distribution')
    print('=' * 50)

    active_dist = df['days_logged_d3'].value_counts().sort_index()
    # Count players whose lifecycle_days >= each day threshold
    days_arr = np.arange(1, 31)
    day_counts = [len(df[df['duration'] >= d]) for d in days_arr]
    # Players whose lifecycle ended exactly at each day (i.e. lifecycle_days < day)
    day_drops = [day_counts[i-1] - day_counts[i] for i in range(1, len(day_counts))]
    day_drop_rate = [day_drops[i] / max(day_counts[i], 1) for i in range(len(day_drops))]
    top_drops = sorted(zip(range(2, 31), day_drop_rate), key=lambda x: x[1], reverse=True)[:5]
    print('  Highest lifecycle-end rate days (fraction whose lifecycle ended at day N):')
    for day, rate in top_drops:
        print(f'    Day {day}: {rate*100:.1f}% of surviving players ended at this day')

    label_dicts = {
        'cn': {
            't1': '玩家活跃天数分布', 't2': '各天数阈值留存玩家数',
            'x1': '活跃天数', 'y1': '玩家数',
            'x2': '活跃天数阈值', 'y2': '存活≥X天的玩家数',
        },
        'en': {
            't1': 'Distribution of Player Active Days', 't2': 'Players Surviving >= Day Threshold',
            'x1': 'Active Days', 'y1': 'Number of Players',
            'x2': 'Lifecycle Days', 'y2': 'Players Surviving >= Day X',
        }
    }
    for lang, fig_dir, _ in LANGS:
        set_font(lang)
        lb = label_dicts[lang]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.bar(active_dist.index[:30], active_dist.values[:30], color='steelblue', edgecolor='white', alpha=0.8)
        ax1.set_xlabel(lb['x1'], fontsize=12)
        ax1.set_ylabel(lb['y1'], fontsize=12)
        ax1.set_title(lb['t1'], fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.4, axis='y')
        ax2.plot(range(1, 31), day_counts, 'b-o', markersize=4, linewidth=1.5)
        ax2.set_xlabel(lb['x2'], fontsize=12)
        ax2.set_ylabel(lb['y2'], fontsize=12)
        ax2.set_title(lb['t2'], fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'figure3_active_days.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return top_drops


def cox_prediction_model(df):
    """(3) Cox PH model with bilingual output."""
    print('\n' + '=' * 50)
    print('1.3 Cox Proportional Hazards Prediction Model')
    print('=' * 50)

    # ── Strictly Day1-3 features only (no future info) ──
    df_model = df.copy()
    # All features are from Day1-3 window. No lifecycle_days, level_end,
    # total_pay, vip_level_max, or any full-period variables.
    feature_cols = [
        'days_logged_d3', 'level_d3', 'level_change_d3', 'avg_duration_d3',
        'food_reduce_d3', 'wood_reduce_d3', 'stone_reduce_d3',
        'diamond_reduce_d3', 'coins_reduce_d3',
        'diamond_d3', 'gold_d3',
        'is_pay_d3', 'is_league_d3', 'n_event_types_d3',
    ]

    df_clean = df_model.dropna(subset=feature_cols + ['duration', 'event_churned'])
    df_clean = df_clean[~df_clean[feature_cols].isin([np.inf, -np.inf]).any(axis=1)]
    X = df_clean[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

    X_train, X_test = train_test_split(X_scaled, test_size=0.3, random_state=RANDOM_SEED)
    train_data = pd.concat([X_train, df_clean.loc[X_train.index, ['duration', 'event_churned']]], axis=1)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_data, duration_col='duration', event_col='event_churned', show_progress=False)
    print(cph.print_summary())

    # ── PH Assumption Diagnostics ──
    print('\n  --- PH Assumption Check ---')
    violated_vars = []
    try:
        from lifelines.statistics import proportional_hazard_test
        ph_result = proportional_hazard_test(cph, train_data, time_transform='rank')
        summary = getattr(ph_result, 'summary', None)
        if summary is not None and len(summary) > 0:
            p_vals = summary['p']
            for i, p in enumerate(p_vals):
                is_sig = False
                try:
                    is_sig = float(p) < 0.05
                except (ValueError, TypeError):
                    is_sig = isinstance(p, str) and p.strip().startswith('<')
                if is_sig:
                    idx = summary.index[i]
                    if isinstance(idx, tuple):
                        var_name = idx[0]
                    else:
                        var_name = str(idx)
                    if var_name not in violated_vars:
                        violated_vars.append(var_name)
        if violated_vars:
            print(f'  PH violated for {len(violated_vars)} covariate(s): {violated_vars}')
        else:
            print('  All covariates satisfy PH assumption (p>=0.05).')
    except Exception as e:
        print(f'  PH check failed: {e}')

    if violated_vars:
        try:
            schoenfeld_residuals = cph.compute_residuals(train_data, 'schoenfeld')
            worst_var = violated_vars[0]
            # schoenfeld_residuals is a DataFrame (lifelines >=0.29); extract column by name
            if hasattr(schoenfeld_residuals, 'columns') and worst_var in schoenfeld_residuals.columns:
                resid_vals = schoenfeld_residuals[worst_var].values
            else:
                var_cols = [c for c in train_data.columns if c not in ('duration', 'event_churned')]
                var_idx = var_cols.index(worst_var) if worst_var in var_cols else 0
                resid_vals = schoenfeld_residuals[:, var_idx] if hasattr(schoenfeld_residuals, 'shape') else schoenfeld_residuals.iloc[:, var_idx].values

            for lang, fig_dir, _ in LANGS:
                set_font(lang)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(range(len(resid_vals)), resid_vals,
                           alpha=0.3, s=10, color='steelblue')
                ax.axhline(y=0, color='red', linestyle='--')
                ax.set_xlabel('Rank of duration' if lang == 'en' else 'Duration排名', fontsize=11)
                ax.set_ylabel(f'Schoenfeld Residual: {worst_var}', fontsize=11)
                ax.set_title(f'Schoenfeld Residuals: {worst_var}', fontsize=13)
                ax.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, 'figure_schoenfeld.png'), dpi=300, bbox_inches='tight')
                plt.close()
            print(f'  Schoenfeld residual plots saved (worst: {worst_var})')
        except Exception as e:
            print(f'  Schoenfeld plot failed: {e}')

    # Predict
    test_data = X_test.copy()
    surv_funcs = cph.predict_survival_function(test_data)
    pred_days = [np.trapz(surv_funcs[idx].values.flatten(), surv_funcs[idx].index.values) for idx in test_data.index]
    actual_days = df_clean.loc[X_test.index, 'duration'].values
    c_index = concordance_index(actual_days, pred_days, df_clean.loc[X_test.index, 'event_churned'].values)
    mae = mean_absolute_error(actual_days, pred_days)
    print(f'  C-index: {c_index:.4f}, MAE: {mae:.2f} days')

    # ── Random Survival Forest Comparison ──
    print('\n  --- RSF Comparison ---')
    rsf_c_index = None
    try:
        from sksurv.ensemble import RandomSurvivalForest

        y_train_struct = np.array(
            [(bool(e), d) for e, d in zip(train_data['event_churned'], train_data['duration'])],
            dtype=[('event', bool), ('time', float)]
        )
        rsf = RandomSurvivalForest(n_estimators=100, min_samples_leaf=5, random_state=RANDOM_SEED)
        rsf.fit(X_train.values, y_train_struct)

        rsf_risk = rsf.predict(X_test.values)
        rsf_c_index = concordance_index(
            df_clean.loc[X_test.index, 'duration'].values,
            -rsf_risk,
            df_clean.loc[X_test.index, 'event_churned'].values
        )
        print(f'  RSF C-index: {rsf_c_index:.4f} (Cox: {c_index:.4f})')
    except ImportError as e:
        print(f'  scikit-survival not available; skipping RSF. ({e})')
    except Exception as e:
        print(f'  RSF failed: {e}')

    # Bilingual Cox coefficients
    coef_df = cph.summary[['coef', 'exp(coef)']].sort_values('coef')
    for lang, fig_dir, _ in LANGS:
        set_font(lang)
        if lang == 'cn':
            title = 'Cox模型：各特征对流失风险的影响'
            xlabel = '回归系数（负值=保护，正值=风险）'
        else:
            title = 'Cox Model: Feature Impact on Churn Risk'
            xlabel = 'Coefficient (Negative = Protective, Positive = Risk)'

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['red' if c < 0 else 'green' for c in coef_df['coef']]
        ax.barh(range(len(coef_df)), coef_df['coef'].values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(coef_df)))
        ax.set_yticklabels(coef_df.index, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(title, fontsize=14)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.grid(True, linestyle='--', alpha=0.4, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'figure4_cox_coefficients.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Bilingual Pred vs Actual
    for lang, fig_dir, _ in LANGS:
        set_font(lang)
        if lang == 'cn':
            title = f'预测生存天数 vs 实际生存天数 (MAE={mae:.1f})'
            xl, yl, perf = '实际生存天数', '预测生存天数', '完美预测'
        else:
            title = f'Predicted vs Actual Survival Days (MAE={mae:.1f})'
            xl, yl, perf = 'Actual Survival Days', 'Predicted Survival Days', 'Perfect Prediction'

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(actual_days, pred_days, alpha=0.3, s=15, c='steelblue', edgecolors='none')
        ax.plot([0, max(actual_days)], [0, max(actual_days)], 'r--', linewidth=1, label=perf)
        ax.set_xlabel(xl, fontsize=12)
        ax.set_ylabel(yl, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'figure5_pred_vs_actual.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return cph, {'C_index': c_index, 'MAE': mae, 'RSF_C_index': rsf_c_index}


def segmented_retention(df):
    """(4) Segmented retention curves with bilingual output."""
    print('\n' + '=' * 50)
    print('1.4 Segmented Retention Analysis')
    print('=' * 50)

    kmf = KaplanMeierFitter()

    for lang, fig_dir, _ in LANGS:
        set_font(lang)
        if lang == 'cn':
            t1, t2 = '留存率：付费 vs 非付费玩家', '留存率：加联盟 vs 未加联盟'
            xl, yl = '注册后天数', '留存概率'
            pay_label, nonpay_label = '付费玩家', '非付费玩家'
            league_label, noleague_label = '加联盟', '未加联盟'
        else:
            t1, t2 = 'Retention: Paying vs Non-Paying Players', 'Retention: League vs Non-League Players'
            xl, yl = 'Days Since Registration', 'Survival Probability'
            pay_label, nonpay_label = 'Paying', 'Non-Paying'
            league_label, noleague_label = 'In League', 'No League'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for mask, label, color in [
            (df['is_paying']==1, pay_label, 'darkorange'),
            (df['is_paying']==0, nonpay_label, 'steelblue')
        ]:
            sub = df[mask]
            kmf.fit(sub['duration'], sub['event_churned'], label=label)
            kmf.plot_survival_function(ax=ax1, color=color)
        ax1.set_xlabel(xl, fontsize=12)
        ax1.set_ylabel(yl, fontsize=12)
        ax1.set_title(t1, fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.4)

        for mask, label, color in [
            (df['is_in_league']==1, league_label, 'darkgreen'),
            (df['is_in_league']==0, noleague_label, 'steelblue')
        ]:
            sub = df[mask]
            kmf.fit(sub['duration'], sub['event_churned'], label=label)
            kmf.plot_survival_function(ax=ax2, color=color)
        ax2.set_xlabel(xl, fontsize=12)
        ax2.set_ylabel(yl, fontsize=12)
        ax2.set_title(t2, fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.4)

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'figure6_segmented_retention.png'), dpi=300, bbox_inches='tight')
        plt.close()

    for name, mask in [('Paying', df['is_paying']==1), ('Non-Paying', df['is_paying']==0),
                        ('In League', df['is_in_league']==1), ('No League', df['is_in_league']==0)]:
        sub = df[mask]
        kmf.fit(sub['duration'], sub['event_churned'])
        ret_7 = kmf.survival_function_at_times(7).values[0]
        ret_30 = kmf.survival_function_at_times(30).values[0] if 30 <= sub['duration'].max() else 0
        print(f'  {name}: 7-day ret={ret_7*100:.1f}%, 30-day ret={ret_30*100:.1f}%, n={len(sub)}')


def main():
    print('=' * 60)
    print('Problem 1: Retention & Churn Prediction')
    print('=' * 60)
    df = load_data()
    print(f'Dataset: {len(df)} players, churn rate: {df["event_churned"].mean()*100:.1f}%')

    kmf = km_survival_curve(df)
    top_drops = churn_key_moments(df)
    cph, metrics = cox_prediction_model(df)
    segmented_retention(df)

    results = [
        '=== 问题1 结果 ===',
        f'次日留存率: {kmf.survival_function_at_times(1).values[0]*100:.2f}%',
        f'7日留存率: {kmf.survival_function_at_times(7).values[0]*100:.2f}%',
        f'14日留存率: {kmf.survival_function_at_times(14).values[0]*100:.2f}%',
        f'30日留存率: {kmf.survival_function_at_times(30).values[0]*100:.2f}%',
        f'中位生存时间: {kmf.median_survival_time_:.1f} 天',
        f'活跃跨度分布: {top_drops}',
        f'Cox: C-index={metrics["C_index"]:.4f}, MAE={metrics["MAE"]:.2f}天',
        f'RSF: C-index={metrics["RSF_C_index"]:.4f}' if metrics.get("RSF_C_index") is not None else 'RSF: skipped (scikit-survival not available)',
    ]
    with open(os.path.join(RES_DIR, '问题1_results.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    print('\nResults saved.')


if __name__ == '__main__':
    main()
