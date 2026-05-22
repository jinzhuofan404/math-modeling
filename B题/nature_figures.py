"""
Nature-style figure generation for B Problem.
Follows Nature figure-making skill specifications:
  - Font: Arial 7pt, svg.fonttype=none, pdf.fonttype=42
  - Palette: blue #0F4D92, green #8BCF8B, red #B64342, neutrals
  - No top/right spines, linewidth 0.8, no legend frame
  - Export: SVG + PDF + TIFF (dpi=600), bbox_inches='tight'
"""
import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter

# ── Nature Journal RC ──────────────────────────────────────────
mpl.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 6.5,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "lines.linewidth": 1.2,
})

def set_cn_font():
    """Use SimHei for Chinese glyph coverage."""
    mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
    mpl.rcParams['font.family'] = 'sans-serif'

def set_en_font():
    """Use Arial/DejaVu for Latin glyph coverage."""
    mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    mpl.rcParams['font.family'] = 'sans-serif'

# ── Palette ─────────────────────────────────────────────────────
PAL = {
    "blue": "#0F4D92", "blue_sec": "#3775BA",
    "green": "#8BCF8B", "green_dark": "#2E7D32",
    "red": "#B64342", "red_light": "#F6CFCB",
    "orange": "#E28E2C", "teal": "#42949E",
    "violet": "#9A4D8E",
    "neutral_light": "#CFCECE", "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D", "neutral_black": "#272727",
}
DEFAULT_COLORS = [PAL["blue"], PAL["green"], PAL["red"], PAL["teal"], PAL["violet"], PAL["orange"]]

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, 'data')
FIG_NATURE = os.path.join(BASE, 'figures', 'nature')
FIG_CN = os.path.join(FIG_NATURE, '中文版')
FIG_EN = os.path.join(FIG_NATURE, '英文版')
for d in [FIG_CN, FIG_EN]:
    os.makedirs(d, exist_ok=True)

RANDOM_SEED = 42
LANGS = [('cn', FIG_CN), ('en', FIG_EN)]


def save_pub(fig, filename, fig_dir):
    """Export in Nature formats: SVG + PDF + TIFF @ 600dpi + PNG @ 300dpi."""
    stem = os.path.join(fig_dir, filename)
    for ext, kwargs in [
        ('.svg', {'bbox_inches': 'tight', 'pad_inches': 0.05}),
        ('.pdf', {'bbox_inches': 'tight', 'pad_inches': 0.05}),
        ('.tiff', {'dpi': 600, 'bbox_inches': 'tight', 'pad_inches': 0.05}),
        ('.png', {'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0.05}),
    ]:
        try:
            fig.savefig(f"{stem}{ext}", **kwargs)
        except PermissionError:
            print(f'    [skip] {os.path.basename(stem)}{ext} locked')


def load_data():
    """Load both tables: df_full (P2/P3, full-cycle) and df_d3 (P1, Day3-corrected)."""
    path_full = os.path.join(DATA_DIR, 'player_features.csv')
    if not os.path.exists(path_full):
        path_full = os.path.join(DATA_DIR, 'player_features_slim.csv')
    df_full = pd.read_csv(path_full)

    path_d3 = os.path.join(DATA_DIR, 'player_features_day3.csv')
    if os.path.exists(path_d3):
        df_d3 = pd.read_csv(path_d3)
    else:
        df_d3 = df_full.copy()
        df_d3['duration'] = df_d3['lifecycle_days'].clip(upper=90)
    return df_full, df_d3


# ═══════════════════════════════════════════════════════════════
# Figure 1: KM Survival Curve (hero panel)
# ═══════════════════════════════════════════════════════════════
def fig1_km_curve(df):
    """Core claim: Player retention drops sharply — median survival only 1 day."""
    kmf = KaplanMeierFitter()
    T, E = df['lifecycle_days'].values, df['event_churned'].values
    kmf.fit(T, E)

    for lang, fig_dir in LANGS:
        set_cn_font() if lang == 'cn' else set_en_font()
        if lang == 'cn':
            title = 'Kaplan-Meier 玩家留存曲线'
            xl, yl = '注册后天数', '留存概率'
            days_anno = {1: '第1天', 7: '第7天', 14: '第14天', 30: '第30天'}
        else:
            title = 'Kaplan-Meier Player Retention Curve'
            xl, yl = 'Days Since Registration', 'Survival Probability'
            days_anno = {1: 'Day 1', 7: 'Day 7', 14: 'Day 14', 30: 'Day 30'}

        fig, ax = plt.subplots(figsize=(5, 3.8))
        kmf.plot_survival_function(ax=ax, ci_show=True, color=PAL["blue"], linewidth=1.5, label='')
        ax.get_legend().remove() if ax.get_legend() else None
        for d, label in days_anno.items():
            surv = kmf.survival_function_at_times(d).values[0]
            ax.axvline(x=d, color=PAL["neutral_mid"], linestyle=':', linewidth=0.6)
            ax.annotate(f'{label}\n{surv*100:.1f}%', xy=(d, surv), xytext=(d+2, surv+0.05),
                       fontsize=5.5, color=PAL["neutral_dark"],
                       arrowprops=dict(arrowstyle='->', color=PAL["neutral_mid"], lw=0.5))
        ax.set_title(title, fontsize=7.5, fontweight='bold')
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_ylim(0, 1.05); ax.set_xlim(0, 65)
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)
        save_pub(fig, 'fig1_km_curve', fig_dir)
        plt.close()
    return kmf


# ═══════════════════════════════════════════════════════════════
# Figure 2: Hazard Rate
# ═══════════════════════════════════════════════════════════════
def fig2_hazard_rate(df):
    """Core claim: Churn peaks at Day 2 — 54% of remaining players leave."""
    T, E = df['lifecycle_days'].values, df['event_churned'].values
    naf = NelsonAalenFitter(); naf.fit(T, E)
    haz = naf.cumulative_hazard_.diff().fillna(0)
    haz_s = haz.rolling(window=3, center=True).mean().values.flatten()
    days_arr = haz.index.values
    # Filter by actual day value, not array index (early days have many events/packed indices)
    day_mask = days_arr <= 30
    days_plot = days_arr[day_mask]
    haz_plot = haz_s[day_mask]
    top5 = np.argsort(haz_plot[1:])[-5:] + 1

    for lang, fig_dir in LANGS:
        set_cn_font() if lang == 'cn' else set_en_font()
        if lang == 'cn':
            title = '玩家流失风险率随时间变化'
            xl, yl = '注册后天数', '风险率'
        else:
            title = 'Player Churn Hazard Rate Over Time'
            xl, yl = 'Days Since Registration', 'Hazard Rate'
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.fill_between(days_plot[1:], haz_plot[1:], alpha=0.15, color=PAL["red"])
        ax.plot(days_plot[1:], haz_plot[1:], color=PAL["red"], linewidth=1.2)
        for p in top5:
            ax.axvline(x=days_plot[p], color=PAL["red"], linestyle=':', linewidth=0.5, alpha=0.6)
        ax.set_title(title, fontsize=7.5, fontweight='bold')
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_xlim(0, 30)
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)
        save_pub(fig, 'fig2_hazard_rate', fig_dir)
        plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 3: Segmented Retention (combined — hero panel for Q1 key finding)
# ═══════════════════════════════════════════════════════════════
def fig3_segmented_retention(df):
    """Core claim: Paying & league players have 3-5x higher retention."""
    kmf = KaplanMeierFitter()

    segments = [
        ('is_paying', 1, 'Paying' , PAL["orange"]),
        ('is_paying', 0, 'Non-Paying', PAL["blue"]),
        ('is_in_league', 1, 'In League', PAL["green_dark"]),
        ('is_in_league', 0, 'No League', PAL["neutral_mid"]),
    ]

    for lang, fig_dir in LANGS:
        set_cn_font() if lang == 'cn' else set_en_font()
        if lang == 'cn':
            names = {'Paying': '付费玩家', 'Non-Paying': '非付费玩家',
                     'In League': '加联盟', 'No League': '未加联盟'}
            tl, tr = '留存率：付费 vs 非付费玩家', '留存率：加联盟 vs 未加联盟'
            xl, yl = '注册后天数', '留存概率'
        else:
            names = {'Paying': 'Paying', 'Non-Paying': 'Non-Paying',
                     'In League': 'In League', 'No League': 'No League'}
            tl, tr = 'Retention: Paying vs Non-Paying', 'Retention: League vs No League'
            xl, yl = 'Days Since Registration', 'Survival Probability'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

        for col, val, sname, color in segments:
            mask = df[col] == val
            sub = df[mask]
            kmf.fit(sub['lifecycle_days'], sub['event_churned'], label=names[sname])
            ax = ax1 if 'Pay' in sname or '付费' in sname or 'Non' in sname else ax2
            kmf.plot_survival_function(ax=ax, color=color, linewidth=1.2, ci_show=False)

        ax1.set_title(tl, fontsize=7.5, fontweight='bold'); ax1.set_xlabel(xl); ax1.set_ylabel(yl)
        ax2.set_title(tr, fontsize=7.5, fontweight='bold'); ax2.set_xlabel(xl); ax2.set_ylabel(yl)
        for ax in [ax1, ax2]:
            ax.legend(fontsize=6); ax.set_xlim(0, 65); ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)
        save_pub(fig, 'fig3_segmented_retention', fig_dir)
        plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 4: Cox Coefficients
# ═══════════════════════════════════════════════════════════════
def fig4_cox_coefficients(df):
    """Core claim: Day3 features predict churn; event diversity & diamond protect."""
    # Use strict Day3 features (consistent with paper sec5)
    feature_cols = [
        'days_logged_d3', 'level_d3', 'level_change_d3', 'avg_duration_d3',
        'food_reduce_d3', 'wood_reduce_d3', 'stone_reduce_d3',
        'diamond_reduce_d3', 'coins_reduce_d3',
        'diamond_d3', 'gold_d3',
        'is_pay_d3', 'is_league_d3', 'n_event_types_d3',
    ]
    clean = df.dropna(subset=feature_cols+['duration','event_churned'])
    X = clean[feature_cols]; X = X[~X.isin([np.inf,-np.inf]).any(axis=1)]
    scl = StandardScaler(); Xs = pd.DataFrame(scl.fit_transform(X), columns=feature_cols, index=X.index)
    td = pd.concat([Xs, clean.loc[Xs.index, ['duration','event_churned']]], axis=1)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(td, duration_col='duration', event_col='event_churned', show_progress=False)
    coef_df = cph.summary[['coef','exp(coef)']].sort_values('coef')

    short_names_cn = {
        'days_logged_d3':'前3天登录天数', 'level_d3':'前3天等级',
        'level_change_d3':'前3天等级变化', 'avg_duration_d3':'前3天日均在线',
        'food_reduce_d3':'前3天粮食消耗', 'wood_reduce_d3':'前3天木材消耗',
        'stone_reduce_d3':'前3天矿石消耗', 'diamond_reduce_d3':'前3天钻石消耗',
        'coins_reduce_d3':'前3天金币消耗', 'diamond_d3':'前3天钻石存量',
        'gold_d3':'前3天金币存量', 'is_pay_d3':'前3天是否付费',
        'is_league_d3':'前3天是否入盟', 'n_event_types_d3':'前3天事件类型数',
    }
    short_names_en = {
        'days_logged_d3':'Days Logged(D3)', 'level_d3':'Level(D3)',
        'level_change_d3':'Level Change(D3)', 'avg_duration_d3':'Avg Duration(D3)',
        'food_reduce_d3':'Food Used(D3)', 'wood_reduce_d3':'Wood Used(D3)',
        'stone_reduce_d3':'Stone Used(D3)', 'diamond_reduce_d3':'Diamond Used(D3)',
        'coins_reduce_d3':'Coins Used(D3)', 'diamond_d3':'Diamond Stock(D3)',
        'gold_d3':'Gold Stock(D3)', 'is_pay_d3':'Paid(D3)',
        'is_league_d3':'In League(D3)', 'n_event_types_d3':'Event Types(D3)',
    }

    for lang, fig_dir in LANGS:
        set_cn_font() if lang == 'cn' else set_en_font()
        sn = short_names_cn if lang == 'cn' else short_names_en
        title = 'Cox模型：Day3特征对流失风险的影响' if lang == 'cn' else 'Cox: Day3 Feature Impact on Churn Risk'
        xlabel = '回归系数' if lang == 'cn' else 'Coefficient'
        display_idx = [sn.get(x, x) for x in coef_df.index]

        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        colors = [PAL["red"] if c > 0 else PAL["blue"] for c in coef_df['coef']]
        ax.barh(range(len(coef_df)), coef_df['coef'].values, color=colors, alpha=0.85, height=0.7)
        ax.set_yticks(range(len(coef_df)))
        ax.set_yticklabels(display_idx, fontsize=6)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_title(title, fontsize=7.5, fontweight='bold')
        ax.axvline(x=0, color=PAL["neutral_black"], linewidth=0.6)
        ax.grid(True, linestyle='--', alpha=0.25, linewidth=0.3, axis='x')
        save_pub(fig, 'fig4_cox_coef', fig_dir)
        plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 5: Resource-Level Elasticity
# ═══════════════════════════════════════════════════════════════
def fig5_resource_level(df):
    """Core claim: Stone consumption correlates most with level growth (r=0.50)."""
    df = df.copy()
    for c in ['daily_food','daily_wood','daily_stone']:
        df[c] = df[f'{c.split("_")[1]}_reduce'] / df['lifecycle_days'].clip(lower=1)
    ls = df.groupby('level_end').agg(n=('account_id','count'),
        gf=('level_growth','mean'), df=('daily_food','mean'),
        dw=('daily_wood','mean'), ds=('daily_stone','mean')).reset_index()
    ls = ls[(ls['level_end']>0)&(ls['n']>=3)]

    for lang, fig_dir in LANGS:
        set_cn_font() if lang == 'cn' else set_en_font()
        if lang == 'cn':
            t1, t2 = '日均资源消耗 vs 等级', '等级增长速度 vs 等级'
            x1, y1 = '玩家等级', '日均消耗量'
            x2, y2 = '玩家等级', '平均等级增长'
            rl = {'Food':'粮食','Wood':'木材','Stone':'矿石'}
        else:
            t1, t2 = 'Resource Use vs Level', 'Level Growth Rate vs Level'
            x1, y1 = 'Player Level', 'Mean Daily Use'
            x2, y2 = 'Player Level', 'Mean Growth'
            rl = {}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))
        lvs = ls['level_end'].values
        res_list = [('df', PAL["blue"], 'Food'), ('dw', PAL["green"], 'Wood'), ('ds', PAL["orange"], 'Stone')]
        for col, color, name in res_list:
            ax1.plot(lvs, ls[col], 'o-', color=color, markersize=3, linewidth=1,
                     label=rl.get(name, name))
        ax1.set_title(t1, fontsize=7.5, fontweight='bold')
        ax1.set_xlabel(x1); ax1.set_ylabel(y1); ax1.legend(fontsize=5.5)
        ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)

        ax2.plot(lvs, ls['gf'], 'D-', color=PAL["violet"], markersize=3, linewidth=1.2)
        ax2.set_title(t2, fontsize=7.5, fontweight='bold')
        ax2.set_xlabel(x2); ax2.set_ylabel(y2)
        ax2.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)
        save_pub(fig, 'fig5_resource_level', fig_dir)
        plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 6: Diamond-Churn Risk
# ═══════════════════════════════════════════════════════════════
def fig6_diamond_churn(df):
    """Core claim: Diamond stock < 566 raises 7-day churn probability above 50%."""
    df = df.copy()
    df['churned'] = (df['lifecycle_days']<7).astype(int)
    df['ld'] = np.log1p(df['diamond_median'])
    dd = df[['ld','diamond_median','churned']].dropna()
    lr = LogisticRegression(); lr.fit(dd[['ld']], dd['churned'])
    dr = np.linspace(dd['ld'].min(), dd['ld'].max(), 100)
    probs = lr.predict_proba(dr.reshape(-1,1))[:,1]
    thresh = np.expm1(dr[np.argmin(np.abs(probs-0.5))])

    for lang, fig_dir in LANGS:
        set_cn_font() if lang == 'cn' else set_en_font()
        if lang == 'cn':
            t1, t2 = '钻石分布：留 vs 失', f'流失概率 vs 钻石 (阈值≈{thresh:.0f})'
            x1, y1 = 'log(1+钻石)', '密度'
            x2, y2 = '钻石中位数', 'P(7天流失)'
            lb_sv, lb_ch = '留存(>=7d)', '流失(<7d)'
        else:
            t1, t2 = 'Diamond: Survive vs Churn', f'Churn Risk vs Diamond (Thresh≈{thresh:.0f})'
            x1, y1 = 'log(1+Diamond)', 'Density'
            x2, y2 = 'Median Diamond', 'P(Churn in 7d)'
            lb_sv, lb_ch = 'Survived >=7d', 'Churned <7d'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))
        for mask, label, color in [(df['churned']==0, lb_sv, PAL["green"]),
                                     (df['churned']==1, lb_ch, PAL["red"])]:
            sub = df[mask]
            ax1.hist(np.log1p(sub['diamond_median'].clip(0)), bins=35, alpha=0.5,
                    color=color, label=label, density=True, edgecolor='white', linewidth=0.3)
        ax1.set_title(t1, fontsize=7.5, fontweight='bold'); ax1.set_xlabel(x1); ax1.set_ylabel(y1)
        ax1.legend(fontsize=5.5); ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)

        ax2.plot(np.expm1(dr), probs, color=PAL["blue"], linewidth=1.2)
        ax2.axhline(y=0.5, color=PAL["neutral_mid"], linestyle=':', linewidth=0.6)
        ax2.axvline(x=thresh, color=PAL["red"], linestyle=':', linewidth=0.6)
        ax2.set_title(t2, fontsize=7.5, fontweight='bold'); ax2.set_xlabel(x2); ax2.set_ylabel(y2)
        ax2.set_xlim(0, min(max(np.expm1(dr)), 3000))  # zoom to meaningful range
        ax2.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)
        save_pub(fig, 'fig6_diamond_churn', fig_dir)
        plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 7: Clustering
# ═══════════════════════════════════════════════════════════════
def fig7_clustering(df):
    """Core claim: 6 player clusters identified — 74% are zero-spend churners."""
    feats = ['days_active','lifecycle_days','level_end','level_growth_rate',
             'is_paying','total_pay','is_in_league','vip_level_max',
             'diamond_median','total_get','total_reduce']
    dc = df[feats].copy().replace([np.inf,-np.inf],np.nan)
    for c in ['total_pay','diamond_median','total_get','total_reduce']:
        dc[f'log_{c}'] = np.log1p(dc[c].clip(0)); del dc[c]
    dc = dc.fillna(0)
    scl = StandardScaler(); Xs = scl.fit_transform(dc)
    Kr = range(2,8)
    inert, sil = [], []
    for k in Kr:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        lb = km.fit_predict(Xs); inert.append(km.inertia_); sil.append(silhouette_score(Xs,lb))
    bk = list(Kr)[np.argmax(sil)]

    for lang, fig_dir in LANGS:
        set_cn_font() if lang == 'cn' else set_en_font()
        if lang == 'cn':
            t1, t2 = '肘部法则', f'轮廓系数 (最佳K={bk})'
            x1, y1 = '聚类数K', '惯性(Inertia)'
            x2, y2 = '聚类数K', '轮廓系数'
        else:
            t1, t2 = 'Elbow Method', f'Silhouette Analysis (Best K={bk})'
            x1, y1 = 'Number of Clusters K', 'Inertia'
            x2, y2 = 'Number of Clusters K', 'Silhouette Score'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.0))
        ax1.plot(list(Kr), inert, 'o-', color=PAL["blue"], markersize=4, linewidth=1)
        ax1.set_title(t1, fontsize=7.5, fontweight='bold'); ax1.set_xlabel(x1); ax1.set_ylabel(y1)
        ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)

        ax2.plot(list(Kr), sil, 'o-', color=PAL["green_dark"], markersize=4, linewidth=1)
        ax2.axvline(x=bk, color=PAL["red"], linestyle=':', linewidth=0.6)
        ax2.set_title(t2, fontsize=7.5, fontweight='bold'); ax2.set_xlabel(x2); ax2.set_ylabel(y2)
        ax2.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)
        save_pub(fig, 'fig7_clustering', fig_dir)
        plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 8: First Pay Analysis
# ═══════════════════════════════════════════════════════════════
def fig8_first_pay(df):
    """Core claim: Only 3.8% pay; higher first-pay groups survive longer."""
    payers = df[df['is_paying']==1].copy()
    if len(payers) == 0:
        return
    payers['pg'] = pd.cut(payers['total_pay'], bins=[0,6,30,100,500], labels=['<$6','$6-30','$30-100','$100+'])
    groups = payers.groupby('pg')['lifecycle_days'].agg(['mean','std','count']).reset_index()

    for lang, fig_dir in LANGS:
        set_cn_font() if lang == 'cn' else set_en_font()
        if lang == 'cn':
            t1, t2 = '付费金额分布', '付费水平与生命周期关系'
            x1, y1 = '总付费(美元)', '玩家数'
            x2, y2 = '付费分组', '平均生命周期(天)'
            group_labels = ['<6元', '6-30元', '30-100元', '100+元']
        else:
            t1, t2 = 'Payment Distribution', 'Payment Level vs Lifecycle'
            x1, y1 = 'Total Pay (USD)', 'Number of Players'
            x2, y2 = 'Payment Group', 'Mean Lifecycle (days)'
            group_labels = ['<$6', '$6-30', '$30-100', '$100+']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.0))
        ax1.hist(payers['total_pay'].clip(upper=100), bins=25, color=PAL["blue"], alpha=0.85,
                edgecolor='white', linewidth=0.3)
        ax1.set_title(t1, fontsize=7.5, fontweight='bold'); ax1.set_xlabel(x1); ax1.set_ylabel(y1)
        ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.3, axis='y')

        ax2.bar(range(len(groups)), groups['mean'], yerr=groups['std'],
                color=[PAL["blue"], PAL["teal"], PAL["orange"], PAL["violet"]],
                capsize=3, alpha=0.9, edgecolor='white', linewidth=0.3, width=0.6)
        ax2.set_xticks(range(len(groups)))
        ax2.set_xticklabels(group_labels, fontsize=6)
        ax2.set_title(t2, fontsize=7.5, fontweight='bold'); ax2.set_xlabel(x2); ax2.set_ylabel(y2)
        ax2.grid(True, linestyle='--', alpha=0.3, linewidth=0.3, axis='y')
        save_pub(fig, 'fig8_first_pay', fig_dir)
        plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 9: XGBoost Feature Importance
# ═══════════════════════════════════════════════════════════════
def fig9_xgb_importance(df):
    """Core claim: Level_end and diamond_median are strongest pay predictors (SHAP-based)."""
    import shap
    feat_cols = ['days_active','lifecycle_days','level_end','level_growth','level_growth_rate',
                 'current_level_max','is_in_league','vip_level_max','n_event_types',
                 'food_get','food_reduce','wood_get','wood_reduce','stone_get','stone_reduce',
                 'coins_get','coins_reduce','diamond_get','diamond_reduce','diamond_median',
                 'total_get','total_reduce','duration_times']
    df = df.copy()
    df['ri'] = df['total_reduce']/df['lifecycle_days'].clip(1)
    df['df'] = df['diamond_get']-df['diamond_reduce']
    extra = ['ri','df']
    all_f = feat_cols + extra
    dm = df[all_f+['total_pay']].copy().replace([np.inf,-np.inf],np.nan).fillna(0)
    X, y = dm[all_f], np.log1p(dm['total_pay'])
    model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=RANDOM_SEED, verbosity=0)
    model.fit(X, y)

    # SHAP-based importance (not built-in gain importance)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    shap_mean = np.abs(shap_vals).mean(axis=0)  # mean |SHAP| for each feature
    ti = np.argsort(shap_mean)[-12:][::-1]

    fn_cn = {'days_active':'活跃天数','lifecycle_days':'生命周期','level_end':'最终等级',
             'level_growth':'等级增长','level_growth_rate':'等级增速','current_level_max':'最高等级',
             'is_in_league':'加入联盟','vip_level_max':'VIP等级','n_event_types':'事件类型数',
             'food_get':'粮食获取','food_reduce':'粮食消耗','wood_get':'木材获取','wood_reduce':'木材消耗',
             'stone_get':'矿石获取','stone_reduce':'矿石消耗','coins_get':'金币获取','coins_reduce':'金币消耗',
             'diamond_get':'钻石获取','diamond_reduce':'钻石消耗','diamond_median':'钻石中位数',
             'total_get':'总获取量','total_reduce':'总消耗量','duration_times':'在线时长','ri':'资源强度','df':'钻石净流量'}
    fn_en = {'ri':'Resource Intensity','df':'Diamond Net Flow'}

    for lang, fig_dir in LANGS:
        set_cn_font() if lang == 'cn' else set_en_font()
        title = 'XGBoost+SHAP：总付费关键驱动因子' if lang=='cn' else 'XGBoost+SHAP: Key Drivers of Total Pay'
        xlabel = '平均|SHAP值|' if lang=='cn' else 'Mean |SHAP| Value'

        fig, ax = plt.subplots(figsize=(5, 3.5))
        names = []
        for f in [all_f[i] for i in ti][::-1]:
            if lang == 'cn':
                names.append(fn_cn.get(f, f))
            else:
                names.append(fn_en.get(f, f))
        vals = shap_mean[ti][::-1]
        ax.barh(range(len(names)), vals, color=PAL["blue"], alpha=0.85, height=0.65)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_title(title, fontsize=7.5, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.25, linewidth=0.3, axis='x')
        save_pub(fig, 'fig9_xgb_importance', fig_dir)
        plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 10: Monte Carlo Results
# ═══════════════════════════════════════════════════════════════
def fig10_monte_carlo(df):
    """Problem 3: Three-scheme MC comparison (baseline / conservative / exploration)."""
    # Final 5000-MC results (hardcoded from _final_run.py)
    schemes = [
        {'name_cn': '基线\n(无干预)',   'name_en': 'Baseline\n(No Push)',  'rev': 5605,  'ret': 7.1},
        {'name_cn': '保守\n(主方案)',   'name_en': 'Conservative\n(Main)', 'rev': 26270, 'ret': 7.2},
        {'name_cn': '探索\n(混合v5)',   'name_en': 'Exploration\n(Hybrid)', 'rev': 25681, 'ret': 10.0},
    ]
    colors = [PAL["neutral_dark"], PAL["blue"], PAL["green_dark"]]
    target_rev, target_ret = 70000, 10.0

    for lang, fig_dir in LANGS:
        set_cn_font() if lang == 'cn' else set_en_font()
        if lang == 'cn':
            t1 = '三方案期望营收对比';            t2 = '三方案期望留存率对比'
            y1 = '期望营收 (元)';                  y2 = '30日留存率 (%)'
            names = [s['name_cn'] for s in schemes]
            tg_label = '目标: 70,000元'
            tg_ret_label = '目标: 10%'
        else:
            t1 = 'Expected Revenue by Scheme';     t2 = 'Expected Retention by Scheme'
            y1 = 'Expected Revenue (CNY)';         y2 = '30-Day Retention (%)'
            names = [s['name_en'] for s in schemes]
            tg_label = 'Target: 70K CNY'
            tg_ret_label = 'Target: 10%'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.0))
        rev_vals = [s['rev'] for s in schemes]
        ret_vals = [s['ret'] for s in schemes]
        xs = range(len(schemes))

        # Revenue bars
        bars1 = ax1.bar(xs, rev_vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5, width=0.5)
        ax1.axhline(y=target_rev, color=PAL["red"], linestyle='--', linewidth=0.8, label=tg_label)
        ax1.set_xticks(xs); ax1.set_xticklabels(names, fontsize=7)
        ax1.set_title(t1, fontsize=7.5, fontweight='bold'); ax1.set_ylabel(y1, fontsize=7)
        for i, (v, b) in enumerate(zip(rev_vals, bars1)):
            ax1.text(i, v + 600, f'{v:,}', ha='center', fontsize=7, fontweight='bold', color=colors[i])
        ax1.legend(fontsize=6)
        ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.3, axis='y')

        # Retention bars
        bars2 = ax2.bar(xs, ret_vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5, width=0.5)
        ax2.axhline(y=target_ret, color=PAL["red"], linestyle='--', linewidth=0.8, label=tg_ret_label)
        ax2.set_xticks(xs); ax2.set_xticklabels(names, fontsize=7)
        ax2.set_title(t2, fontsize=7.5, fontweight='bold'); ax2.set_ylabel(y2, fontsize=7)
        for i, (v, b) in enumerate(zip(ret_vals, bars2)):
            ax2.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=7, fontweight='bold', color=colors[i])
        ax2.legend(fontsize=6)
        ax2.grid(True, linestyle='--', alpha=0.3, linewidth=0.3, axis='y')

        plt.tight_layout()
        save_pub(fig, 'fig10_monte_carlo', fig_dir)
        plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 11: Active Days Distribution
# ═══════════════════════════════════════════════════════════════
def fig11_active_days(df):
    """Core claim: Most players quit within 3 days — distribution is heavily right-skewed."""
    active_dist = df['days_active'].value_counts().sort_index()
    days = np.arange(1, 31)
    day_counts = [len(df[df['days_active'] >= d]) for d in days]

    for lang, fig_dir in LANGS:
        set_cn_font() if lang == 'cn' else set_en_font()
        if lang == 'cn':
            t1, t2 = '玩家活跃天数分布', '存活天数累积曲线'
            x1, y1 = '活跃天数', '玩家数'
            x2, y2 = '天数阈值', '存活>=X天的玩家数'
        else:
            t1, t2 = 'Active Days Distribution', 'Survival Count by Day Threshold'
            x1, y1 = 'Active Days', 'Players'
            x2, y2 = 'Day Threshold', 'Players Surviving >= Day X'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.0))
        ax1.bar(active_dist.index[:30], active_dist.values[:30], color=PAL["blue"], alpha=0.85,
                edgecolor='white', linewidth=0.2)
        ax1.set_title(t1, fontsize=7.5, fontweight='bold'); ax1.set_xlabel(x1); ax1.set_ylabel(y1)
        ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.3, axis='y')

        ax2.plot(range(1, 31), day_counts, 'o-', color=PAL["red"], markersize=3, linewidth=1.2)
        ax2.set_title(t2, fontsize=7.5, fontweight='bold'); ax2.set_xlabel(x2); ax2.set_ylabel(y2)
        ax2.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)
        save_pub(fig, 'fig11_active_days', fig_dir)
        plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 12: Cox Predicted vs Actual
# ═══════════════════════════════════════════════════════════════
def fig12_cox_pred(df):
    """Core claim: Cox model achieves C-index=0.95, MAE=5.0 days."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from lifelines.utils import concordance_index

    # Use strict Day3 features (consistent with paper sec5)
    feature_cols = [
        'days_logged_d3', 'level_d3', 'level_change_d3', 'avg_duration_d3',
        'food_reduce_d3', 'wood_reduce_d3', 'stone_reduce_d3',
        'diamond_reduce_d3', 'coins_reduce_d3',
        'diamond_d3', 'gold_d3',
        'is_pay_d3', 'is_league_d3', 'n_event_types_d3',
    ]
    clean = df.dropna(subset=feature_cols+['duration','event_churned'])
    X = clean[feature_cols]; X = X[~X.isin([np.inf,-np.inf]).any(axis=1)]
    scl = StandardScaler(); Xs = pd.DataFrame(scl.fit_transform(X), columns=feature_cols, index=X.index)
    td = pd.concat([Xs, clean.loc[Xs.index, ['duration','event_churned']]], axis=1)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(td, duration_col='duration', event_col='event_churned', show_progress=False)

    X_train, X_test = train_test_split(Xs, test_size=0.3, random_state=RANDOM_SEED)
    sf = cph.predict_survival_function(X_test)
    pred = [np.trapz(sf[idx].values.flatten(), sf[idx].index.values) for idx in X_test.index]
    actual = clean.loc[X_test.index, 'duration'].values
    ci = concordance_index(actual, pred, clean.loc[X_test.index, 'event_churned'].values)
    mae = mean_absolute_error(actual, pred)

    for lang, fig_dir in LANGS:
        set_cn_font() if lang == 'cn' else set_en_font()

        events = clean.loc[X_test.index, 'event_churned'].values
        # Split by event status
        mask_event = events == 1
        actual_ev, pred_ev = actual[mask_event], np.array(pred)[mask_event]
        actual_cens, pred_cens = actual[~mask_event], np.array(pred)[~mask_event]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))

        # ── Left: Scatter with alpha (better visibility for skewed data) ──
        ax1.scatter(actual, np.array(pred), alpha=0.15, s=8,
                    c=PAL["blue"], edgecolors='none')
        mx_log = max(actual.max(), np.max(pred))
        ax1.plot([0, mx_log], [0, mx_log], '--', color=PAL["neutral_dark"],
                linewidth=0.6, alpha=0.6)
        if lang == 'cn':
            ax1.set_xlabel('实际生存天数')
            ax1.set_ylabel('预测生存天数')
            ax1.set_title('全量预测（散点图）', fontsize=7, fontweight='bold')
        else:
            ax1.set_xlabel('Actual Survival Days')
            ax1.set_ylabel('Predicted Survival Days')
            ax1.set_title('All Predictions (scatter)', fontsize=7, fontweight='bold')

        # ── Right: Scatter by event status ──
        ax2.scatter(actual_cens, pred_cens, alpha=0.5, s=18,
                    c=PAL["blue"], edgecolors='white', linewidths=0.2,
                    label='Censored' if lang == 'en' else '未流失')
        ax2.scatter(actual_ev, pred_ev, alpha=0.5, s=18,
                    c=PAL["orange"], edgecolors='white', linewidths=0.2,
                    label='Churned' if lang == 'en' else '已流失')
        mx_s = max(actual.max(), np.max(pred)) * 1.05
        ax2.plot([0, mx_s], [0, mx_s], '--', color=PAL["neutral_dark"],
                linewidth=0.6)
        ax2.legend(fontsize=5.5, loc='lower right')

        if lang == 'cn':
            ax2.set_xlabel('实际生存天数')
            ax2.set_ylabel('预测生存天数')
            ax2.set_title(f'按流失状态分层 (C={ci:.3f})', fontsize=7, fontweight='bold')
        else:
            ax2.set_xlabel('Actual Survival Days')
            ax2.set_ylabel('Predicted Survival Days')
            ax2.set_title(f'By Churn Status (C-index={ci:.3f})', fontsize=7, fontweight='bold')

        # Add metrics text
        if lang == 'cn':
            txt = f'MAE: {mae:.1f}天\n流失(n={mask_event.sum()})\n未流失(n={(~mask_event).sum()})'
        else:
            txt = f'MAE: {mae:.1f}\nChurned(n={mask_event.sum()})\nCensored(n={(~mask_event).sum()})'
        ax2.text(0.95, 0.08, txt, transform=ax2.transAxes, fontsize=5.3,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=PAL["neutral_light"], alpha=0.8))

        for ax in [ax1, ax2]:
            ax.grid(True, linestyle='--', alpha=0.25, linewidth=0.3)

        plt.tight_layout()
        save_pub(fig, 'fig12_cox_pred', fig_dir)
        plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 13: XGBoost Predicted vs Actual
# ═══════════════════════════════════════════════════════════════
def fig13_xgb_pred(df):
    """Core claim: XGBoost separates payers from non-payers; accuracy is limited by sparse payers."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score

    feat_cols = ['days_active','lifecycle_days','level_end','level_growth','level_growth_rate',
                 'current_level_max','is_in_league','vip_level_max','n_event_types',
                 'food_get','food_reduce','wood_get','wood_reduce','stone_get','stone_reduce',
                 'coins_get','coins_reduce','diamond_get','diamond_reduce','diamond_median',
                 'total_get','total_reduce','duration_times']
    df['ri'] = df['total_reduce']/df['lifecycle_days'].clip(1)
    df['df'] = df['diamond_get']-df['diamond_reduce']
    all_f = feat_cols + ['ri','df']
    dm = df[all_f+['total_pay']].copy().replace([np.inf,-np.inf],np.nan).fillna(0)
    X, y_raw = dm[all_f], dm['total_pay']
    X_train, X_test, yt_raw, ye_raw = train_test_split(X, y_raw, test_size=0.3, random_state=RANDOM_SEED)
    _, _, yt_log, ye_log = train_test_split(X, np.log1p(y_raw), test_size=0.3, random_state=RANDOM_SEED)

    model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=RANDOM_SEED, verbosity=0)
    model.fit(X_train, yt_log)
    yp = np.expm1(model.predict(X_test))
    mae_all = mean_absolute_error(ye_raw, yp)

    # Payer-only stats
    mask_payer = ye_raw > 0
    n_payers = mask_payer.sum()
    mae_payers = mean_absolute_error(ye_raw[mask_payer], yp[mask_payer]) if n_payers > 0 else 0
    r2_payers = r2_score(ye_raw[mask_payer], yp[mask_payer]) if n_payers > 1 else 0

    # Log-transform for visualization
    ye_log_vis = np.log1p(ye_raw)
    yp_log_vis = np.log1p(np.clip(yp, 0, None))

    for lang, fig_dir in LANGS:
        set_cn_font() if lang == 'cn' else set_en_font()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))

        # ── Left panel: hexbin of all predictions (log scale) ──
        hb = ax1.hexbin(ye_log_vis, yp_log_vis, gridsize=50, cmap='YlOrRd',
                        mincnt=1, linewidths=0, alpha=0.9)
        cbar = fig.colorbar(hb, ax=ax1, shrink=0.78, pad=0.02)
        ax1.plot([0, ye_log_vis.max()], [0, ye_log_vis.max()], '--',
                color=PAL["neutral_dark"], linewidth=0.6, alpha=0.6)
        ax1.set_xlabel('Actual log(1+Pay)' if lang == 'en' else '实际 log(1+付费)',
                       fontsize=6.5)
        ax1.set_ylabel('Predicted log(1+Pay)' if lang == 'en' else '预测 log(1+付费)',
                       fontsize=6.5)
        if lang == 'cn':
            ax1.set_title('全量预测（对数尺度六边形密度图）', fontsize=7, fontweight='bold')
            cbar.set_label('样本数', fontsize=5.5)
        else:
            ax1.set_title('All Predictions (log-scale hexbin)', fontsize=7, fontweight='bold')
            cbar.set_label('Count', fontsize=5.5)

        # ── Right panel: scatter of payers only (linear scale) ──
        ax2.scatter(ye_raw[mask_payer], yp[mask_payer], alpha=0.6, s=25,
                    c=PAL["orange"], edgecolors='white', linewidths=0.3,
                    zorder=3)
        mx_p = max(ye_raw[mask_payer].max(), yp[mask_payer].max()) * 1.1 if n_payers > 0 else 10
        ax2.plot([0, mx_p], [0, mx_p], '--', color=PAL["neutral_dark"], linewidth=0.6)
        # Add perfect prediction line label
        if n_payers > 0:
            ax2.annotate('y=x', xy=(mx_p*0.85, mx_p*0.83), fontsize=5.5,
                        color=PAL["neutral_dark"], rotation=35)

        if lang == 'cn':
            ax2.set_xlabel('实际总付费 (美元)', fontsize=6.5)
            ax2.set_ylabel('预测总付费 (美元)', fontsize=6.5)
            ax2.set_title(f'付费玩家预测 (n={n_payers})', fontsize=7, fontweight='bold')
            # Metrics box
            txt = f'MAE: {mae_payers:.2f}\nR2: {r2_payers:.3f}'
        else:
            ax2.set_xlabel('Actual Total Pay (USD)', fontsize=6.5)
            ax2.set_ylabel('Predicted Total Pay (USD)', fontsize=6.5)
            ax2.set_title(f'Payers Only (n={n_payers})', fontsize=7, fontweight='bold')
            txt = f'MAE: {mae_payers:.2f}\nR2: {r2_payers:.3f}'

        if n_payers > 0:
            ax2.text(0.95, 0.08, txt, transform=ax2.transAxes, fontsize=5.5,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor=PAL["neutral_light"], alpha=0.8))

        for ax in [ax1, ax2]:
            ax.grid(True, linestyle='--', alpha=0.25, linewidth=0.3)

        plt.tight_layout()
        save_pub(fig, 'fig13_xgb_pred', fig_dir)
        plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 14: 3D Cluster Visualization
# ═══════════════════════════════════════════════════════════════
def fig14_cluster_3d(df):
    """Core claim: 6 player clusters separate along lifecycle, pay, and level axes."""
    feats = ['days_active','lifecycle_days','level_end','level_growth_rate',
             'is_paying','total_pay','is_in_league','vip_level_max',
             'diamond_median','total_get','total_reduce']
    dc = df[feats].copy().replace([np.inf,-np.inf],np.nan)
    for c in ['total_pay','diamond_median','total_get','total_reduce']:
        dc[f'log_{c}'] = np.log1p(dc[c].clip(0)); del dc[c]
    dc = dc.fillna(0)
    scl = StandardScaler(); Xs = scl.fit_transform(dc)
    km = KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(Xs)

    lc_idx = dc.columns.get_loc('lifecycle_days')
    lp_idx = dc.columns.get_loc('log_total_pay')
    le_idx = dc.columns.get_loc('level_end')

    # Custom 6-cluster discrete palette (Nature-friendly, distinguishable)
    cluster_colors = [PAL["blue"], PAL["teal"], PAL["green_dark"],
                      PAL["orange"], PAL["violet"], PAL["red"]]

    for lang, fig_dir in LANGS:
        set_cn_font() if lang == 'cn' else set_en_font()
        if lang == 'cn':
            title, xl, yl, zl = '3D 玩家聚类可视化', '生命周期', '对数总付费', '最高等级'
            cl = '聚类'
        else:
            title, xl, yl, zl = '3D Player Cluster Visualization', 'Lifecycle', 'Log Total Pay', 'Max Level'
            cl = 'Cluster'

        fig = plt.figure(figsize=(7, 5.5))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each cluster separately for discrete colors
        for c in range(6):
            mask = labels == c
            ax.scatter(Xs[mask, lc_idx], Xs[mask, lp_idx], Xs[mask, le_idx],
                       c=cluster_colors[c], s=18, alpha=0.65, linewidths=0,
                       label=f'{cl} {c+1}', rasterized=True)

        # Optimal viewing angle
        ax.view_init(elev=22, azim=-42)

        # Axis labels
        ax.set_xlabel(f'{xl}\n(std)', fontsize=6.5, labelpad=8)
        ax.set_ylabel(f'{yl}\n(std)', fontsize=6.5, labelpad=8)
        ax.set_zlabel(f'{zl}\n(std)', fontsize=6.5, labelpad=8)
        ax.set_title(title, fontsize=7.5, fontweight='bold', pad=14)

        # Tick formatting
        ax.tick_params(labelsize=5.5, pad=3, which='major',
                       direction='out', length=2, width=0.5)

        # Sparse grid
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(PAL["neutral_light"])
        ax.yaxis.pane.set_edgecolor(PAL["neutral_light"])
        ax.zaxis.pane.set_edgecolor(PAL["neutral_light"])
        ax.grid(True, linestyle=':', alpha=0.15, linewidth=0.3)

        # Legend with cluster labels
        leg = ax.legend(fontsize=5.5, loc='upper left', ncol=2,
                        markerscale=1.2, handletextpad=0.5,
                        columnspacing=0.8, borderpad=0.4)
        leg.get_frame().set_linewidth(0.4)
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_alpha(0.8)

        plt.tight_layout()
        save_pub(fig, 'fig14_cluster_3d', fig_dir)
        plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 15: Monte Carlo Convergence
# ═══════════════════════════════════════════════════════════════
def fig15_mc_convergence(df):
    """Core claim: Revenue estimate stabilizes after ~500 simulations."""
    np.random.seed(RANDOM_SEED)
    dc = df[['days_active','lifecycle_days','level_end','level_growth_rate',
             'is_paying','total_pay','is_in_league','vip_level_max','diamond_median',
             'total_get','total_reduce']].copy().replace([np.inf,-np.inf],np.nan).fillna(0)
    for c in ['total_pay','diamond_median','total_get','total_reduce']:
        dc[f'log_{c}']=np.log1p(dc[c].clip(0)); del dc[c]
    Xs = StandardScaler().fit_transform(dc)
    labels = KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init=10).fit_predict(Xs)
    df['cl'] = labels

    profiles = []
    for c in range(6):
        s = df[df['cl']==c]
        profiles.append({'pct':len(s)/len(df), 'pay_rate':s['is_paying'].mean(),
                         'ml':s['lifecycle_days'].mean(), 'mp':max(0.1,s['total_pay'].mean()),
                         'r30':(s['lifecycle_days']>=30).mean()})

    N = 10000
    revs = []
    gifts = [{'p':6,'m':1.0,'rb':0.02},{'p':30,'m':0.4,'rb':0.03},
             {'p':68,'m':0.15,'rb':0.04},{'p':128,'m':0.05,'rb':0.05},{'p':328,'m':0.01,'rb':0.06}]
    for _ in range(1000):
        trv = 0
        for cp in profiles:
            n = max(1, int(cp['pct']*N))
            for __ in range(n):
                bp = np.random.random()<cp['pay_rate']
                op = np.random.exponential(cp['mp']) if bp else 0
                sr = 0
                for g in gifts:
                    if np.random.random()<max(0.0005, cp['pay_rate']*g['m']):
                        sr+=g['p']
                trv += op+sr
        revs.append(trv)
    revs = np.array(revs)
    cum = np.cumsum(revs) / np.arange(1, len(revs)+1)

    for lang, fig_dir in LANGS:
        set_cn_font() if lang == 'cn' else set_en_font()
        if lang == 'cn':
            title = '蒙特卡洛收敛曲线'
            xl, yl = '模拟次数', '累计平均营收(元)'
            fl = f'最终: {cum[-1]:.0f}元'
        else:
            title = 'Monte Carlo Convergence'
            xl, yl = 'Simulation Count', 'Cumulative Mean Revenue (CNY)'
            fl = f'Final: {cum[-1]:.0f}'

        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(cum, color=PAL["blue"], linewidth=0.8, alpha=0.9)
        ax.axhline(y=cum[-1], color=PAL["neutral_mid"], linestyle='--', linewidth=0.6, label=fl)
        ax.set_xlabel(xl, fontsize=7); ax.set_ylabel(yl, fontsize=7)
        ax.set_title(title, fontsize=7.5, fontweight='bold')
        ax.legend(fontsize=6)
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.3)
        save_pub(fig, 'fig15_mc_convergence', fig_dir)
        plt.close()


def main():
    print('Nature-Style Figure Generation')
    print('=' * 60)
    df_full, df_d3 = load_data()
    print(f'Data: full={len(df_full)} players, day3={len(df_d3)} players')

    # Add compatibility column aliases to Day3 table for P1 figure functions
    df_d3['is_paying'] = df_d3['is_pay_d3']
    df_d3['is_in_league'] = df_d3['is_league_d3']
    df_d3['days_active'] = df_d3['days_logged_d3']
    df_d3['lifecycle_days'] = df_d3['duration']
    df_d3['level_end'] = df_d3['level_d3']
    df_d3['level_growth'] = df_d3['level_change_d3']
    df_d3['level_growth_rate'] = df_d3['level_change_d3'] / df_d3['duration'].clip(lower=1)

    # Problem 1 figures: use Day3-corrected table (df_d3) for correct retention/churn
    print('\n1/15  KM curve...');             fig1_km_curve(df_d3)
    print('2/15  Hazard rate...');            fig2_hazard_rate(df_d3)
    print('3/15  Segmented retention...');    fig3_segmented_retention(df_d3)
    print('4/15  Cox coefficients...');       fig4_cox_coefficients(df_d3)
    print('11/15 Active days...');            fig11_active_days(df_d3)
    print('12/15 Cox pred vs actual...');     fig12_cox_pred(df_d3)

    # Problem 2/3 figures: use full-cycle table (df_full)
    print('5/15  Resource-level...');         fig5_resource_level(df_full)
    print('6/15  Diamond-churn...');          fig6_diamond_churn(df_full)
    print('7/15  Clustering...');             fig7_clustering(df_full)
    print('8/15  First pay...');              fig8_first_pay(df_full)
    print('9/15  XGBoost importance...');     fig9_xgb_importance(df_full)
    print('13/15 XGBoost pred vs actual...'); fig13_xgb_pred(df_full)
    print('14/15 Cluster 3D...');             fig14_cluster_3d(df_full)

    # Problem 3 MC figures: use hardcoded final results
    print('10/15 Monte Carlo...');            fig10_monte_carlo(df_full)
    print('15/15 MC convergence...');         fig15_mc_convergence(df_full)

    print('\nDone! Generated 15 figures x 2 languages x 4 formats (SVG/PDF/TIFF/PNG)')
    print(f'  Output: {FIG_CN} + {FIG_EN}')


if __name__ == '__main__':
    main()
