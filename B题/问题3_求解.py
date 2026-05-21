"""
问题3：新服首月付费策略设计与收益最大化模型
方法：K-Means聚类 + 多目标优化 + 蒙特卡洛模拟验证
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats

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
N_SIM_PLAYERS = 10000
TARGET_REVENUE = 70000
TARGET_RETENTION = 0.10
N_MC_SIMS = 5000
LANGS = [('cn', FIG_CN), ('en', FIG_EN)]


def set_font(lang):
    if lang == 'cn':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei', 'Arial']
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Arial Unicode MS', 'SimHei', 'Microsoft YaHei']


def load_features():
    path = os.path.join(DATA_DIR, 'player_features.csv')
    return pd.read_csv(path)


def player_clustering(df):
    """(1) K-Means clustering with bilingual output."""
    print('\n' + '=' * 50)
    print('3.1 Player Clustering')
    print('=' * 50)

    cluster_feats = [
        'days_active', 'lifecycle_days', 'level_end', 'level_growth_rate',
        'is_paying', 'total_pay', 'is_in_league', 'vip_level_max',
        'diamond_median', 'total_get', 'total_reduce',
    ]
    df_cluster = df[cluster_feats].copy().replace([np.inf, -np.inf], np.nan)
    for col in ['total_pay', 'diamond_median', 'total_get', 'total_reduce']:
        df_cluster[f'log_{col}'] = np.log1p(df_cluster[col].clip(lower=0))
        df_cluster = df_cluster.drop(columns=[col])
    df_cluster = df_cluster.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    K_range = range(2, 8)
    inertias, silhouettes = [], []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    best_k = list(K_range)[np.argmax(silhouettes)]
    print(f'  Optimal K = {best_k} (silhouette = {max(silhouettes):.3f})')

    km = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    df['cluster'] = km.fit_predict(X_scaled)

    # Cluster profiles
    cluster_profiles = []
    for c in range(best_k):
        sub = df[df['cluster'] == c]
        profile = {
            'cluster': c, 'n': len(sub), 'pct': len(sub)/len(df)*100,
            'pay_rate': sub['is_paying'].mean()*100, 'mean_pay': sub['total_pay'].mean(),
            'mean_lifecycle': sub['lifecycle_days'].mean(), 'mean_level': sub['level_end'].mean(),
            'league_rate': sub['is_in_league'].mean()*100, 'mean_diamond': sub['diamond_median'].mean(),
            'retention_7d': (sub['lifecycle_days']>=7).mean()*100,
            'retention_30d': (sub['lifecycle_days']>=30).mean()*100,
        }
        if profile['pay_rate'] < 1 and profile['mean_lifecycle'] < 5:
            name_cn, name_en = '零氪休闲党', 'F2P Casual'
        elif profile['pay_rate'] < 20 and profile['mean_lifecycle'] < 15:
            name_cn, name_en = '微氪月卡党', 'Light Spender'
        elif profile['pay_rate'] >= 20 and profile['mean_pay'] < 50:
            name_cn, name_en = '中氪战力党', 'Mid Spender'
        else:
            name_cn, name_en = '高氪核心党', 'Whale'
        profile['name_cn'] = name_cn
        profile['name_en'] = name_en
        cluster_profiles.append(profile)
        print(f'    C{c} ({name_cn}): n={len(sub)}, pay={profile["pay_rate"]:.1f}%, life={profile["mean_lifecycle"]:.1f}d')

    # Bilingual elbow + silhouette
    for lang, fig_dir in LANGS:
        set_font(lang)
        if lang == 'cn':
            t1, t2 = '肘部法则', f'轮廓系数 (最佳K={best_k})'
            x1, y1 = 'K (聚类数)', '惯性'
            x2, y2 = 'K (聚类数)', '轮廓系数'
        else:
            t1, t2 = f'Silhouette Analysis (Best K={best_k})', 'Elbow Method'
            # Actually swap: elbow left, silhouette right
            t1, t2 = 'Elbow Method', f'Silhouette Analysis (Best K={best_k})'
            x1, y1 = 'K (Clusters)', 'Inertia'
            x2, y2 = 'K (Clusters)', 'Silhouette Score'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(list(K_range), inertias, 'bo-', markersize=6)
        ax1.set_xlabel(x1, fontsize=12); ax1.set_ylabel(y1, fontsize=12)
        ax1.set_title(t1, fontsize=14); ax1.grid(True, linestyle='--', alpha=0.4)

        ax2.plot(list(K_range), silhouettes, 'go-', markersize=6)
        ax2.axvline(x=best_k, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel(x2, fontsize=12); ax2.set_ylabel(y2, fontsize=12)
        ax2.set_title(t2, fontsize=14); ax2.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'figure12_clustering.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Bilingual 3D scatter
    for lang, fig_dir in LANGS:
        set_font(lang)
        if lang == 'cn':
            title = '3D 玩家聚类可视化'
            xl, yl, zl = '生命周期', '对数总付费', '最高等级'
            cl = '聚类'
        else:
            title = '3D Player Cluster Visualization'
            xl, yl, zl = 'Lifecycle', 'Log Total Pay', 'Level End'
            cl = 'Cluster'

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        lc_idx = df_cluster.columns.get_loc('lifecycle_days')
        lp_idx = df_cluster.columns.get_loc('log_total_pay')
        le_idx = df_cluster.columns.get_loc('level_end')
        sc = ax.scatter(X_scaled[:, lc_idx], X_scaled[:, lp_idx], X_scaled[:, le_idx],
                        c=df['cluster'], cmap='viridis', s=30, alpha=0.7)
        ax.set_xlabel(f'{xl} (std)', fontsize=11)
        ax.set_ylabel(f'{yl} (std)', fontsize=11)
        ax.set_zlabel(f'{zl} (std)', fontsize=11)
        ax.set_title(title, fontsize=14, pad=20)
        cbar = plt.colorbar(sc, label=cl)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'figure13_cluster_3d.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return cluster_profiles, df


def monte_carlo_simulation(cluster_profiles, df):
    """Monte Carlo simulation with bilingual output."""
    print('\n' + '=' * 50)
    print('3.3 Monte Carlo Simulation')
    print('=' * 50)

    np.random.seed(RANDOM_SEED)

    cluster_sizes = []
    for cp in cluster_profiles:
        n = max(1, int(cp['pct']/100*N_SIM_PLAYERS))
        cluster_sizes.append(n)
    cluster_sizes[-1] += N_SIM_PLAYERS - sum(cluster_sizes)

    # Historical distributions per cluster
    cluster_dists = {}
    for c, cp in enumerate(cluster_profiles):
        sub = df[df['cluster'] == c]
        cluster_dists[c] = {
            'lifecycle_mean': sub['lifecycle_days'].mean(),
            'lifecycle_std': max(0.5, sub['lifecycle_days'].std()),
            'pay_rate': cp['pay_rate']/100,
            'mean_pay': max(0.1, cp['mean_pay']),
            'retention_base': cp['retention_30d']/100,
        }

    # Strategy parameters: gift packs per cluster
    gift_packs = [
        {'name_cn': '首充礼包', 'name_en': 'First Purchase Pack', 'price': 6, 'prob_mult': 1.0, 'ret_boost': 0.02},
        {'name_cn': '资源补给包', 'name_en': 'Resource Supply Pack', 'price': 30, 'prob_mult': 0.4, 'ret_boost': 0.03},
        {'name_cn': '成长加速包', 'name_en': 'Growth Accelerator', 'price': 68, 'prob_mult': 0.15, 'ret_boost': 0.04},
        {'name_cn': '战力突破包', 'name_en': 'Power Breakthrough', 'price': 128, 'prob_mult': 0.05, 'ret_boost': 0.05},
        {'name_cn': '至尊大礼包', 'name_en': 'Ultimate Pack', 'price': 328, 'prob_mult': 0.01, 'ret_boost': 0.06},
    ]

    # Run simulation
    sim_revenues, sim_retentions = [], []
    for sim_idx in range(N_MC_SIMS):
        if sim_idx % 1000 == 0 and sim_idx > 0:
            print(f'  Progress: {sim_idx}/{N_MC_SIMS}')

        total_revenue = 0
        retained = 0

        for c, (cp, n_players) in enumerate(zip(cluster_profiles, cluster_sizes)):
            cd = cluster_dists[c]
            base_pay_prob = cd['pay_rate']

            for _ in range(n_players):
                lifecycle = max(1, np.random.normal(cd['lifecycle_mean'], cd['lifecycle_std']))
                ret_base = np.random.random() < cd['retention_base']

                # Organic payment
                organic_pay = 0
                if np.random.random() < base_pay_prob:
                    organic_pay = np.random.exponential(cd['mean_pay'])

                # Strategy-driven payment
                strategy_rev = 0
                strategy_ret_boost = 0
                for gp in gift_packs:
                    prob = max(0.0005, base_pay_prob * gp['prob_mult'])
                    if np.random.random() < prob:
                        strategy_rev += gp['price']
                        strategy_ret_boost += gp['ret_boost']

                total_revenue += organic_pay + strategy_rev
                final_ret = min(0.99, cd['retention_base'] + strategy_ret_boost)
                if np.random.random() < final_ret or ret_base:
                    retained += 1

        sim_revenues.append(total_revenue)
        sim_retentions.append(retained / N_SIM_PLAYERS)

    revenues = np.array(sim_revenues)
    retentions = np.array(sim_retentions)

    mean_rev = revenues.mean()
    std_rev = revenues.std()
    prob_rev = (revenues >= TARGET_REVENUE).mean()
    mean_ret = retentions.mean()
    prob_ret = (retentions >= TARGET_RETENTION).mean()

    print(f'\n  Results:')
    print(f'    Mean Revenue: {mean_rev:.2f} CNY')
    print(f'    P(Revenue >= {TARGET_REVENUE}): {prob_rev*100:.2f}%')
    print(f'    Mean Retention: {mean_ret*100:.2f}%')
    print(f'    P(Retention >= {TARGET_RETENTION*100:.0f}%): {prob_ret*100:.2f}%')

    # Bilingual Monte Carlo results
    for lang, fig_dir in LANGS:
        set_font(lang)
        if lang == 'cn':
            t1 = f'营收分布 (P>={TARGET_REVENUE}={prob_rev*100:.1f}%)'
            t2 = f'留存率分布 (P>={TARGET_RETENTION*100:.0f}%={prob_ret*100:.1f}%)'
            x1, y1 = '总营收(元)', '密度'
            x2, y2 = '30日留存率(%)', '密度'
            t_label = f'目标: {TARGET_REVENUE}'
            m_label = f'均值: {mean_rev:.0f}'
            tr_label = f'目标: {TARGET_RETENTION*100:.0f}%'
            mr_label = f'均值: {mean_ret*100:.1f}%'
        else:
            t1 = f'Revenue Distribution (P>={TARGET_REVENUE}={prob_rev*100:.1f}%)'
            t2 = f'Retention Distribution (P>={TARGET_RETENTION*100:.0f}%={prob_ret*100:.1f}%)'
            x1, y1 = 'Total Revenue (CNY)', 'Density'
            x2, y2 = '30-Day Retention Rate (%)', 'Density'
            t_label = f'Target: {TARGET_REVENUE}'
            m_label = f'Mean: {mean_rev:.0f}'
            tr_label = f'Target: {TARGET_RETENTION*100:.0f}%'
            mr_label = f'Mean: {mean_ret*100:.1f}%'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.hist(revenues, bins=60, color='steelblue', edgecolor='white', alpha=0.8, density=True)
        ax1.axvline(x=TARGET_REVENUE, color='red', linestyle='--', linewidth=2, label=t_label)
        ax1.axvline(x=mean_rev, color='green', linestyle='-', linewidth=2, label=m_label)
        x_norm = np.linspace(revenues.min(), revenues.max(), 200)
        ax1.plot(x_norm, stats.norm.pdf(x_norm, mean_rev, std_rev), 'orange', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel(x1, fontsize=12); ax1.set_ylabel(y1, fontsize=12)
        ax1.set_title(t1, fontsize=13); ax1.legend(fontsize=9); ax1.grid(True, linestyle='--', alpha=0.4)

        ax2.hist(retentions*100, bins=60, color='darkorange', edgecolor='white', alpha=0.8, density=True)
        ax2.axvline(x=TARGET_RETENTION*100, color='red', linestyle='--', linewidth=2, label=tr_label)
        ax2.axvline(x=mean_ret*100, color='green', linestyle='-', linewidth=2, label=mr_label)
        ax2.set_xlabel(x2, fontsize=12); ax2.set_ylabel(y2, fontsize=12)
        ax2.set_title(t2, fontsize=13); ax2.legend(fontsize=9); ax2.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'figure14_monte_carlo.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Bilingual convergence
    for lang, fig_dir in LANGS:
        set_font(lang)
        if lang == 'cn':
            title = '蒙特卡洛模拟收敛曲线'
            xl, yl = '模拟迭代次数', '累计平均营收(元)'
            fl = f'最终均值: {mean_rev:.0f}'
        else:
            title = 'Monte Carlo Convergence'
            xl, yl = 'Simulation Iteration', 'Cumulative Mean Revenue (CNY)'
            fl = f'Final Mean: {mean_rev:.0f}'

        fig, ax = plt.subplots(figsize=(10, 5))
        cum_rev = np.cumsum(revenues) / np.arange(1, len(revenues)+1)
        ax.plot(cum_rev, 'b-', linewidth=1, alpha=0.7)
        ax.axhline(y=mean_rev, color='red', linestyle='--', label=fl)
        ax.set_xlabel(xl, fontsize=12); ax.set_ylabel(yl, fontsize=12)
        ax.set_title(title, fontsize=14); ax.legend(fontsize=10); ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'figure15_mc_convergence.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return {'mean_revenue': mean_rev, 'std_revenue': std_rev, 'prob_revenue': prob_rev,
            'mean_retention': mean_ret, 'prob_retention': prob_ret}, cluster_sizes, cluster_profiles


def generate_strategy_schedule(cluster_profiles):
    """Generate final strategy schedule table."""
    print('\n' + '=' * 50)
    print('3.4 Strategy Schedule')
    print('=' * 50)

    gift_packs = [
        {'name_cn': '首充礼包', 'name_en': 'First Purchase Pack', 'price': 6,
         'timing_cn': 'Day 1', 'timing_en': 'Day 1',
         'trigger_cn': '注册首日自动推送', 'trigger_en': 'Auto on registration day',
         'targets': ['F2P Casual', 'Light Spender']},
        {'name_cn': '资源补给包', 'name_en': 'Resource Supply Pack', 'price': 30,
         'timing_cn': 'Day 7', 'timing_en': 'Day 7',
         'trigger_cn': '活跃>=5天且未付费触发', 'trigger_en': 'Active>=5d & no payment',
         'targets': ['Light Spender', 'Mid Spender']},
        {'name_cn': '成长加速包', 'name_en': 'Growth Accelerator', 'price': 68,
         'timing_cn': 'Day 3-5', 'timing_en': 'Day 3-5',
         'trigger_cn': '连续2天资源缺口>30%', 'trigger_en': '2-day resource gap >30%',
         'targets': ['Mid Spender']},
        {'name_cn': '战力突破包', 'name_en': 'Power Breakthrough', 'price': 128,
         'timing_cn': 'Day 7-10', 'timing_en': 'Day 7-10',
         'trigger_cn': '等级停滞(>=3天未升级)', 'trigger_en': 'Level stagnation >=3 days',
         'targets': ['Mid Spender', 'Whale']},
        {'name_cn': '至尊大礼包', 'name_en': 'Ultimate Pack', 'price': 328,
         'timing_cn': 'Day 1-7', 'timing_en': 'Day 1-7',
         'trigger_cn': '注册首周且等级>=5触发', 'trigger_en': 'Day 1-7 & level >= 5',
         'targets': ['Whale']},
    ]

    # Build bilingual table
    rows = []
    for gp in gift_packs:
        rows.append({
            'Timing_CN': gp['timing_cn'], 'Timing_EN': gp['timing_en'],
            'Trigger_CN': gp['trigger_cn'], 'Trigger_EN': gp['trigger_en'],
            'Pack_CN': gp['name_cn'], 'Pack_EN': gp['name_en'],
            'Price': gp['price'],
            'Target_Clusters': ', '.join(gp['targets']),
        })

    sched_df = pd.DataFrame(rows)
    sched_df.to_csv(os.path.join(RES_DIR, '问题3_strategy_schedule.csv'), index=False, encoding='utf-8-sig')
    print(sched_df[['Timing_CN', 'Pack_CN', 'Price', 'Target_Clusters']].to_string(index=False))
    return sched_df


def main():
    print('=' * 60)
    print('Problem 3: Revenue Maximization Strategy')
    print('=' * 60)

    df = load_features()
    print(f'Loaded {len(df)} players')

    cluster_profiles, df = player_clustering(df)
    mc_results, cluster_sizes, cluster_profiles = monte_carlo_simulation(cluster_profiles, df)
    sched_df = generate_strategy_schedule(cluster_profiles)

    results = [
        '=== 问题3 结果 ===',
        f'最优聚类数: {len(cluster_profiles)}',
        f'期望总营收: {mc_results["mean_revenue"]:.2f} CNY',
        f'P(营收>={TARGET_REVENUE}): {mc_results["prob_revenue"]*100:.2f}%',
        f'期望留存率: {mc_results["mean_retention"]*100:.2f}%',
        f'P(留存率>={TARGET_RETENTION*100:.0f}%): {mc_results["prob_retention"]*100:.2f}%',
    ]
    with open(os.path.join(RES_DIR, '问题3_results.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    print('\nResults saved.')


if __name__ == '__main__':
    main()
