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
        # Will be filled after all clusters are profiled (sorting-based naming)
        profile['name_cn'] = f'Cluster{c}'
        profile['name_en'] = f'Cluster{c}'
        cluster_profiles.append(profile)

    # ── Sorting-based naming ──
    # Sort by mean_pay descending; highest paying = 高氪核心党
    sorted_by_pay = sorted(cluster_profiles, key=lambda x: x['mean_pay'], reverse=True)
    for rank, cp in enumerate(sorted_by_pay):
        if cp['mean_pay'] > 0 and rank == 0:
            cp['name_cn'] = '高氪核心党'; cp['name_en'] = 'Whale'
        elif cp['mean_pay'] > 0:
            cp['name_cn'] = '中氪战力党'; cp['name_en'] = 'Mid Spender'
        elif cp['pay_rate'] > 0:
            cp['name_cn'] = '微氪月卡党'; cp['name_en'] = 'Light Spender'
        elif cp['mean_lifecycle'] >= 10:
            cp['name_cn'] = '零氪休闲党'; cp['name_en'] = 'F2P Casual'
        else:
            cp['name_cn'] = '零氪流失党'; cp['name_en'] = 'F2P Churner'

    # Print and save
    for cp in cluster_profiles:
        print(f'    C{cp["cluster"]} ({cp["name_cn"]}): n={cp["n"]}, pay={cp["pay_rate"]:.1f}%, life={cp["mean_lifecycle"]:.1f}d')

    # Save name map
    import csv
    map_path = os.path.join(os.path.dirname(__file__), 'data', 'cluster_name_map.csv')
    with open(map_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['cluster_id', 'name_cn', 'name_en', 'mean_pay', 'pay_rate', 'mean_lifecycle', 'mean_level', 'league_rate'])
        for cp in cluster_profiles:
            w.writerow([cp['cluster'], cp['name_cn'], cp['name_en'],
                   f"{cp['mean_pay']:.2f}", f"{cp['pay_rate']:.1f}",
                   f"{cp['mean_lifecycle']:.1f}", f"{cp['mean_level']:.1f}", f"{cp['league_rate']:.1f}"])
    print(f'  Name map saved to {map_path}')

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


def estimate_demand_curves(cluster_profiles, df):
    """Estimate demand curves: shared beta, stratified alpha_c."""
    print('\n' + '=' * 50)
    print('3.3 Demand Curve Estimation')
    print('=' * 50)

    # From historical data: fit f(p) = alpha * exp(-beta * p)
    # Using observed purchase amounts as proxy for "willingness to pay"
    payers = df[df['total_pay'] > 0]
    pay_amounts = payers['total_pay'].values

    # Fit beta from all payers (shared price sensitivity)
    # Using the exponential distribution MLE: beta = 1 / mean(pay)
    beta_hat = 1.0 / max(0.01, pay_amounts.mean())
    print(f'  Shared beta (price sensitivity): {beta_hat:.6f}')

    # Stratified alpha_c per cluster
    prices = np.array([6, 12, 18, 30, 68, 128, 328])
    demand = {}
    for c, cp in enumerate(cluster_profiles):
        alpha_c = max(0.001, cp['pay_rate'] / 100.0)
        demand[c] = {
            'alpha': alpha_c,
            'beta': beta_hat,
            'curve': lambda p, a=alpha_c, b=beta_hat: a * np.exp(-b * p),
        }
        print(f'  Cluster {c} ({cp["name_cn"]}): alpha={alpha_c:.4f}')

    return demand, beta_hat


def monte_carlo_conservative(cluster_profiles, df):
    """Conservative scheme: based on historical pay rates, no intervention boost."""
    print('\n' + '=' * 50)
    print('3.4 Conservative Scheme MC')
    print('=' * 50)

    np.random.seed(RANDOM_SEED)

    cluster_sizes = []
    for cp in cluster_profiles:
        n = max(1, int(cp['pct']/100*N_SIM_PLAYERS))
        cluster_sizes.append(n)
    cluster_sizes[-1] += N_SIM_PLAYERS - sum(cluster_sizes)

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

    gift_packs = [
        {'price': 6, 'prob_mult': 1.0, 'ret_boost': 0.02},
        {'price': 30, 'prob_mult': 0.4, 'ret_boost': 0.03},
        {'price': 68, 'prob_mult': 0.15, 'ret_boost': 0.04},
        {'price': 128, 'prob_mult': 0.05, 'ret_boost': 0.05},
        {'price': 328, 'prob_mult': 0.01, 'ret_boost': 0.06},
    ]

    sim_revenues, sim_retentions = [], []
    for sim_idx in range(N_MC_SIMS):
        if sim_idx % 1000 == 0 and sim_idx > 0:
            print(f'  Progress: {sim_idx}/{N_MC_SIMS}')
        total_revenue, retained = 0, 0
        for c, (cp, n_players) in enumerate(zip(cluster_profiles, cluster_sizes)):
            cd = cluster_dists[c]
            base_pay_prob = cd['pay_rate']
            for _ in range(n_players):
                lifecycle = max(1, np.random.normal(cd['lifecycle_mean'], cd['lifecycle_std']))
                ret_base = np.random.random() < cd['retention_base']
                organic_pay = np.random.exponential(cd['mean_pay']) if np.random.random() < base_pay_prob else 0
                strategy_rev, strategy_ret_boost = 0, 0
                for gp in gift_packs:
                    if np.random.random() < max(0.0005, base_pay_prob * gp['prob_mult']):
                        strategy_rev += gp['price']
                        strategy_ret_boost += gp['ret_boost']
                total_revenue += organic_pay + strategy_rev
                if np.random.random() < min(0.99, cd['retention_base'] + strategy_ret_boost) or ret_base:
                    retained += 1
        sim_revenues.append(total_revenue)
        sim_retentions.append(retained / N_SIM_PLAYERS)

    revenues = np.array(sim_revenues)
    retentions = np.array(sim_retentions)
    print(f'  Mean revenue: {revenues.mean():.0f}, P(>=70K): {(revenues>=70000).mean()*100:.1f}%')
    print(f'  Mean retention: {retentions.mean()*100:.1f}%, P(>=10%): {(retentions>=0.10).mean()*100:.1f}%')
    return {'mean_revenue': revenues.mean(), 'std_revenue': revenues.std(),
            'prob_revenue': (revenues >= 70000).mean(),
            'mean_retention': retentions.mean(), 'prob_retention': (retentions >= 0.10).mean(),
            'revenues': revenues, 'retentions': retentions, 'label': 'Conservative'}


def monte_carlo_target(cluster_profiles, df, demand, beta_hat):
    """Target scheme: intervention elasticity + greedy sequential push."""
    print('\n' + '=' * 50)
    print('3.5 Target Scheme MC (Intervention Elasticity)')
    print('=' * 50)

    np.random.seed(RANDOM_SEED)

    # ── Intervention elasticity multipliers per cluster ──
    # λ_c: how much more likely cluster c is to purchase when actively pushed
    # vs. their organic pay rate. Based on: (1) intra-cluster pay willingness,
    # (2) SLG industry first-purchase conversion benchmarks (10-20%).
    lambda_c = {
        0: 0.15,  # 零氪休闲党: active but never paid → 15% first-purchase conversion
        1: 0.05,  # 零氪流失党: fast churn, reachable only Day1 → 5% baseline
        2: 0.05,  # 零氪流失党: same
        3: 1.20,  # 微氪月卡党: already paying → 20% uplift on organic
        4: 1.50,  # 高氪核心党: proven willingness → 50% uplift (scarcity/limited packs)
        5: 1.30,  # 中氪战力党: paying → 30% uplift
    }

    cluster_sizes = []
    for cp in cluster_profiles:
        n = max(1, int(cp['pct']/100*N_SIM_PLAYERS))
        cluster_sizes.append(n)
    cluster_sizes[-1] += N_SIM_PLAYERS - sum(cluster_sizes)

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

    # ── Gift packs ──
    gift_packs = [
        {'name_cn': '首充礼包', 'price': 6,  'timing': 1},
        {'name_cn': '新手补给', 'price': 12, 'timing': 3},
        {'name_cn': '资源补给包', 'price': 30, 'timing': 7},
        {'name_cn': '成长加速包', 'price': 68, 'timing': 10},
        {'name_cn': '战力突破包', 'price': 128,'timing': 14},
        {'name_cn': '至尊大礼包', 'price': 328,'timing': 21},
    ]

    # ── Compute P_push per cluster×pack with conversion floor ──
    # Zero-pay clusters get a minimum first-purchase conversion probability
    # (the "barrier-breaking" effect of a well-timed push)
    for c, cp in enumerate(cluster_profiles):
        lam = lambda_c.get(c, 0.05)
        er_list = []
        for gp in gift_packs:
            if cp['pay_rate'] > 0:
                # Paying clusters: organic rate + intervention uplift
                p_push = lam * cp['pay_rate']/100.0 * np.exp(-beta_hat * gp['price'])
                p_push = max(p_push, 0.01)
            else:
                # Zero-pay clusters: baseline first-purchase conversion × price decay
                # λ_c already encodes the push-conversion baseline (0.05-0.15)
                p_push = lam * np.exp(-beta_hat * gp['price'])
                p_push = max(p_push, 0.001)
            er = gp['price'] * p_push
            er_list.append((er, gp, p_push))
        er_list.sort(key=lambda x: x[0], reverse=True)
        top3 = er_list[:3]
        print(f'  {cp["name_cn"]}: top3 packs = ' +
              ', '.join([f'{gp["price"]}yuan(P={pp:.3f},ER={er:.1f})' for er, gp, pp in top3]))

    # ── MC Simulation ──
    sim_revenues, sim_retentions = [], []
    MAX_PUSHES = 8
    lam_pen, mu_pen = 0.003, 0.001

    for sim_idx in range(N_MC_SIMS):
        if sim_idx % 1000 == 0 and sim_idx > 0:
            print(f'  Progress: {sim_idx}/{N_MC_SIMS}')
        total_revenue, retained = 0, 0
        for c, (cp, n_players) in enumerate(zip(cluster_profiles, cluster_sizes)):
            cd = cluster_dists[c]
            lam = lambda_c.get(c, 0.05)
            # Build ranked pack list for this cluster
            ranked = []
            for gp in gift_packs:
                if cp['pay_rate'] > 0:
                    pp = lam * cp['pay_rate']/100.0 * np.exp(-beta_hat * gp['price'])
                    pp = max(pp, 0.01)
                else:
                    pp = lam * np.exp(-beta_hat * gp['price'])
                    pp = max(pp, 0.001)
                ranked.append((gp, pp))
            ranked.sort(key=lambda x: x[1] * x[0]['price'], reverse=True)

            for _ in range(n_players):
                lifecycle = max(1, np.random.normal(cd['lifecycle_mean'], cd['lifecycle_std']))
                ret_base = np.random.random() < cd['retention_base']
                organic_pay = np.random.exponential(cd['mean_pay']) if np.random.random() < cd['pay_rate'] else 0
                strategy_rev, n_pushes, total_price = 0, 0, 0
                for gp, pp in ranked:
                    if n_pushes >= MAX_PUSHES:
                        break
                    if gp['timing'] > lifecycle:
                        continue
                    if np.random.random() < pp:
                        strategy_rev += gp['price']
                    n_pushes += 1
                    total_price += gp['price']
                ret_penalty = lam_pen * n_pushes + mu_pen * total_price / 100
                ret_boost = min(0.04, 0.008 * n_pushes)
                total_revenue += organic_pay + strategy_rev
                if np.random.random() < min(0.99, cd['retention_base'] + ret_boost - ret_penalty) or ret_base:
                    retained += 1
        sim_revenues.append(total_revenue)
        sim_retentions.append(retained / N_SIM_PLAYERS)

    revs = np.array(sim_revenues); rets = np.array(sim_retentions)
    mean_rev, std_rev = revs.mean(), revs.std()
    prob_rev, mean_ret = (revs >= TARGET_REVENUE).mean(), rets.mean()
    prob_ret = (rets >= TARGET_RETENTION).mean()

    print(f'  Mean revenue: {mean_rev:.0f}, P(>=70K): {prob_rev*100:.1f}%')
    print(f'  Mean retention: {mean_ret*100:.1f}%, P(>=10%): {prob_ret*100:.1f}%')
    return {'mean_revenue': mean_rev, 'std_revenue': std_rev, 'prob_revenue': prob_rev,
            'mean_retention': mean_ret, 'prob_retention': prob_ret,
            'revenues': revs, 'retentions': rets, 'label': 'Target'}


def monte_carlo_baseline(cluster_profiles, df):
    """No-intervention baseline: organic payments only."""
    print('\n' + '=' * 50)
    print('3.3b Baseline (No Intervention)')
    print('=' * 50)
    np.random.seed(RANDOM_SEED)
    cl_sizes = []
    for cp in cluster_profiles:
        cl_sizes.append(max(1, int(cp['pct']/100*N_SIM_PLAYERS)))
    cl_sizes[-1] += N_SIM_PLAYERS - sum(cl_sizes)
    cl_dists = {}
    for c, cp in enumerate(cluster_profiles):
        sub = df[df['cluster'] == c]
        cl_dists[c] = {
            'lifecycle_mean': sub['lifecycle_days'].mean(),
            'lifecycle_std': max(0.5, sub['lifecycle_days'].std()),
            'pay_rate': cp['pay_rate']/100,
            'mean_pay': max(0.1, cp['mean_pay']),
            'retention_base': cp['retention_30d']/100,
        }
    sim_revenues, sim_retentions = [], []
    for _ in range(N_MC_SIMS):
        tr, ret = 0, 0
        for c, (cp, n) in enumerate(zip(cluster_profiles, cl_sizes)):
            cd = cl_dists[c]
            for _ in range(n):
                op = np.random.exponential(cd['mean_pay']) if np.random.random() < cd['pay_rate'] else 0
                tr += op
                if np.random.random() < cd['retention_base']:
                    ret += 1
        sim_revenues.append(tr)
        sim_retentions.append(ret / N_SIM_PLAYERS)
    revs = np.array(sim_revenues)
    rets = np.array(sim_retentions)
    print(f'  Mean revenue: {revs.mean():.0f}, Mean retention: {rets.mean()*100:.1f}%')
    return {'mean_revenue': revs.mean(), 'std_revenue': revs.std(),
            'prob_revenue': (revs >= 70000).mean(),
            'mean_retention': rets.mean(), 'prob_retention': (rets >= 0.10).mean(),
            'revenues': revs, 'retentions': rets, 'label': 'Baseline'}


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

    # Demand curves
    demand, beta_hat = estimate_demand_curves(cluster_profiles, df)

    # Three MC simulations
    mc_baseline = monte_carlo_baseline(cluster_profiles, df)
    mc_conservative = monte_carlo_conservative(cluster_profiles, df)
    mc_target = monte_carlo_target(cluster_profiles, df, demand, beta_hat)

    sched_df = generate_strategy_schedule(cluster_profiles)

    results = [
        '=== 问题3 结果 ===',
        f'最优聚类数: {len(cluster_profiles)}',
        '',
        '--- 无干预基线 ---',
        f'期望营收: {mc_baseline["mean_revenue"]:.0f} CNY',
        f'期望留存率: {mc_baseline["mean_retention"]*100:.1f}%',
        '',
        '--- 保守方案 ---',
        f'期望营收: {mc_conservative["mean_revenue"]:.0f} CNY',
        f'P(营收>=70000): {mc_conservative["prob_revenue"]*100:.1f}%',
        f'期望留存率: {mc_conservative["mean_retention"]*100:.1f}%',
        '',
        '--- 目标方案(贪心序列推送) ---',
        f'期望营收: {mc_target["mean_revenue"]:.0f} CNY',
        f'P(营收>=70000): {mc_target["prob_revenue"]*100:.1f}%',
        f'期望留存率: {mc_target["mean_retention"]*100:.1f}%',
    ]
    with open(os.path.join(RES_DIR, '问题3_results.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    print('\nResults saved.')


if __name__ == '__main__':
    main()
