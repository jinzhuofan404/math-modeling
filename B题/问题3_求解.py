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
    """Load player features + activity intensity per active day."""
    path = os.path.join(DATA_DIR, 'player_features.csv')
    df = pd.read_csv(path)

    # Load pick_stat files for activity intensity (used in MC, not clustering)
    stat1_path = os.path.join(BASE_DIR, '..', '题目', 'B题：附件 数据集', 'pick_stat1.csv')
    stat2_path = os.path.join(BASE_DIR, '..', '题目', 'B题：附件 数据集', 'pick_stat2.csv')
    if os.path.exists(stat1_path):
        s1 = pd.read_csv(stat1_path)
        df['stat1_count'] = s1['count'].values
    if os.path.exists(stat2_path):
        s2 = pd.read_csv(stat2_path)
        df['stat2_count'] = s2['count'].values

    # Per-active-day intensity
    df['events_per_day'] = df['stat2_count'] / df['days_active'].clip(lower=1)
    df['resources_per_day'] = df['stat1_count'] / df['days_active'].clip(lower=1)
    return df


def player_clustering(df):
    """(1) K-Means clustering with bilingual output."""
    print('\n' + '=' * 50)
    print('3.1 Player Clustering')
    print('=' * 50)

    # Note: stat intensity features (events_per_day, resources_per_day) excluded
    # from clustering because they collapse K to 2, masking payment/league dimensions.
    # They are used in the MC simulation's per-cluster targeting logic instead.
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


def monte_carlo_conservative(cluster_profiles, df, cl_sizes, cl_dists):
    """Conservative scheme: based on historical pay rates, no intervention boost."""
    print('\n' + '=' * 50)
    print('3.4 Conservative Scheme MC')
    print('=' * 50)

    np.random.seed(RANDOM_SEED)

    gift_packs = [
        {'price': 6, 'prob_mult': 1.0, 'ret_boost': 0.02},
        {'price': 30, 'prob_mult': 0.4, 'ret_boost': 0.03},
        {'price': 68, 'prob_mult': 0.15, 'ret_boost': 0.04},
        {'price': 128, 'prob_mult': 0.05, 'ret_boost': 0.05},
        {'price': 328, 'prob_mult': 0.01, 'ret_boost': 0.06},
        {'price': 648, 'prob_mult': 0.003, 'ret_boost': 0.07},
    ]

    sim_revenues, sim_retentions = [], []
    for sim_idx in range(N_MC_SIMS):
        if sim_idx % 1000 == 0 and sim_idx > 0:
            print(f'  Progress: {sim_idx}/{N_MC_SIMS}')
        total_revenue, retained = 0, 0
        for c, (cp, n_players) in enumerate(zip(cluster_profiles, cl_sizes)):
            cd = cl_dists[c]
            base_pay_prob = cd['pay_rate']
            for _ in range(n_players):
                lifecycle = max(1, np.random.normal(cd['lifecycle_mean'], cd['lifecycle_std']))
                organic_pay = np.random.exponential(cd['mean_pay']) if np.random.random() < base_pay_prob else 0
                strategy_rev, strategy_ret_boost = 0, 0
                for gp in gift_packs:
                    if np.random.random() < max(0.0005, base_pay_prob * gp['prob_mult']):
                        strategy_rev += gp['price']
                        strategy_ret_boost += gp['ret_boost']
                total_revenue += organic_pay + strategy_rev
                if np.random.random() < min(0.99, cd['retention_base'] + strategy_ret_boost):
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


def monte_carlo_target(cluster_profiles, df, demand, beta_hat, cl_sizes, cl_dists,
                       quadrant_config=None, sens_label='基准'):
    """v4 Target scheme: four-quadrant uplift + free retention pack + state triggers.

    Quadrants: Persuadable (push), Sure Thing (organic only),
               Sleeping Dog (suppress), Lost Cause (free retention pack only).

    State triggers: lifecycle<3 skip >30¥, whale skip <30¥, post-big-buy skip <68¥.
    Joint decision: each purchase adds ret_boost incrementally.
    """
    print('\n' + '=' * 50)
    print(f'3.5 Target Scheme MC v4 [{sens_label}]')
    print('=' * 50)

    np.random.seed(RANDOM_SEED)

    # ── Quadrant proportions per cluster (default baseline) ──
    if quadrant_config is None:
        quadrant_config = {
            # name: [Persuadable%, SureThing%, SleepingDog%, LostCause%]
            "高氪核心党": [0.40, 0.60, 0.00, 0.00],
            "中氪战力党": [0.50, 0.40, 0.10, 0.00],
            "零氪休闲党": [0.60, 0.00, 0.05, 0.35],
            "零氪流失党": [0.20, 0.00, 0.05, 0.75],
        }

    # ── Intervention parameters by cluster NAME ──
    lambda_by_name = {  # zero-pay: first-purchase conversion baseline
        "零氪休闲党": 0.15,
        "零氪流失党": 0.05,
    }
    delta_by_name = {   # paying: uplift multiplier on conservative base
        "中氪战力党": 0.30,
        "高氪核心党": 0.50,
    }

    # Conservative base probabilities per pack
    con_base = {6: 1.0, 12: 0.5, 30: 0.4, 68: 0.15, 128: 0.05, 328: 0.01, 648: 0.003}

    # ── Gift packs ──
    # Free retention pack (price=0, for Lost Cause)
    free_pack = {'name_cn': '免费挽留包', 'price': 0, 'timing': 1, 'ret_boost': 0.05}
    # Paid packs (price ascending for adaptive logic)
    paid_packs = [
        {'name_cn': '首充礼包',   'price': 6,   'timing': 1,  'ret_boost': 0.02},
        {'name_cn': '新手补给',   'price': 12,  'timing': 3,  'ret_boost': 0.02},
        {'name_cn': '资源补给包', 'price': 30,  'timing': 7,  'ret_boost': 0.03},
        {'name_cn': '成长加速包', 'price': 68,  'timing': 10, 'ret_boost': 0.04},
        {'name_cn': '战力突破包', 'price': 128, 'timing': 14, 'ret_boost': 0.05},
        {'name_cn': '至尊大礼包', 'price': 328, 'timing': 21, 'ret_boost': 0.06},
        {'name_cn': '传说大礼包', 'price': 648, 'timing': 28, 'ret_boost': 0.07},
    ]

    # ── Precompute per-cluster, per-pack purchase probabilities (for Persuadable) ──
    cluster_probs = {}
    for c, cp in enumerate(cluster_profiles):
        name = cp['name_cn']
        lam = lambda_by_name.get(name, 0.05)
        delta = delta_by_name.get(name, 0.0)
        base_pay_prob = cp['pay_rate'] / 100.0

        probs = []
        for gp in paid_packs:
            price = gp['price']
            if base_pay_prob > 0:
                # Paying cluster: conservative base + additive uplift
                con_p = base_pay_prob * con_base.get(price, 0.01)
                p_push = min(0.99, con_p * (1.0 + delta))
            else:
                # Zero-pay cluster: demand curve creation
                p_push = lam * np.exp(-beta_hat * price)
                p_push = max(p_push, 0.001)
            probs.append((gp, p_push))
        probs.sort(key=lambda x: x[0]['price'])
        cluster_probs[c] = probs

        top3 = sorted(probs, key=lambda x: x[0]['price'] * x[1], reverse=True)[:3]
        print(f'  {name}: top3 = ' +
              ', '.join([f'{gp["price"]}yuan(P={pp:.3f},ER={gp["price"]*pp:.1f})' for gp, pp in top3]))

    # ── Quadrant name → index mapping ──
    QUAD_PERSUADABLE, QUAD_SURE, QUAD_SLEEPING, QUAD_LOST = 0, 1, 2, 3
    quad_names = ['Persuadable', 'SureThing', 'SleepingDog', 'LostCause']

    # Build per-cluster quadrant distribution
    cluster_quad = {}
    for c, cp in enumerate(cluster_profiles):
        qcfg = quadrant_config.get(cp['name_cn'], [0.25, 0.25, 0.25, 0.25])
        total = sum(qcfg)
        cluster_quad[c] = [v / total for v in qcfg]

    # ── MC Simulation ──
    sim_revenues, sim_retentions = [], []
    MAX_PUSHES, lam_pen, mu_pen = 10, 0.003, 0.001

    for sim_idx in range(N_MC_SIMS):
        if sim_idx % 1000 == 0 and sim_idx > 0:
            print(f'  Progress: {sim_idx}/{N_MC_SIMS}')
        total_revenue, retained = 0, 0
        for c, (cp, n_players) in enumerate(zip(cluster_profiles, cl_sizes)):
            cd = cl_dists[c]
            probs = cluster_probs[c]
            name = cp['name_cn']
            is_paying = cd['pay_rate'] > 0
            is_whale = (name == '高氪核心党')

            if is_paying:
                # ── Paying cluster: v3 full push + additive uplift (no quadrants) ──
                # Paid users have proven willingness; pushing only increases spend.
                # Uplift literature: risk of Sleeping Dog is minimal for payers.
                for _ in range(n_players):
                    lifecycle = max(1, np.random.normal(cd['lifecycle_mean'], cd['lifecycle_std']))
                    organic_pay = np.random.exponential(cd['mean_pay']) if np.random.random() < cd['pay_rate'] else 0
                    strategy_rev, n_pushes, total_price = 0, 0, 0
                    highest_paid = 0
                    total_ret_boost = 0.0

                    for gp, pp in probs:
                        if n_pushes >= MAX_PUSHES:
                            break
                        if gp['timing'] > lifecycle:
                            continue
                        if gp['price'] <= highest_paid:
                            continue
                        if lifecycle < 3 and gp['price'] > 30:
                            continue
                        if is_whale and gp['price'] < 30:
                            continue
                        if highest_paid >= 128 and gp['price'] < 68:
                            continue
                        if np.random.random() < pp:
                            strategy_rev += gp['price']
                            highest_paid = gp['price']
                            total_ret_boost += gp.get('ret_boost', 0.01)
                        n_pushes += 1
                        total_price += gp['price']

                    ret_penalty = lam_pen * n_pushes + mu_pen * total_price / 100
                    total_revenue += organic_pay + strategy_rev
                    if np.random.random() < min(0.99, cd['retention_base'] + total_ret_boost - ret_penalty):
                        retained += 1

            else:
                # ── Zero-pay cluster: v4 four-quadrant + free retention pack ──
                quad_dist = cluster_quad[c]
                for _ in range(n_players):
                    lifecycle = max(1, np.random.normal(cd['lifecycle_mean'], cd['lifecycle_std']))
                    organic_pay = 0  # zero-pay, no organic
                    strategy_rev, n_pushes, total_price = 0, 0, 0
                    highest_paid = 0
                    total_ret_boost = 0.0

                    quad = np.random.choice(4, p=quad_dist)

                    if quad == QUAD_PERSUADABLE:
                        for gp, pp in probs:
                            if n_pushes >= MAX_PUSHES:
                                break
                            if gp['timing'] > lifecycle:
                                continue
                            if gp['price'] <= highest_paid:
                                continue
                            if lifecycle < 3 and gp['price'] > 30:
                                continue
                            if np.random.random() < pp:
                                strategy_rev += gp['price']
                                highest_paid = gp['price']
                                total_ret_boost += gp.get('ret_boost', 0.01)
                            n_pushes += 1
                            total_price += gp['price']
                        ret_penalty = lam_pen * n_pushes + mu_pen * total_price / 100

                    elif quad == QUAD_LOST:
                        if free_pack['timing'] <= lifecycle:
                            n_pushes = 1
                            total_ret_boost = free_pack['ret_boost']
                        ret_penalty = 0

                    else:
                        # Sure Thing / Sleeping Dog: no push
                        ret_penalty = 0

                    total_revenue += organic_pay + strategy_rev
                    if np.random.random() < min(0.99, cd['retention_base'] + total_ret_boost - ret_penalty):
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
            'revenues': revs, 'retentions': rets, 'label': f'Target v4 [{sens_label}]'}


def build_mc_state(cluster_profiles, df):
    """Shared cluster sizes and distributions for all MC schemes."""
    sizes = []
    for cp in cluster_profiles:
        sizes.append(max(1, int(cp['pct']/100*N_SIM_PLAYERS)))
    sizes[-1] += N_SIM_PLAYERS - sum(sizes)
    dists = {}
    for c, cp in enumerate(cluster_profiles):
        sub = df[df['cluster'] == c]
        dists[c] = {
            'lifecycle_mean': sub['lifecycle_days'].mean(),
            'lifecycle_std': max(0.5, sub['lifecycle_days'].std()),
            'pay_rate': cp['pay_rate']/100,
            'mean_pay': max(0.1, cp['mean_pay']),
            'retention_base': cp['retention_30d']/100,
        }
    return sizes, dists


def monte_carlo_baseline(cluster_profiles, df, cl_sizes, cl_dists):
    """No-intervention baseline: organic payments only."""
    print('\n' + '=' * 50)
    print('3.3b Baseline (No Intervention)')
    print('=' * 50)
    np.random.seed(RANDOM_SEED)
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

    # ── Trigger conditions integrate Problem 1 (churn risk) & Problem 2 (diamond threshold 566) ──
    gift_packs = [
        {'name_cn': '免费挽留包', 'name_en': 'Free Retention Pack', 'price': 0,
         'timing_cn': 'Day 1-30', 'timing_en': 'Day 1-30',
         'trigger_cn': 'Lost Cause象限 (高流失风险零氪)', 'trigger_en': 'Lost Cause quadrant (high-churn F2P)',
         'targets': ['F2P Churner']},
        {'name_cn': '首充礼包', 'name_en': 'First Purchase Pack', 'price': 6,
         'timing_cn': 'Day 1', 'timing_en': 'Day 1',
         'trigger_cn': '注册首日自动推送 (Persuadable仅)', 'trigger_en': 'Auto Day1 (Persuadable only)',
         'targets': ['F2P Casual', 'Mid Spender']},
        {'name_cn': '新手补给', 'name_en': 'Novice Supply Pack', 'price': 12,
         'timing_cn': 'Day 3', 'timing_en': 'Day 3',
         'trigger_cn': '活跃>=2天且未付费 (Persuadable仅)', 'trigger_en': 'Active>=2d & no payment (Persuadable only)',
         'targets': ['F2P Casual', 'F2P Churner']},
        {'name_cn': '资源补给包', 'name_en': 'Resource Supply Pack', 'price': 30,
         'timing_cn': 'Day 7', 'timing_en': 'Day 7',
         'trigger_cn': '活跃>=5天或钻石<566 (Persuadable仅, 问题2阈值)', 'trigger_en': 'Active>=5d or diamond<566 (Persuadable, P2 threshold)',
         'targets': ['Mid Spender']},
        {'name_cn': '成长加速包', 'name_en': 'Growth Accelerator', 'price': 68,
         'timing_cn': 'Day 3-5', 'timing_en': 'Day 3-5',
         'trigger_cn': '连续2天资源缺口>30% (Persuadable仅)', 'trigger_en': '2-day resource gap >30% (Persuadable only)',
         'targets': ['Mid Spender']},
        {'name_cn': '战力突破包', 'name_en': 'Power Breakthrough', 'price': 128,
         'timing_cn': 'Day 7-10', 'timing_en': 'Day 7-10',
         'trigger_cn': '等级停滞>=3天 (Persuadable仅)', 'trigger_en': 'Level stagnation >=3d (Persuadable only)',
         'targets': ['Mid Spender', 'Whale']},
        {'name_cn': '至尊大礼包', 'name_en': 'Ultimate Pack', 'price': 328,
         'timing_cn': 'Day 21', 'timing_en': 'Day 21',
         'trigger_cn': '等级>=10且已购过≥68元包 (Persuadable仅)', 'trigger_en': 'Level>=10 & bought >=68¥ pack (Persuadable)',
         'targets': ['Whale']},
        {'name_cn': '传说大礼包', 'name_en': 'Legendary Pack', 'price': 648,
         'timing_cn': 'Day 28', 'timing_en': 'Day 28',
         'trigger_cn': '已购过328元包 (Persuadable高氪仅)', 'trigger_en': 'Bought 328¥ pack (Persuadable Whale only)',
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

    # Shared MC state
    cl_sizes, cl_dists = build_mc_state(cluster_profiles, df)

    # Three MC simulations
    mc_baseline = monte_carlo_baseline(cluster_profiles, df, cl_sizes, cl_dists)
    mc_conservative = monte_carlo_conservative(cluster_profiles, df, cl_sizes, cl_dists)

    # Target scheme: baseline + 2 sensitivity variants
    mc_target_base = monte_carlo_target(cluster_profiles, df, demand, beta_hat,
                                        cl_sizes, cl_dists, sens_label='基准')

    # Sensitivity A: Persuadable比例 -20% (pessimistic)
    quad_pessimistic = {k: [v[0]*0.8, v[1]+v[0]*0.2, v[2], v[3]] for k, v in {
        "高氪核心党": [0.40, 0.60, 0.00, 0.00],
        "中氪战力党": [0.50, 0.40, 0.10, 0.00],
        "零氪休闲党": [0.60, 0.00, 0.05, 0.35],
        "零氪流失党": [0.20, 0.00, 0.05, 0.75],
    }.items()}
    mc_target_pes = monte_carlo_target(cluster_profiles, df, demand, beta_hat,
                                       cl_sizes, cl_dists,
                                       quadrant_config=quad_pessimistic,
                                       sens_label='悲观')

    # Sensitivity B: Persuadable比例 +20% (optimistic)
    quad_optimistic = {k: [min(1.0, v[0]*1.2), max(0, v[1]-v[0]*0.2), v[2], v[3]] for k, v in {
        "高氪核心党": [0.40, 0.60, 0.00, 0.00],
        "中氪战力党": [0.50, 0.40, 0.10, 0.00],
        "零氪休闲党": [0.60, 0.00, 0.05, 0.35],
        "零氪流失党": [0.20, 0.00, 0.05, 0.75],
    }.items()}
    mc_target_opt = monte_carlo_target(cluster_profiles, df, demand, beta_hat,
                                       cl_sizes, cl_dists,
                                       quadrant_config=quad_optimistic,
                                       sens_label='乐观')

    sched_df = generate_strategy_schedule(cluster_profiles)

    results = [
        '=== 问题3 v4 结果 ===',
        f'最优聚类数: {len(cluster_profiles)}',
        '',
        '--- 无干预基线（自然有机付费） ---',
        f'期望营收: {mc_baseline["mean_revenue"]:.0f} CNY',
        f'期望留存率: {mc_baseline["mean_retention"]*100:.1f}%',
        '',
        '--- 主方案：历史分布增强方案 ---',
        f'期望营收: {mc_conservative["mean_revenue"]:.0f} CNY',
        f'P(营收>=70000): {mc_conservative["prob_revenue"]*100:.1f}%',
        f'期望留存率: {mc_conservative["mean_retention"]*100:.1f}%',
        '',
        '--- 探索方案v4：四象限+增量叠加+免费挽留+状态触发 ---',
        f'[基准] 期望营收: {mc_target_base["mean_revenue"]:.0f} CNY, 留存: {mc_target_base["mean_retention"]*100:.1f}%',
        f'[悲观] 期望营收: {mc_target_pes["mean_revenue"]:.0f} CNY, 留存: {mc_target_pes["mean_retention"]*100:.1f}%',
        f'[乐观] 期望营收: {mc_target_opt["mean_revenue"]:.0f} CNY, 留存: {mc_target_opt["mean_retention"]*100:.1f}%',
        f'[基准] P(>=70K): {mc_target_base["prob_revenue"]*100:.1f}%',
        '',
        '--- 对比 ---',
        f'主方案 vs 基线: {mc_conservative["mean_revenue"]/max(1,mc_baseline["mean_revenue"]):.1f}x',
        f'探索方案(基准) vs 主方案: {mc_target_base["mean_revenue"]/max(1,mc_conservative["mean_revenue"]):.1f}x',
        f'探索方案(乐观) vs 主方案: {mc_target_opt["mean_revenue"]/max(1,mc_conservative["mean_revenue"]):.1f}x',
    ]
    with open(os.path.join(RES_DIR, '问题3_results.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    print('\nResults saved.')


if __name__ == '__main__':
    main()
