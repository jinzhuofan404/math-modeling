"""
Batch parameter tuning: 6 combinations x 500 MC iterations (fast scan).
Select best combo for full 5000-iter verification.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

# ── Reduce MC iterations for fast scan ──
import 问题3_求解 as q3
q3.N_MC_SIMS = 500

from 问题3_求解 import (
    load_features, player_clustering, estimate_demand_curves,
    build_mc_state, monte_carlo_baseline, monte_carlo_target,
    RANDOM_SEED,
)

# ── 6 combos ──
combos = [
    {'label': 'A', 'desc': '当前基线(对照)', 'lam_x': 0.25, 'lam_l': 0.10, 'd_zk': 0.60, 'd_gk': 0.80, 'bm': 0.35},
    {'label': 'B', 'desc': '保守上调',       'lam_x': 0.30, 'lam_l': 0.10, 'd_zk': 0.85, 'd_gk': 0.80, 'bm': 0.30},
    {'label': 'C', 'desc': '激进上调',       'lam_x': 0.35, 'lam_l': 0.10, 'd_zk': 1.00, 'd_gk': 0.80, 'bm': 0.27},
    {'label': 'D', 'desc': '仅λ+δ',        'lam_x': 0.30, 'lam_l': 0.10, 'd_zk': 0.85, 'd_gk': 0.80, 'bm': 0.35},
    {'label': 'E', 'desc': '仅δ+β',        'lam_x': 0.25, 'lam_l': 0.10, 'd_zk': 0.85, 'd_gk': 0.80, 'bm': 0.30},
    {'label': 'F', 'desc': 'λ+δ激进β适中', 'lam_x': 0.30, 'lam_l': 0.10, 'd_zk': 1.00, 'd_gk': 0.80, 'bm': 0.30},
]

print('=' * 70)
print('BATCH PARAMETER TUNING (K=5, 500 MC each)')
print('=' * 70)

# ── Load & cluster once ──
df = load_features()
print(f'Loaded {len(df)} players')

cluster_profiles, df = player_clustering(df)
demand, beta_hat = estimate_demand_curves(cluster_profiles, df)
cl_sizes, cl_dists = build_mc_state(cluster_profiles, df)

baseline = monte_carlo_baseline(cluster_profiles, df, cl_sizes, cl_dists)
print(f'Baseline: {baseline["mean_revenue"]:.0f} CNY, ret {baseline["mean_retention"]*100:.1f}%')

# ── Run each combo ──
results = []
for cfg in combos:
    beta_tuned = beta_hat * cfg['bm']
    lam = {"零氪休闲党": cfg['lam_x'], "零氪流失党": cfg['lam_l']}
    delta = {"中氪战力党": cfg['d_zk'], "高氪核心党": cfg['d_gk']}

    print(f'\nCombo {cfg["label"]} [{cfg["desc"]}]: λ_x={cfg["lam_x"]}, λ_l={cfg["lam_l"]}, δ_zk={cfg["d_zk"]}, β_mul={cfg["bm"]}')

    mc = monte_carlo_target(cluster_profiles, df, demand, beta_tuned,
                            cl_sizes, cl_dists, sens_label=cfg['label'],
                            lambda_by_name=lam, delta_by_name=delta)

    results.append({
        **cfg,
        'revenue': mc['mean_revenue'],
        'retention': mc['mean_retention'] * 100,
        'prob_70k': mc['prob_revenue'] * 100,
    })
    print(f'  => Revenue: {mc["mean_revenue"]:.0f} CNY, Retention: {mc["mean_retention"]*100:.1f}%, P(>=70K): {mc["prob_revenue"]*100:.1f}%')

# ── Summary table ──
print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'{"Lbl":>4} {"Desc":16s} {"λ休闲":>6} {"δ中氪":>6} {"β_mul":>6} {"Revenue":>10} {"Ret%":>7} {"P>=70K":>7}')
print('-' * 65)
for r in results:
    print(f'{r["label"]:>4} {r["desc"]:16s} {r["lam_x"]:>6.2f} {r["d_zk"]:>6.2f} {r["bm"]:>6.2f} {r["revenue"]:>10.0f} {r["retention"]:>6.1f}% {r["prob_70k"]:>6.1f}%')

# Pick best: retention >= 10%, max revenue
valid = [r for r in results if r['retention'] >= 10.0]
if valid:
    best = max(valid, key=lambda r: r['revenue'])
    print(f'\n>>> Best (ret>=10%): Combo {best["label"]} [{best["desc"]}]')
    print(f'    Revenue={best["revenue"]:.0f} CNY, Retention={best["retention"]:.1f}%, P(>=70K)={best["prob_70k"]:.1f}%')
else:
    best = max(results, key=lambda r: r['revenue'])
    print(f'\n>>> No combo met retention>=10%. Best revenue: Combo {best["label"]} [{best["desc"]}]')

best_any = max(results, key=lambda r: r['revenue'])
print(f'>>> Best revenue (any ret): Combo {best_any["label"]} [{best_any["desc"]}], {best_any["revenue"]:.0f} CNY')

print('\nDone. Verify best combo with 5000 MC iterations.')
