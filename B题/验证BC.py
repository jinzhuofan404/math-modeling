"""Verify Combos B and C with 5000 MC iterations each."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import 问题3_求解 as q3
q3.N_MC_SIMS = 5000
from 问题3_求解 import *

# Load once
df = load_features()
cluster_profiles, df = player_clustering(df)
demand, beta_hat = estimate_demand_curves(cluster_profiles, df)
cl_sizes, cl_dists = build_mc_state(cluster_profiles, df)

baseline = monte_carlo_baseline(cluster_profiles, df, cl_sizes, cl_dists)
print(f'Baseline: {baseline["mean_revenue"]:.0f} CNY\n')

for label, lam_x, d_zk, bm, desc in [
    ('B', 0.30, 0.85, 0.30, '保守上调'),
    ('C', 0.35, 1.00, 0.27, '激进上调'),
]:
    beta_tuned = beta_hat * bm
    lam = {"零氪休闲党": lam_x, "零氪流失党": 0.10}
    delta = {"中氪战力党": d_zk, "高氪核心党": 0.80}
    print(f'\n{"="*60}')
    print(f'VERIFY Combo {label} [{desc}]: λ_x={lam_x}, δ_zk={d_zk}, β_mul={bm}')
    print(f'{"="*60}')
    mc = monte_carlo_target(cluster_profiles, df, demand, beta_tuned,
                            cl_sizes, cl_dists, sens_label=label,
                            lambda_by_name=lam, delta_by_name=delta)
    print(f'DONE: Revenue={mc["mean_revenue"]:.0f}, Retention={mc["mean_retention"]*100:.1f}%, P(>=70K)={mc["prob_revenue"]*100:.1f}%')
