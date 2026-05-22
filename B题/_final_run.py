"""Final 5000-MC run with explicit file output."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

# Patch N_MC_SIMS before importing so all functions see it
import 问题3_求解 as m
m.N_MC_SIMS = 5000

out_path = os.path.join(m.RES_DIR, '问题3_final_results.txt')
log = open(out_path, 'w', encoding='utf-8', buffering=1)  # line-buffered

def log_print(s):
    print(s)
    log.write(s + '\n')

log_print('=== Problem 3 Final Run (5000 MC) ===')

df = m.load_features()
log_print(f'Loaded {len(df)} players')

cp, df2 = m.player_clustering(df)
demand, bh = m.estimate_demand_curves(cp, df2)
cl_sizes, cl_dists = m.build_mc_state(cp, df2)

log_print('\n>>> BASELINE <<<')
mb = m.monte_carlo_baseline(cp, df2, cl_sizes, cl_dists)

log_print('\n>>> CONSERVATIVE <<<')
mc = m.monte_carlo_conservative(cp, df2, cl_sizes, cl_dists)

log_print('\n>>> TARGET <<<')
mt = m.monte_carlo_target(cp, df2, demand, bh, cl_sizes, cl_dists, sens_label='Final')

log_print('\n' + '=' * 60)
log_print('FINAL RESULTS (5000 MC)')
log_print(f'Baseline:     rev={mb["mean_revenue"]:,.0f} ret={mb["mean_retention"]*100:.1f}%')
log_print(f'Conservative: rev={mc["mean_revenue"]:,.0f} ret={mc["mean_retention"]*100:.1f}%')
log_print(f'Target v5:    rev={mt["mean_revenue"]:,.0f} ret={mt["mean_retention"]*100:.1f}%')
log_print(f'Target/Conserv: {mt["mean_revenue"]/max(1,mc["mean_revenue"]):.3f}x')
log_print(f'P(ret>=10%): {mt["prob_retention"]*100:.1f}%')
log_print(f'P(rev>=70K): {mt["prob_revenue"]*100:.1f}%')
log_print(f'Conserv 95%CI: [{mc["mean_revenue"]-1.96*mc["std_revenue"]/np.sqrt(5000):.0f}, {mc["mean_revenue"]+1.96*mc["std_revenue"]/np.sqrt(5000):.0f}]')
log_print(f'Target 95%CI:  [{mt["mean_revenue"]-1.96*mt["std_revenue"]/np.sqrt(5000):.0f}, {mt["mean_revenue"]+1.96*mt["std_revenue"]/np.sqrt(5000):.0f}]')

# Write results.txt
with open(os.path.join(m.RES_DIR, '问题3_results.txt'), 'w', encoding='utf-8') as f:
    f.write(f'=== 问题3 最终结果 (5000 MC) ===\n')
    f.write(f'聚类数: {len(cp)}\n\n')
    f.write(f'基线(无干预): {mb["mean_revenue"]:.0f} CNY, 留存 {mb["mean_retention"]*100:.1f}%\n')
    f.write(f'保守方案(主方案): {mc["mean_revenue"]:.0f} CNY, 留存 {mc["mean_retention"]*100:.1f}%\n')
    f.write(f'探索方案(混合v5): {mt["mean_revenue"]:.0f} CNY, 留存 {mt["mean_retention"]*100:.1f}%\n')
    f.write(f'探索/保守: {mt["mean_revenue"]/max(1,mc["mean_revenue"]):.3f}x\n')
    f.write(f'P(ret>=10%): {mt["prob_retention"]*100:.1f}%\n')
    f.write(f'P(rev>=70K): {mt["prob_revenue"]*100:.1f}%\n')

log_print('\nDone. Results saved.')
log.close()
