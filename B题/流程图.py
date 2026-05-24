"""
Generate overall modeling flowchart for the paper (Nature style).
Clean layout with large fonts, no overlapping.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Nature palette ──
BLUE   = '#0F4D92'
GREEN  = '#5A9E6F'
RED    = '#B64342'
ORANGE = '#E8923F'
PURPLE = '#7B5EA7'
GRAY   = '#888888'
DARK   = '#2C2C2C'
WHITE  = '#FFFFFF'
BG     = '#FAFAFA'

BASE_DIR = os.path.dirname(__file__)
FIG_DIR = os.path.join(BASE_DIR, 'figures', 'nature')
os.makedirs(os.path.join(FIG_DIR, '中文版'), exist_ok=True)
os.makedirs(os.path.join(FIG_DIR, '英文版'), exist_ok=True)

def set_font(lang):
    if lang == 'cn':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei', 'Arial']
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

def box(ax, x, y, w, h, text, color, fs=11, tc=WHITE, bold=False):
    """Draw a rounded box with text."""
    r = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.2", facecolor=color, edgecolor='none', alpha=0.92, zorder=2
    )
    ax.add_patch(r)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            color=tc, fontweight='bold' if bold else 'normal', zorder=3)

def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=1,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, connectionstyle='arc3,rad=0'))

def draw_flowchart(lang='cn'):
    """Simpler top-down layout with more space."""
    if lang == 'cn':
        T = {
            'title': '整体建模流程图',
            'data': '原始数据\n资源变更记录 + 行为事件记录',
            'fe': '特征工程',
            'fe_left': 'Day1–3可部署特征\n(问题1预测用)',
            'fe_right': '全周期解释特征\n(问题2/3分析用)',
            'q1_title': '问题1：留存规律挖掘',
            'q1_a': 'KM生存估计\n+ 风险率分析',
            'q1_b': 'Cox比例风险模型\n(+ PH诊断 + RSF对照)',
            'q1_c': '付费/入盟分层\n留存对比',
            'q1_out': '→ 流失风险因子',
            'q2_title': '问题2：付费关联分析',
            'q2_a': '资源消耗与等级增长\nPearson相关分析',
            'q2_b': '钻石流失阈值识别\n(Logistic回归, 阈值566)',
            'q2_c': '两阶段Hurdle模型\n(XGBoost + Gamma GLM)',
            'q2_out': '→ 付费驱动因子',
            'q3_title': '问题3：差异化付费策略设计',
            'q3_a': 'K-Means聚类分群\n(K=5, 稳定性约束)',
            'q3_b': 'Uplift四象限框架\n+ 需求曲线估计',
            'q3_c': '蒙特卡洛模拟验证\n(5000次 × 10000人)',
            'q3_out': '→ 优化方案: 67,715元, 留存11.4%',
            'q4_title': '问题4：数据采集与闭环验证',
            'q4_a': '优先补采：社交行为数据 + 实时进度数据',
            'q4_b': 'AB测试框架：实验组 vs 对照组, 30日ARPU+留存对比',
        }
    else:
        T = {
            'title': 'Overall Modeling Framework',
            'data': 'Raw Data\nResource Logs + Behavior Events',
            'fe': 'Feature Engineering',
            'fe_left': 'Day 1–3 Deployable Features\n(for Q1 Prediction)',
            'fe_right': 'Full-Period Explanatory Features\n(for Q2/Q3 Analysis)',
            'q1_title': 'Q1: Retention Pattern Mining',
            'q1_a': 'KM Survival Estimation\n+ Hazard Rate Analysis',
            'q1_b': 'Cox Proportional Hazards\n(+ PH Diagnostics + RSF)',
            'q1_c': 'Paid/League Segmented\nRetention Comparison',
            'q1_out': '→ Churn Risk Factors',
            'q2_title': 'Q2: Payment Correlation Analysis',
            'q2_a': 'Resource–Level Growth\nPearson Correlation',
            'q2_b': 'Diamond Churn Threshold\n(Logistic Regression, 566)',
            'q2_c': 'Two-Part Hurdle Model\n(XGBoost + Gamma GLM)',
            'q2_out': '→ Payment Drivers',
            'q3_title': 'Q3: Differentiated Strategy Design',
            'q3_a': 'K-Means Clustering\n(K=5, Stability Constraint)',
            'q3_b': 'Uplift Four-Quadrant\n+ Demand Curve Estimation',
            'q3_c': 'Monte Carlo Verification\n(5000 runs × 10000 players)',
            'q3_out': '→ Optimal: 67,715 CNY, 11.4% Retention',
            'q4_title': 'Q4: Data Collection & Closed-Loop',
            'q4_a': 'Priority: Social Behavior + Real-Time Progress Data',
            'q4_b': 'A/B Test: Treatment vs Control, 30-day ARPU + Retention',
        }

    fig, ax = plt.subplots(figsize=(14, 16))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 16)
    ax.axis('off')
    ax.set_facecolor(BG)
    ax.text(7, 15.5, T['title'], ha='center', fontsize=18, fontweight='bold', color=DARK)

    y = 14.3

    # ── DATA ──
    box(ax, 7, y, 8, 0.9, T['data'], BLUE, fs=11, bold=True)
    arrow(ax, 7, y-0.45, 7, y-0.9, GRAY, lw=2)
    y -= 1.15

    # ── FEATURE ENGINEERING ──
    box(ax, 7, y, 5, 0.65, T['fe'], PURPLE, fs=11, bold=True)
    arrow(ax, 5.5, y-0.33, 3, y-0.85, GRAY, lw=1.8)
    arrow(ax, 8.5, y-0.33, 11, y-0.85, GRAY, lw=1.8)
    box(ax, 3, y-1.15, 3.8, 0.6, T['fe_left'], PURPLE, fs=8.5)
    box(ax, 11, y-1.15, 3.8, 0.6, T['fe_right'], PURPLE, fs=8.5)
    y -= 1.8

    # ── Q1 (left) + Q2 (right) ──
    arrow(ax, 3, y+0.35, 3, y, GRAY)
    arrow(ax, 11, y+0.35, 11, y, GRAY)

    # Q1
    box(ax, 3, y-0.5, 5, 0.65, T['q1_title'], RED, fs=11, bold=True)
    box(ax, 3, y-1.4, 5, 0.5, T['q1_a'], RED, fs=8.5)
    arrow(ax, 3, y-0.82, 3, y-1.15, GRAY, lw=1.2)
    box(ax, 3, y-2.2, 5, 0.5, T['q1_b'], RED, fs=8.5)
    arrow(ax, 3, y-1.65, 3, y-1.95, GRAY, lw=1.2)
    box(ax, 3, y-3.0, 5, 0.5, T['q1_c'], RED, fs=8.5)
    arrow(ax, 3, y-2.45, 3, y-2.75, GRAY, lw=1.2)
    box(ax, 3, y-3.7, 5, 0.5, T['q1_out'], '#F0C0C0', fs=9, tc=DARK)
    arrow(ax, 3, y-3.25, 3, y-3.45, GRAY, lw=1.2)
    q1_bottom = y - 3.95

    # Q2
    box(ax, 11, y-0.5, 5, 0.65, T['q2_title'], ORANGE, fs=11, bold=True)
    box(ax, 11, y-1.4, 5, 0.5, T['q2_a'], ORANGE, fs=8.5)
    arrow(ax, 11, y-0.82, 11, y-1.15, GRAY, lw=1.2)
    box(ax, 11, y-2.2, 5, 0.5, T['q2_b'], ORANGE, fs=8.5)
    arrow(ax, 11, y-1.65, 11, y-1.95, GRAY, lw=1.2)
    box(ax, 11, y-3.0, 5, 0.5, T['q2_c'], ORANGE, fs=8.5)
    arrow(ax, 11, y-2.45, 11, y-2.75, GRAY, lw=1.2)
    box(ax, 11, y-3.7, 5, 0.5, T['q2_out'], '#F5D0B0', fs=9, tc=DARK)
    arrow(ax, 11, y-3.25, 11, y-3.45, GRAY, lw=1.2)
    q2_bottom = y - 3.95

    y = min(q1_bottom, q2_bottom)

    # ── Merge Q1+Q2 → Q3 ──
    arrow(ax, 3, y, 5.5, y-0.4, GRAY, lw=2)
    arrow(ax, 11, y, 8.5, y-0.4, GRAY, lw=2)
    y -= 0.7

    # ── Q3 ──
    box(ax, 7, y-0.5, 10, 0.65, T['q3_title'], GREEN, fs=11, tc=DARK, bold=True)
    box(ax, 3, y-1.4, 3.5, 0.5, T['q3_a'], GREEN, fs=8.5, tc=DARK)
    box(ax, 7, y-1.4, 3.5, 0.5, T['q3_b'], GREEN, fs=8.5, tc=DARK)
    box(ax, 11, y-1.4, 3.5, 0.5, T['q3_c'], GREEN, fs=8.5, tc=DARK)
    arrow(ax, 7, y-0.82, 3, y-1.15, GRAY, lw=1.2)
    arrow(ax, 7, y-0.82, 7, y-1.15, GRAY, lw=1.2)
    arrow(ax, 7, y-0.82, 11, y-1.15, GRAY, lw=1.2)
    box(ax, 7, y-2.1, 8, 0.5, T['q3_out'], '#C8E6C9', fs=9, tc=DARK)
    arrow(ax, 3, y-1.65, 7, y-1.85, GRAY, lw=1.2)
    arrow(ax, 7, y-1.65, 7, y-1.85, GRAY, lw=1.2)
    arrow(ax, 11, y-1.65, 7, y-1.85, GRAY, lw=1.2)
    y = y - 2.3

    # ── Q4 ──
    arrow(ax, 7, y+0.2, 7, y-0.1, GRAY, lw=2)
    box(ax, 7, y-0.6, 10, 0.65, T['q4_title'], GRAY, fs=11, bold=True)
    box(ax, 4, y-1.4, 4.5, 0.5, T['q4_a'], GRAY, fs=8.5)
    box(ax, 10, y-1.4, 4.5, 0.5, T['q4_b'], GRAY, fs=8.5)
    arrow(ax, 7, y-0.93, 4, y-1.15, GRAY, lw=1.2)
    arrow(ax, 7, y-0.93, 10, y-1.15, GRAY, lw=1.2)

    plt.tight_layout(pad=0.5)
    return fig

# ── Generate ──
for lang, subdir in [('cn', '中文版'), ('en', '英文版')]:
    set_font(lang)
    fig = draw_flowchart(lang)
    for fmt, dpi in [('pdf', None), ('png', 300), ('svg', None)]:
        path = os.path.join(FIG_DIR, subdir, f'figure_flowchart.{fmt}')
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close(fig)
    # Also copy to the simpler figures dir used by paper
    for fmt in ['pdf', 'png', 'svg']:
        import shutil
        src = os.path.join(FIG_DIR, subdir, f'figure_flowchart.{fmt}')
        dst_dir = os.path.join(BASE_DIR, 'figures', subdir)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, os.path.join(dst_dir, f'figure_flowchart.{fmt}'))
    print(f'{subdir} done (PDF+PNG+SVG)')

print('Done.')
