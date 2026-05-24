"""
Generate overall modeling flowchart for the paper (Nature style).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, numpy as np

# ── Nature palette ──
BLUE   = '#0F4D92'
GREEN  = '#8BCF8B'
RED    = '#B64342'
ORANGE = '#E8923F'
PURPLE = '#7B5EA7'
GRAY   = '#6B6B6B'
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

def draw_rounded_box(ax, x, y, w, h, text, color, fontsize=9, text_color='white', bold=False):
    """Draw a rounded rectangle with centered text."""
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.15", facecolor=color, edgecolor='white', linewidth=1.2, alpha=0.92
    )
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight=weight)

def draw_arrow(ax, x1, y1, x2, y2, color=GRAY, lw=1.5):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle='arc3,rad=0'))

def draw_label(ax, x, y, text, color=DARK, fontsize=7.5, rotation=0):
    """Draw a small text label."""
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=color, rotation=rotation)

def draw_flowchart(lang='cn'):
    """Draw the full modeling pipeline flowchart."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_facecolor(BG)

    labels_cn = {
        'title': '整体建模流程图',
        'data': '原始数据\n资源变更 + 行为事件',
        'fe': '特征工程',
        'fe_d3': 'Day1–3\n可部署特征',
        'fe_full': '全周期\n解释特征',
        'q1': '问题1: 留存规律挖掘',
        'q1_m1': 'KM生存估计',
        'q1_m2': 'Cox PH\n(+ RSF对照)',
        'q1_m3': '分层留存\n对比分析',
        'q1_out': '流失风险因子',
        'q2': '问题2: 付费关联分析',
        'q2_m1': '资源–等级\n相关性',
        'q2_m2': 'Logistic\n阈值识别',
        'q2_m3': '两阶段\nHurdle模型',
        'q2_out': '付费驱动因子',
        'q3': '问题3: 差异化策略设计',
        'q3_m1': 'K-Means\n聚类分群',
        'q3_m2': 'Uplift\n四象限框架',
        'q3_m3': '蒙特卡洛\n模拟验证',
        'q4': '问题4: 数据采集与闭环',
        'q4_m1': '社交 / 进度\n补采方向',
        'q4_m2': 'A/B测试\n验证框架',
        'legend': '图例:',
        'leg_data': '数据层',
        'leg_feat': '特征层',
        'leg_model': '模型层',
        'leg_strategy': '策略层',
    }
    labels_en = {
        'title': 'Overall Modeling Framework',
        'data': 'Raw Data\nResource Logs +\nBehavior Events',
        'fe': 'Feature Engineering',
        'fe_d3': 'Day 1–3\nDeployable\nFeatures',
        'fe_full': 'Full-Period\nExplanatory\nFeatures',
        'q1': 'Q1: Retention Analysis',
        'q1_m1': 'KM Survival\nEstimation',
        'q1_m2': 'Cox PH\n(+ RSF Baseline)',
        'q1_m3': 'Segmented\nRetention',
        'q1_out': 'Churn\nRisk Factors',
        'q2': 'Q2: Payment Analysis',
        'q2_m1': 'Resource–Level\nCorrelation',
        'q2_m2': 'Logistic\nThreshold',
        'q2_m3': 'Two-Part\nHurdle Model',
        'q2_out': 'Payment\nDrivers',
        'q3': 'Q3: Strategy Optimization',
        'q3_m1': 'K-Means\nClustering',
        'q3_m2': 'Uplift\nFour-Quadrant',
        'q3_m3': 'Monte Carlo\nSimulation',
        'q4': 'Q4: Data Collection & A/B Test',
        'q4_m1': 'Social / Progress\nData Gaps',
        'q4_m2': 'A/B Testing\nFramework',
        'legend': 'Legend:',
        'leg_data': 'Data',
        'leg_feat': 'Feature',
        'leg_model': 'Model',
        'leg_strategy': 'Strategy',
    }
    L = labels_cn if lang == 'cn' else labels_en

    # ── Title ──
    ax.text(7, 8.55, L['title'], ha='center', va='center', fontsize=16,
            fontweight='bold', color=DARK)

    # ═══ ROW 1: DATA ═══
    draw_rounded_box(ax, 7, 7.4, 5.5, 1.1, L['data'], BLUE, fontsize=10, bold=True)

    # Arrow: data → feature engineering
    draw_arrow(ax, 7, 6.82, 7, 6.45, GRAY)

    # ═══ ROW 2: FEATURE ENGINEERING ═══
    draw_rounded_box(ax, 7, 6.1, 2.8, 0.7, L['fe'], PURPLE, fontsize=9, bold=True)

    # Arrows: feature engineering → two feature tables
    draw_arrow(ax, 5.6, 5.75, 2.3, 5.0, GRAY)
    draw_arrow(ax, 8.4, 5.75, 11.7, 5.0, GRAY)

    # Feature tables
    draw_rounded_box(ax, 2.3, 4.65, 2.8, 0.7, L['fe_d3'], PURPLE, fontsize=7.5, text_color=WHITE)
    draw_rounded_box(ax, 11.7, 4.65, 2.8, 0.7, L['fe_full'], PURPLE, fontsize=7.5, text_color=WHITE)

    # ═══ ROW 3: Q1 (left) + Q2 (right) ═══
    # Arrows: feature tables → Q1 / Q2
    draw_arrow(ax, 2.3, 4.3, 2.3, 3.9, GRAY)
    draw_arrow(ax, 11.7, 4.3, 11.7, 3.9, GRAY)

    # Q1 header
    draw_rounded_box(ax, 2.3, 3.55, 3.8, 0.7, L['q1'], RED, fontsize=9, bold=True)
    # Q1 methods
    draw_rounded_box(ax, 0.6, 2.5, 1.6, 0.6, L['q1_m1'], RED, fontsize=6.5, text_color=WHITE)
    draw_rounded_box(ax, 2.3, 2.5, 1.6, 0.6, L['q1_m2'], RED, fontsize=6.5, text_color=WHITE)
    draw_rounded_box(ax, 1.45, 1.65, 3.0, 0.6, L['q1_m3'], RED, fontsize=6.5, text_color=WHITE)
    # Q1 output
    draw_rounded_box(ax, 1.45, 0.85, 3.0, 0.55, L['q1_out'], '#E8A0A0', fontsize=7.5, text_color=DARK, bold=True)

    # Arrows Q1
    draw_arrow(ax, 2.3, 3.2, 0.6, 2.8, GRAY, lw=1.2)
    draw_arrow(ax, 2.3, 3.2, 2.3, 2.8, GRAY, lw=1.2)
    draw_arrow(ax, 0.6, 2.2, 1.45, 1.95, GRAY, lw=1.2)
    draw_arrow(ax, 2.3, 2.2, 1.45, 1.95, GRAY, lw=1.2)
    draw_arrow(ax, 1.45, 1.35, 1.45, 1.15, GRAY, lw=1.2)

    # Q2 header
    draw_rounded_box(ax, 11.7, 3.55, 3.8, 0.7, L['q2'], ORANGE, fontsize=9, bold=True)
    # Q2 methods
    draw_rounded_box(ax, 10.0, 2.5, 1.6, 0.6, L['q2_m1'], ORANGE, fontsize=6.5, text_color=WHITE)
    draw_rounded_box(ax, 11.7, 2.5, 1.6, 0.6, L['q2_m2'], ORANGE, fontsize=6.5, text_color=WHITE)
    draw_rounded_box(ax, 13.4, 2.5, 1.6, 0.6, L['q2_m3'], ORANGE, fontsize=6.5, text_color=WHITE)
    # Q2 output
    draw_rounded_box(ax, 12.55, 0.85, 3.0, 0.55, L['q2_out'], '#F5C8A0', fontsize=7.5, text_color=DARK, bold=True)

    # Arrows Q2
    draw_arrow(ax, 11.7, 3.2, 10.0, 2.8, GRAY, lw=1.2)
    draw_arrow(ax, 11.7, 3.2, 11.7, 2.8, GRAY, lw=1.2)
    draw_arrow(ax, 11.7, 3.2, 13.4, 2.8, GRAY, lw=1.2)
    draw_arrow(ax, 10.0, 2.2, 12.55, 1.15, GRAY, lw=1.2)
    draw_arrow(ax, 11.7, 2.2, 12.55, 1.15, GRAY, lw=1.2)
    draw_arrow(ax, 13.4, 2.2, 12.55, 1.15, GRAY, lw=1.2)

    # ═══ ROW 3: Q3 (center, receives from Q1+Q2) ═══
    # Arrows from Q1/Q2 to Q3
    draw_arrow(ax, 2.3, 3.2, 5.5, 4.15, GRAY, lw=2)
    draw_arrow(ax, 11.7, 3.2, 8.5, 4.15, GRAY, lw=2)

    # Q3 header
    draw_rounded_box(ax, 7, 4.55, 4.5, 0.75, L['q3'], GREEN, fontsize=9, text_color=DARK, bold=True)

    # Q3 methods
    draw_rounded_box(ax, 5.0, 3.55, 1.8, 0.6, L['q3_m1'], GREEN, fontsize=6.5, text_color=DARK)
    draw_rounded_box(ax, 7.0, 3.55, 1.8, 0.6, L['q3_m2'], GREEN, fontsize=6.5, text_color=DARK)
    draw_rounded_box(ax, 9.0, 3.55, 1.8, 0.6, L['q3_m3'], GREEN, fontsize=6.5, text_color=DARK)

    # Arrows Q3
    draw_arrow(ax, 7, 4.15, 5.0, 3.85, GRAY, lw=1.2)
    draw_arrow(ax, 7, 4.15, 7.0, 3.85, GRAY, lw=1.2)
    draw_arrow(ax, 7, 4.15, 9.0, 3.85, GRAY, lw=1.2)

    # ═══ Q4 below Q3 ═══
    draw_arrow(ax, 7, 3.25, 7, 2.95, GRAY, lw=1.5)

    draw_rounded_box(ax, 7, 2.55, 4.5, 0.75, L['q4'], GRAY, fontsize=9, text_color=WHITE, bold=True)

    # Q4 methods
    draw_rounded_box(ax, 5.5, 1.6, 2.0, 0.55, L['q4_m1'], GRAY, fontsize=7, text_color=WHITE)
    draw_rounded_box(ax, 8.5, 1.6, 2.0, 0.55, L['q4_m2'], GRAY, fontsize=7, text_color=WHITE)

    draw_arrow(ax, 7, 2.15, 5.5, 1.9, GRAY, lw=1.2)
    draw_arrow(ax, 7, 2.15, 8.5, 1.9, GRAY, lw=1.2)

    # ═══ LEGEND ═══
    lx, ly = 11.5, 8.3
    ax.text(lx, ly, L['legend'], fontsize=6.5, color=DARK, fontweight='bold')
    for i, (color, label_key) in enumerate([
        (BLUE, 'leg_data'), (PURPLE, 'leg_feat'), (RED, 'leg_model'), (GREEN, 'leg_strategy')
    ]):
        y = ly - 0.25 * (i + 1)
        ax.add_patch(mpatches.Rectangle((lx, y-0.08), 0.35, 0.16,
                      facecolor=color, edgecolor='white', linewidth=0.5))
        ax.text(lx + 0.5, y, L[label_key], fontsize=6, color=DARK, va='center')

    plt.tight_layout(pad=0.5)
    return fig

# ── Generate both languages ──
for lang, subdir in [('cn', '中文版'), ('en', '英文版')]:
    set_font(lang)
    fig = draw_flowchart(lang)
    for fmt, dpi, ext in [('pdf', None, 'pdf'), ('png', 300, 'png'), ('svg', None, 'svg')]:
        path = os.path.join(FIG_DIR, subdir, f'figure_flowchart.{ext}')
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close(fig)
    print(f'{subdir} flowchart saved (PNG+PDF+SVG)')

print('Done.')
