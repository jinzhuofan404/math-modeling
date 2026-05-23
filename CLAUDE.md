# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

电子科技大学数学建模竞赛仓库。当前选做 **B题：策略类游戏的用户留存与付费策略优化**。

### 当前进度

| 阶段 | 状态 | 产出 |
|------|------|------|
| 建模分析 | 已完成 | `B题_题目分析报告.md`, `B题_术语表格.md` |
| 代码实现 | 已完成 | `B题/特征工程.py`, `问题1-4_求解.py` |
| Nature图表 | 已完成 | 15图×2语言×4格式(SVG/PDF/TIFF/PNG) |
| LaTeX论文 | 已完成 | `论文/main.pdf` 28页（正文24页） |
| 论文修订 | 已完成 | 摘要重写、引用改中括号、排版压缩、附录补充 |

## 题目结构

### A题：库位分配优化问题（未选）

### B题：策略类游戏的用户留存与付费策略优化（已选）

实际数据是两类文件：

| 实际文件 | 内容 | 规模 |
|----------|------|------|
| pickdata1.csv | 资源变更记录(33字段) | ~193万行 |
| pickdata2.csv | 用户行为事件(52字段) | ~416万行 |
| pick_stat1/2.csv | 各用户记录条数统计 | 2000行 |
| other_pickdata1/2.csv | 测试集(1000用户) | ~7GB |

**关键字段映射：**
- 付费金额 → pickdata2的`total_pay`（以USD标价，论文统一按1 USD ≈ 6 CNY折算）
- 资源消耗 → pickdata1的`change_type='reduce'`记录
- 联盟状态 → pickdata1的`league_name`非空
- 流失判定 → 连续2天无pickdata2行为记录

## 项目结构

```
数学建模/
├── CLAUDE.md
├── 当前进度.md
├── B题/
│   ├── 特征工程.py          # 分块读取→玩家级特征表(2000×39)
│   ├── 问题1_求解.py        # KM+Cox生存分析
│   ├── 问题2_求解.py        # Hurdle模型(XGBoost+Gamma GLM)+SHAP
│   ├── 问题3_求解.py        # K-Means聚类+蒙特卡洛(5000次)
│   ├── 问题4_求解.py        # 数据采集建议
│   ├── nature_figures.py    # 15图×4格式(SVG/PDF/TIFF/PNG)
│   ├── data/                # player_features.csv
│   ├── results/             # 各问题结果TXT
│   └── figures/
│       └── nature/
│           ├── 中文版/       # 15图
│           └── 英文版/
├── 论文/
│   ├── main.tex             # ctexart, XeLaTeX编译
│   ├── main.pdf             # 28页（正文24页）
│   └── sections/            # sec1-11分章节
└── 题目/                    # 竞赛原始数据（大CSV未入git）
```

## 代码规范

### 图表生成（Nature风格）

- **中文字体**：`SimHei`。英文用`DejaVu Sans, Arial`。
- **导出格式**：SVG + PDF + TIFF(600dpi) + PNG(300dpi)
- **配色**：Nature色板 `PAL = {"blue":"#0F4D92", "green":"#8BCF8B", "red":"#B64342", ...}`
- **子图限制**：每个大图最多2个子图

### Python编码

- 大数据用`pd.read_csv(chunksize=200000)`分块
- 随机种子统一用42
- 每个求解脚本可独立运行
- 结果输出到终端 + TXT

### LaTeX论文

- 编译器：XeLaTeX（ctex中文）
- 字体：12pt（小四号宋体）
- 页边距：2cm；正文+参考文献≤25页
- 图片路径：`\graphicspath{{../B题/figures/nature/中文版/}}`
- 引用格式：中括号数字（natbib numbers）
- 编译至少2遍以解析交叉引用

## 核心分析结果

| 指标 | 数值 |
|------|------|
| 次日/7日/14日/30日留存 | 33.55% / 6.30% / 2.65% / 0.85% |
| 中位生存时间 | 1天 |
| 付费用户占比 | 3.75% (75/2000) |
| 付费者7日留存 | 40.6% (vs 非付费5.2%) |
| 入盟者7日留存 | 28.0% (vs 未入盟1.2%) |
| 钻石流失风险阈值 | 566 |
| Cox C-index | 0.899 |
| Cox MAE | 1.33天 |
| 最优聚类数 | K=6 (轮廓系数0.490) |
| Hurdle阶段1 AUC | 0.9315 |
| Hurdle阶段2 Deviance explained | 0.839 |
| 优化方案期望营收 | 71,045元 (10K玩家) |
| 优化方案留存率 | 11.3% |
| P(营收≥70,000) | 52.4% |
| P(留存≥10%) | 98% |

## Git工作流

- 不提交大CSV文件（pickdata1/2, other_pickdata1/2, 约5GB）
- 不提交`.claude/`目录
- Commit message格式: `主题：简明描述`
- 每次commit后手动push到GitHub
