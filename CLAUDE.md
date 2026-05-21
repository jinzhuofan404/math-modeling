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
| LaTeX论文 | 初版完成 | `论文/main.pdf` 24页 |
| 论文修订 | 待进行 | 见"`已知待修复问题`" |

### 已知待修复问题（按优先级）

1. **P0: 问题1存在未来信息泄漏。** Cox模型特征中`daily_activity_rate`分母为`lifecycle_days`（预测目标本身），`level_end`/`total_pay`/`vip_level_max`使用了全周期信息。需重建"仅前3天"特征表，排除`lifecycle_days`, `level_end`, `total_pay`(全周期), `level_growth`(全周期), `vip_level_max`。重跑后C-index会下降但结果真实。
2. **P0: 流失定义代码与论文不一致。** 论文写"连续2天无登录"，代码用`(days_active<20)&(lifecycle_days<30)`。需统一为连续缺席检测+右删失。
3. **P0: 问题3未达70,000元营收目标。** 当前期望22,424元。需新增"目标达成型方案"，引入运营干预弹性（需求曲线估计+最优定价求解），保留当前为"保守方案"形成双方案对比。
4. **P0: 聚类命名代码与论文/CSV不一致。** 代码中动态命名的聚类标签可能与论文表、策略CSV不对齐，需统一。
5. **P1: 问题2语言修正。** 全文搜索"导致/引起/造成"，替换为预测性表述（"与...相关/是...的预测因子"）。
6. **P2: 补充方法选择论证、章节过渡段、Baseline对比表。**

## 题目结构

### A题：库位分配优化问题（未选）

优化原材料在仓库中的库位分配，以最小化生产线订料的取货成本/时间。数据集较小（~280KB），5问层层递进。

### B题：策略类游戏的用户留存与付费策略优化（已选）

**实际数据与题目描述的差异：** 题目描述4张表（玩家基础信息、登录日志、付费流水、资源消耗），但实际数据是两类文件：

| 实际文件 | 内容 | 规模 |
|----------|------|------|
| pickdata1.csv | 资源变更记录(33字段): resource_name, change_type(get/reduce), change_num | ~193万行 |
| pickdata2.csv | 用户行为事件(52字段): event, total_pay, current_diamond/gold, device info | ~416万行 |
| pick_stat1/2.csv | 各用户记录条数统计 | 2000行 |
| other_pickdata1/2.csv | 测试集(1000用户，与训练集互斥) | ~7GB |

**关键字段映射：**
- 付费金额 → pickdata2的`total_pay`（pickdata1的`change_dollar`全为0，不可用）
- 资源消耗 → pickdata1的`change_type='reduce'`记录
- 联盟状态 → pickdata1的`league_name`非空
- 流失判定 → 需从两表时间戳联合推断

## 项目结构

```
数学建模/
├── CLAUDE.md
├── B题/
│   ├── 特征工程.py          # 分块读取→玩家级特征表(2000×39)
│   ├── 问题1_求解.py        # KM+Cox生存分析
│   ├── 问题2_求解.py        # 资源-付费关联+XGBoost+SHAP
│   ├── 问题3_求解.py        # K-Means聚类+蒙特卡洛
│   ├── 问题4_求解.py        # 数据采集建议
│   ├── nature_figures.py    # Nature风格双语图表(15图×4格式)
│   ├── README.md
│   ├── 论文行文大纲.md
│   ├── data/                # player_features.csv
│   ├── results/             # 各问题结果CSV/TXT
│   └── figures/
│       ├── 中文版/           # 旧版中文PNG(已废弃，以nature为准)
│       ├── 英文版/           # 旧版英文PNG(已废弃)
│       └── nature/
│           ├── 中文版/       # 15图×4格式(SVG/PDF/TIFF/PNG)
│           └── 英文版/
├── 论文/
│   ├── main.tex             # ctexart, XeLaTeX编译
│   ├── main.pdf             # 24页输出
│   └── sections/            # sec1-11分章节文件
├── B题_题目分析报告.md
├── B题_术语表格.md
└── 题目/                    # 竞赛原始数据（大CSV未入git）
```

## 代码规范

### 图表生成（Nature风格）

- **中文字体**：`SimHei`（黑体）作为第一字体，备用`Microsoft YaHei`。英文用`DejaVu Sans, Arial, Helvetica`。
- **全局禁止**在rcParams中写死字体——每个language loop内部调用`set_cn_font()`/`set_en_font()`切换。
- **导出格式**：SVG + PDF + TIFF(600dpi) + PNG(300dpi)。`svg.fonttype=none`使文字可编辑。
- **配色**：使用Nature色板 `PAL = {"blue":"#0F4D92", "green":"#8BCF8B", "red":"#B64342", ...}`
- **散点图禁忌**：不要用原始散点展示严重偏态数据（如付费金额96%为0）。用hexbin密度图+对数尺度+分层子图替代。
- **3D图**：逐簇绘制以支持离散配色+图例，设置`view_init`优化视角，透明面板+稀疏网格。
- **子图限制**：每个大图最多2个子图。
- **保存函数**：`save_pub(fig, filename, fig_dir)` 统一处理4格式导出+PermissionError容错。

### Python编码

- 大数据用`pd.read_csv(chunksize=200000)`分块，按account_id分组聚合
- 特征工程输出到`data/player_features.csv`，后续脚本从该文件加载
- 随机种子统一用42
- 每个求解脚本可独立运行，不依赖其他脚本的中间变量
- 结果同时输出到终端 + CSV + TXT

### LaTeX论文

- 编译器：XeLaTeX（支持ctex中文）
- 图片格式：优先PDF矢量，回退PNG（`\DeclareGraphicsExtensions{.pdf,.png,.jpg}`）
- 图片路径：`\graphicspath{{../B题/figures/nature/中文版/}}`
- 编译至少2遍以解析交叉引用和目录
- 忽略`.aux/.log/.out/.toc`等构建产物（已加入`.gitignore`）

## 核心分析结果

| 指标 | 数值 |
|------|------|
| 次日/7日/14日/30日留存 | 42.4% / 23.0% / 16.2% / 7.6% |
| 中位生存时间 | 1天 |
| 付费用户占比 | 3.8% (75/2000) |
| 付费玩家7日留存 | 68.0% (vs 非付费21.2%) |
| 加联盟玩家7日留存 | 61.6% (vs 未加联盟13.4%) |
| 钻石流失风险阈值 | 566 (Logistic回归, 7天窗口) |
| Cox C-index | 0.9479（⚠️含信息泄漏，修复后会下降） |
| 最优聚类数 | K=6 (轮廓系数0.490) |
| 保守方案期望30日营收 | 22,424元 (10K玩家) |
| 保守方案P(留存≥10%) | 100% |

## 技能使用

| 技能 | 用途 |
|------|------|
| `math-modeling` | 三阶段建模流程（分析→代码→论文） |
| `nature-skills:nature-figure` | Nature期刊图表规范（配色/字体/导出/面板设计） |
| `kimi-pdf` | LaTeX编译与PDF处理 |
| `kimi-xlsx` | Excel数据表头读取 |

## Git工作流

- 不提交大CSV文件（pickdata1/2, other_pickdata1/2, 约5GB）
- 不提交`.claude/`目录
- 提交前检查`git status`，排除构建产物
- Commit message格式: `主题：简明描述`，正文用`$(cat <<'EOF' ... EOF)` HEREDOC
