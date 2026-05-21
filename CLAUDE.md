# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

数学建模竞赛仓库，包含两道题目及其附件数据集。目前处于初始阶段，无现有代码。

## 题目结构

### A题：库位分配优化问题

优化原材料在仓库中的库位分配，以最小化生产线订料的取货成本/时间。

数据集（`题目/A题：附件 数据集/`）：
- `表1_原材料库存数据.csv` — 原材料编号、箱子类型(E1-E4)、消耗占比、库存数量
- `表2_生产线订料数据.csv` — 订单号、箱号、箱子类型、原材料号、时间
- `表3_入库材料数据.csv` — 入库批次、原材料编号、箱子类型、入库数量、入库时间
- `库位分配优化问题.docx` — 完整题目描述

### B题：策略类游戏的用户留存与付费策略优化

分析游戏用户行为数据，建模用户留存与付费转化。

数据集（`题目/B题：附件 数据集/`）：
- `pickdata1.csv` (~490MB) / `pickdata2.csv` (~2.8GB) — 主数据：游戏用户资源变更与事件记录，字段包括 account_id, game_id, platform, channel_id, country, user_type, resource_name, change_type, change_num, dt, r_lvl, league_name, change_dollar 等
- `pick_stat1.csv` / `pick_stat2.csv` — 用户聚合统计（account_id, count）
- `other_pickdata1.csv` (~423MB) / `other_pickdata2.csv` (~2.5GB) — 其他游戏数据
- `other_pick_stat1.csv` / `other_pick_stat2.csv` — 其他游戏聚合统计
- `B题 资源变更与用户事件数据表表头.xlsx` — 数据字段说明
- `B题 数据文件说明.docx` — 数据文件说明文档
- `策略类游戏的用户留存与付费策略优化.docx` — 完整题目描述

## 分析工具

推荐使用 Python 数据科学生态：
- `pandas` — 数据处理（注意 B 题数据量大，需分块读取 `chunksize`）
- `numpy`, `scipy` — 数值计算与优化
- `matplotlib`, `seaborn` — 可视化
- `scikit-learn` — 机器学习模型
- Gurobi / CPLEX（如有 license）或 `scipy.optimize` — 优化求解（A题）

B 题数据量极大（单文件 ~2.8GB），建议：
- 使用 `pd.read_csv(chunksize=...)` 分块读取
- 先用 `pick_stat` 聚合统计了解数据分布
- 考虑使用 `dask` 或 `polars` 处理超大数据集

## 数学建模技能

本仓库可使用 `math-modeling` 技能辅助建模论文写作与分析求解。
绘图也可以使用`nature-skills`相关技能
