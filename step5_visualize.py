"""
===========================================================
步骤 5: 生成可视化图表 (Generate Visualizations)
===========================================================

功能说明：
    加载 step4 生成的评估统计结果，生成论文中的所有图表。

    前置条件：
    - results/evaluation_stats_*.json 已存在 (step4)

    输出图表：
    - figures/fig3_lexical_comparison.png   （词汇相似度对比）
    - figures/fig4_semantic_comparison.png  （语义相似度对比，含 NLI）
    - figures/table3_improvement_heatmap.png（改进百分比热力图，含 NLI）
    - figures/fig5_cross_model_comparison.png（跨模型对比，含 NLI）
    - figures/all_metrics_overview.png      （综合概览，含 NLI）

使用方式：
    python step5_visualize.py
    python step5_visualize.py --input results/evaluation_stats_20260328.json
    python step5_visualize.py --figure-dir figures
"""

import argparse
import json
import os
import sys

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.model_config import EXPERIMENT_CONFIG
from visualization.plot_results import generate_all_plots


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="步骤 5: 生成可视化图表")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="评估统计 JSON 文件路径（默认使用最新的）",
    )
    parser.add_argument(
        "--figure-dir", type=str, default=None, help="图表输出目录（默认 figures）"
    )
    return parser.parse_args()


def find_latest_stats(output_dir: str) -> str:
    """
    查找最新的评估统计文件。

    参数:
        output_dir (str): 结果目录

    返回:
        str: 最新文件的完整路径
    """
    files = [
        f
        for f in os.listdir(output_dir)
        if f.startswith("evaluation_stats_") and f.endswith(".json")
    ]
    if not files:
        raise FileNotFoundError(
            f"❌ 在 {output_dir}/ 中未找到 evaluation_stats_*.json\n"
            f"请先运行 step4_evaluate.py"
        )
    latest = sorted(files)[-1]
    return os.path.join(output_dir, latest)


def main():
    args = parse_args()

    output_dir = EXPERIMENT_CONFIG["output_dir"]
    figure_dir = args.figure_dir or EXPERIMENT_CONFIG["figure_dir"]

    print("=" * 60)
    print("📊 步骤 5: 生成可视化图表")
    print("=" * 60)

    # 确定输入文件
    if args.input:
        stats_path = args.input
    else:
        stats_path = find_latest_stats(output_dir)

    print(f"   输入文件: {stats_path}")
    print(f"   图表目录: {figure_dir}")

    # 加载结果
    with open(stats_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"   包含模型: {list(results.keys())}")

    # 生成所有图表
    generate_all_plots(results, figure_dir)

    print(f"\n✅ 步骤 5 完成!")
    print(f"   图表已保存至: {figure_dir}/")


if __name__ == "__main__":
    main()
