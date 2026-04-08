"""
===========================================================
可视化模块 (Visualization / Plot Results Module)
===========================================================

功能说明：
    复现论文中的所有对比图表：

    Figure 3: 词汇相似度对比图
        - Baseline LLMs vs RAG-Augmented LLMs
        - 指标: BLEU, ROUGE-1, ROUGE-2, ROUGE-L
        - 类型: 分组柱状图 (grouped bar chart)

    Figure 4: 语义相似度对比图
        - Baseline LLMs vs RAG-Augmented LLMs
        - 指标: BERTScore Precision, Recall, F1, NLI
        - 类型: 分组柱状图

    Table 3: RAG 改进百分比表
        - 计算每个模型的改进幅度
        - 输出为图表或表格

    Figure 5: 跨模型对比图
        - RAG-Augmented 小模型 vs Baseline 大模型
        - 用于分析 trade-off

图表要求（论文规范）：
    - 清晰标题
    - 坐标轴标签
    - 图例（legend）
    - 使用高 DPI（300）保存为 PNG
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

# 设置全局字体和样式
# 使用 seaborn 的白色网格样式，更学术化
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams["figure.dpi"] = 100
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["figure.figsize"] = (12, 6)


# 定义配色方案（论文风格）
# 使用对比鲜明的颜色区分 Baseline 和 RAG
COLORS = {
    "baseline": "#4A90D9",  # 蓝色系：Baseline
    "rag": "#E74C3C",  # 红色系：RAG-Augmented
    "improvement": "#2ECC71",  # 绿色系：改进
}

# 模型显示名称映射
MODEL_DISPLAY_NAMES = {
    "tinyllama": "TinyLlama\n1.1B",
    "mistral": "Mistral\n7B",
    "llama3.1": "Llama 3.1\n8B",
    "llama1-13b": "Llama 1\n13B",
}

# 指标显示名称映射
METRIC_DISPLAY_NAMES = {
    "bleu": "BLEU",
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rougeL": "ROUGE-L",
    "bert_precision": "BERT\nPrecision",
    "bert_recall": "BERT\nRecall",
    "bert_f1": "BERT\nF1",
    "nli": "NLI",
}


def _prepare_comparison_data(results: dict) -> pd.DataFrame:
    """
    将原始结果转换为适合绘图的 DataFrame。

    参数:
        results (dict): 实验结果字典，格式:
            {
                "model_key": {
                    "baseline": {"metric_name": {"mean": float, "ci_lower": float, "ci_upper": float}},
                    "rag": {"metric_name": {"mean": float, "ci_lower": float, "ci_upper": float}}
                }
            }

    返回:
        pd.DataFrame: 包含 model, mode, metric, mean, ci_lower, ci_upper 列
    """
    rows = []
    for model_key, modes in results.items():
        for mode, metrics in modes.items():
            for metric_name, stats in metrics.items():
                rows.append(
                    {
                        "model": model_key,
                        "model_display": MODEL_DISPLAY_NAMES.get(model_key, model_key),
                        "mode": mode,
                        "mode_display": "Baseline"
                        if mode == "baseline"
                        else "RAG-Augmented",
                        "metric": metric_name,
                        "metric_display": METRIC_DISPLAY_NAMES.get(
                            metric_name, metric_name
                        ),
                        "mean": stats["mean"],
                        "ci_lower": stats.get("ci_lower", stats["mean"]),
                        "ci_upper": stats.get("ci_upper", stats["mean"]),
                        "margin_of_error": stats.get("margin_of_error", 0),
                    }
                )

    return pd.DataFrame(rows)


def plot_lexical_comparison(
    results: dict,
    save_path: str = "figures/lexical_comparison.png",
    title: str = "Lexical Similarity Score Comparison:\nBaseline LLMs vs RAG-Augmented LLMs",
):
    """
    绘制词汇相似度对比图（复现论文 Figure 3）。

    图表类型: 分组柱状图 (Grouped Bar Chart)
    - X 轴: 4 个模型
    - Y 轴: 指标得分
    - 颜色: 蓝色 = Baseline, 红色 = RAG-Augmented
    - 子图: 每个指标 (BLEU, ROUGE-1, ROUGE-2, ROUGE-L) 一个子图
    - 误差线: 90% 置信区间

    参数:
        results (dict): 实验结果
        save_path (str): 图表保存路径
        title (str): 图表总标题
    """
    # 准备数据
    df = _prepare_comparison_data(results)

    # 词汇指标
    lexical_metrics = ["bleu", "rouge1", "rouge2", "rougeL"]
    df_lexical = df[df["metric"].isin(lexical_metrics)]

    # 获取模型列表（保持顺序）
    model_keys = list(results.keys())

    # 创建 2x2 子图布局
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    for idx, metric in enumerate(lexical_metrics):
        ax = axes[idx // 2][idx % 2]
        metric_data = df_lexical[df_lexical["metric"] == metric]

        # 准备柱状图数据
        x = np.arange(len(model_keys))
        width = 0.35  # 柱宽

        baseline_means = []
        rag_means = []
        baseline_errors = []
        rag_errors = []

        for model in model_keys:
            # Baseline 数据
            bl = metric_data[
                (metric_data["model"] == model) & (metric_data["mode"] == "baseline")
            ]
            if not bl.empty:
                baseline_means.append(bl["mean"].values[0])
                baseline_errors.append(bl["margin_of_error"].values[0])
            else:
                baseline_means.append(0)
                baseline_errors.append(0)

            # RAG 数据
            rg = metric_data[
                (metric_data["model"] == model) & (metric_data["mode"] == "rag")
            ]
            if not rg.empty:
                rag_means.append(rg["mean"].values[0])
                rag_errors.append(rg["margin_of_error"].values[0])
            else:
                rag_means.append(0)
                rag_errors.append(0)

        # 绘制柱状图
        bars1 = ax.bar(
            x - width / 2,
            baseline_means,
            width,
            yerr=baseline_errors,
            label="Baseline",
            color=COLORS["baseline"],
            alpha=0.85,
            capsize=3,
            edgecolor="white",
            linewidth=0.5,
        )
        bars2 = ax.bar(
            x + width / 2,
            rag_means,
            width,
            yerr=rag_errors,
            label="RAG-Augmented",
            color=COLORS["rag"],
            alpha=0.85,
            capsize=3,
            edgecolor="white",
            linewidth=0.5,
        )

        # 在柱状图上方显示数值
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

        # 设置坐标轴
        display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
        ax.set_title(display_name, fontsize=14, fontweight="bold")
        ax.set_ylabel("Score", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_DISPLAY_NAMES.get(m, m) for m in model_keys], fontsize=9
        )
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()

    # 保存图表
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"📊 词汇相似度对比图已保存: {save_path}")


def plot_semantic_comparison(
    results: dict,
    save_path: str = "figures/semantic_comparison.png",
    title: str = "Semantic Similarity Score Comparison:\nBaseline LLMs vs RAG-Augmented LLMs",
):
    """
    绘制语义相似度对比图（复现论文 Figure 4）。

    图表类型: 分组柱状图
    - 指标: BERTScore Precision, Recall, F1
    - 其他设置同词汇对比图

    参数:
        results (dict): 实验结果
        save_path (str): 图表保存路径
        title (str): 图表总标题
    """
    df = _prepare_comparison_data(results)

    # 语义指标
    semantic_metrics = ["bert_precision", "bert_recall", "bert_f1", "nli"]
    df_semantic = df[df["metric"].isin(semantic_metrics)]

    model_keys = list(results.keys())

    # 创建 2x2 子图布局
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    for idx, metric in enumerate(semantic_metrics):
        ax = axes[idx // 2][idx % 2]
        metric_data = df_semantic[df_semantic["metric"] == metric]

        x = np.arange(len(model_keys))
        width = 0.35

        baseline_means = []
        rag_means = []
        baseline_errors = []
        rag_errors = []

        for model in model_keys:
            bl = metric_data[
                (metric_data["model"] == model) & (metric_data["mode"] == "baseline")
            ]
            if not bl.empty:
                baseline_means.append(bl["mean"].values[0])
                baseline_errors.append(bl["margin_of_error"].values[0])
            else:
                baseline_means.append(0)
                baseline_errors.append(0)

            rg = metric_data[
                (metric_data["model"] == model) & (metric_data["mode"] == "rag")
            ]
            if not rg.empty:
                rag_means.append(rg["mean"].values[0])
                rag_errors.append(rg["margin_of_error"].values[0])
            else:
                rag_means.append(0)
                rag_errors.append(0)

        bars1 = ax.bar(
            x - width / 2,
            baseline_means,
            width,
            yerr=baseline_errors,
            label="Baseline",
            color=COLORS["baseline"],
            alpha=0.85,
            capsize=3,
            edgecolor="white",
            linewidth=0.5,
        )
        bars2 = ax.bar(
            x + width / 2,
            rag_means,
            width,
            yerr=rag_errors,
            label="RAG-Augmented",
            color=COLORS["rag"],
            alpha=0.85,
            capsize=3,
            edgecolor="white",
            linewidth=0.5,
        )

        # 数值标注
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

        display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
        ax.set_title(display_name, fontsize=14, fontweight="bold")
        ax.set_ylabel("Score", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_DISPLAY_NAMES.get(m, m) for m in model_keys], fontsize=9
        )
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"📊 语义相似度对比图已保存: {save_path}")


def plot_improvement_heatmap(
    results: dict,
    save_path: str = "figures/improvement_heatmap.png",
    title: str = "RAG-Augmented Improvement Over Baseline LLMs (%)",
):
    """
    绘制 RAG 改进百分比热力图（复现论文 Table 3 的可视化版本）。

    显示每个模型在每个指标上的百分比改进幅度。
    使用热力图直观展示改进的大小。

    参数:
        results (dict): 实验结果
        save_path (str): 保存路径
        title (str): 标题
    """

    # 内联改进计算函数，避免跨模块导入问题
    def _compute_improvement(baseline_mean, rag_mean):
        if baseline_mean == 0:
            return float("inf") if rag_mean > 0 else 0.0
        return ((rag_mean - baseline_mean) / baseline_mean) * 100

    model_keys = list(results.keys())
    all_metrics = [
        "bleu",
        "rouge1",
        "rouge2",
        "rougeL",
        "bert_precision",
        "bert_recall",
        "bert_f1",
        "nli",
    ]

    # 构建改进百分比矩阵
    improvement_data = []
    for model in model_keys:
        row = {}
        for metric in all_metrics:
            baseline_mean = results[model]["baseline"][metric]["mean"]
            rag_mean = results[model]["rag"][metric]["mean"]
            row[METRIC_DISPLAY_NAMES.get(metric, metric)] = _compute_improvement(
                baseline_mean, rag_mean
            )
        improvement_data.append(row)

    # 创建 DataFrame
    df_improvement = pd.DataFrame(
        improvement_data,
        index=[MODEL_DISPLAY_NAMES.get(m, m).replace("\n", " ") for m in model_keys],
    )

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12, 5))

    sns.heatmap(
        df_improvement,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",  # 红黄绿渐变色，绿色表示更大改进
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Improvement (%)"},
        annot_kws={"fontsize": 10, "fontweight": "bold"},
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_xlabel("Metric", fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"📊 改进百分比热力图已保存: {save_path}")


def plot_cross_model_comparison(
    results: dict,
    comparisons: list = None,
    save_path: str = "figures/cross_model_comparison.png",
    title: str = "Cross-Model Comparison:\nRAG-Augmented Small LLMs vs Baseline Large LLMs",
):
    """
    绘制跨模型对比图（复现论文 Figure 5）。

    论文中的三个比较案例：
    1. Baseline Llama 3.1 8B vs RAG-Augmented Mistral 7B
    2. Baseline Llama 1 13B vs RAG-Augmented Mistral 7B
    3. Baseline Llama 1 13B vs RAG-Augmented Llama 3.1 8B

    目的：探索使用 RAG 增强小模型是否能替代更大的 Baseline 模型

    参数:
        results (dict): 实验结果
        comparisons (list): 对比列表，每个元素为元组
            (baseline_model_key, rag_model_key, comparison_label)
        save_path (str): 保存路径
        title (str): 标题
    """
    if comparisons is None:
        # 默认的论文比较案例
        comparisons = [
            ("llama3.1", "mistral", "Baseline Llama 3.1 8B\nvs RAG Mistral 7B"),
            ("llama1-13b", "mistral", "Baseline Llama 1 13B\nvs RAG Mistral 7B"),
            ("llama1-13b", "llama3.1", "Baseline Llama 1 13B\nvs RAG Llama 3.1 8B"),
        ]

    all_metrics = [
        "bleu",
        "rouge1",
        "rouge2",
        "rougeL",
        "bert_precision",
        "bert_recall",
        "bert_f1",
        "nli",
    ]

    # 创建子图
    n_comparisons = len(comparisons)
    fig, axes = plt.subplots(1, n_comparisons, figsize=(7 * n_comparisons, 6))
    if n_comparisons == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.05)

    for c_idx, (baseline_key, rag_key, label) in enumerate(comparisons):
        ax = axes[c_idx]

        # 检查数据是否存在
        if baseline_key not in results or rag_key not in results:
            ax.text(
                0.5,
                0.5,
                f"Data not available\nfor {label}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        x = np.arange(len(all_metrics))
        width = 0.35

        baseline_means = [
            results[baseline_key]["baseline"][m]["mean"] for m in all_metrics
        ]
        rag_means = [results[rag_key]["rag"][m]["mean"] for m in all_metrics]

        baseline_errors = [
            results[baseline_key]["baseline"][m].get("margin_of_error", 0)
            for m in all_metrics
        ]
        rag_errors = [
            results[rag_key]["rag"][m].get("margin_of_error", 0) for m in all_metrics
        ]

        ax.bar(
            x - width / 2,
            baseline_means,
            width,
            yerr=baseline_errors,
            label=f"Baseline {MODEL_DISPLAY_NAMES.get(baseline_key, baseline_key).replace(chr(10), ' ')}",
            color=COLORS["baseline"],
            alpha=0.85,
            capsize=2,
        )
        ax.bar(
            x + width / 2,
            rag_means,
            width,
            yerr=rag_errors,
            label=f"RAG {MODEL_DISPLAY_NAMES.get(rag_key, rag_key).replace(chr(10), ' ')}",
            color=COLORS["rag"],
            alpha=0.85,
            capsize=2,
        )

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Score", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [METRIC_DISPLAY_NAMES.get(m, m) for m in all_metrics],
            fontsize=7,
            rotation=45,
            ha="right",
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"📊 跨模型对比图已保存: {save_path}")


def plot_all_metrics_overview(
    results: dict,
    save_path: str = "figures/all_metrics_overview.png",
    title: str = "Complete Metrics Overview: All Models",
):
    """
    绘制所有指标的综合概览图。

    一张图展示所有 7 个指标在所有模型上的表现。
    适合快速对比总体趋势。

    参数:
        results (dict): 实验结果
        save_path (str): 保存路径
        title (str): 标题
    """
    all_metrics = [
        "bleu",
        "rouge1",
        "rouge2",
        "rougeL",
        "bert_precision",
        "bert_recall",
        "bert_f1",
        "nli",
    ]
    model_keys = list(results.keys())

    fig, ax = plt.subplots(figsize=(16, 7))

    x = np.arange(len(all_metrics))
    n_models = len(model_keys)
    total_width = 0.75
    single_width = total_width / (n_models * 2)  # baseline + rag

    # 为每个模型的 baseline 和 rag 分别绘制柱状图
    colors_baseline = ["#3498DB", "#2ECC71", "#E67E22", "#9B59B6"]  # 冷色调
    colors_rag = ["#2980B9", "#27AE60", "#D35400", "#8E44AD"]  # 深色调

    for m_idx, model_key in enumerate(model_keys):
        offset_base = -total_width / 2 + m_idx * 2 * single_width + single_width / 2
        offset_rag = offset_base + single_width

        baseline_means = [
            results[model_key]["baseline"][metric]["mean"] for metric in all_metrics
        ]
        rag_means = [
            results[model_key]["rag"][metric]["mean"] for metric in all_metrics
        ]

        model_name = MODEL_DISPLAY_NAMES.get(model_key, model_key).replace("\n", " ")

        ax.bar(
            x + offset_base,
            baseline_means,
            single_width,
            label=f"{model_name} (Baseline)",
            color=colors_baseline[m_idx % len(colors_baseline)],
            alpha=0.7,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.bar(
            x + offset_rag,
            rag_means,
            single_width,
            label=f"{model_name} (RAG)",
            color=colors_rag[m_idx % len(colors_rag)],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.3,
            hatch="//",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METRIC_DISPLAY_NAMES.get(m, m) for m in all_metrics], fontsize=10
    )
    ax.legend(fontsize=7, loc="upper left", ncol=2, bbox_to_anchor=(0, 1))
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"📊 综合概览图已保存: {save_path}")


def generate_all_plots(results: dict, figure_dir: str = "figures"):
    """
    生成论文中所有图表。

    一次性生成所有可视化输出：
    1. 词汇相似度对比图 (Figure 3)
    2. 语义相似度对比图 (Figure 4)
    3. 改进百分比热力图 (Table 3 可视化)
    4. 跨模型对比图 (Figure 5)
    5. 综合概览图 (额外)

    参数:
        results (dict): 完整的实验结果
        figure_dir (str): 图表保存目录
    """
    print("\n" + "=" * 60)
    print("📊 生成所有图表")
    print("=" * 60)

    os.makedirs(figure_dir, exist_ok=True)

    # Figure 3: 词汇相似度对比
    print("\n--- Figure 3: 词汇相似度对比 ---")
    plot_lexical_comparison(
        results, save_path=os.path.join(figure_dir, "fig3_lexical_comparison.png")
    )

    # Figure 4: 语义相似度对比
    print("\n--- Figure 4: 语义相似度对比 ---")
    plot_semantic_comparison(
        results, save_path=os.path.join(figure_dir, "fig4_semantic_comparison.png")
    )

    # Table 3: 改进百分比热力图
    print("\n--- Table 3: 改进百分比热力图 ---")
    plot_improvement_heatmap(
        results, save_path=os.path.join(figure_dir, "table3_improvement_heatmap.png")
    )

    # Figure 5: 跨模型对比
    print("\n--- Figure 5: 跨模型对比 ---")
    plot_cross_model_comparison(
        results, save_path=os.path.join(figure_dir, "fig5_cross_model_comparison.png")
    )

    # 额外: 综合概览
    print("\n--- 综合概览图 ---")
    plot_all_metrics_overview(
        results, save_path=os.path.join(figure_dir, "all_metrics_overview.png")
    )

    print("\n✅ 所有图表生成完成!")
    print(f"   保存目录: {figure_dir}/")
    print("=" * 60)


# =============================================
# 单元测试 / 功能演示
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("可视化模块功能测试（使用模拟数据）")
    print("=" * 60)

    # 创建模拟数据（近似论文中的结果）
    mock_results = {
        "tinyllama": {
            "baseline": {
                "bleu": {
                    "mean": 0.025,
                    "ci_lower": 0.020,
                    "ci_upper": 0.030,
                    "margin_of_error": 0.005,
                },
                "rouge1": {
                    "mean": 0.15,
                    "ci_lower": 0.13,
                    "ci_upper": 0.17,
                    "margin_of_error": 0.02,
                },
                "rouge2": {
                    "mean": 0.06,
                    "ci_lower": 0.05,
                    "ci_upper": 0.07,
                    "margin_of_error": 0.01,
                },
                "rougeL": {
                    "mean": 0.12,
                    "ci_lower": 0.10,
                    "ci_upper": 0.14,
                    "margin_of_error": 0.02,
                },
                "bert_precision": {
                    "mean": 0.83,
                    "ci_lower": 0.82,
                    "ci_upper": 0.84,
                    "margin_of_error": 0.01,
                },
                "bert_recall": {
                    "mean": 0.86,
                    "ci_lower": 0.85,
                    "ci_upper": 0.87,
                    "margin_of_error": 0.01,
                },
                "bert_f1": {
                    "mean": 0.845,
                    "ci_lower": 0.84,
                    "ci_upper": 0.85,
                    "margin_of_error": 0.005,
                },
                "nli": {
                    "mean": 0.72,
                    "ci_lower": 0.68,
                    "ci_upper": 0.76,
                    "margin_of_error": 0.04,
                },
            },
            "rag": {
                "bleu": {
                    "mean": 0.032,
                    "ci_lower": 0.028,
                    "ci_upper": 0.036,
                    "margin_of_error": 0.004,
                },
                "rouge1": {
                    "mean": 0.20,
                    "ci_lower": 0.18,
                    "ci_upper": 0.22,
                    "margin_of_error": 0.02,
                },
                "rouge2": {
                    "mean": 0.085,
                    "ci_lower": 0.07,
                    "ci_upper": 0.10,
                    "margin_of_error": 0.015,
                },
                "rougeL": {
                    "mean": 0.157,
                    "ci_lower": 0.14,
                    "ci_upper": 0.17,
                    "margin_of_error": 0.015,
                },
                "bert_precision": {
                    "mean": 0.845,
                    "ci_lower": 0.84,
                    "ci_upper": 0.85,
                    "margin_of_error": 0.005,
                },
                "bert_recall": {
                    "mean": 0.88,
                    "ci_lower": 0.87,
                    "ci_upper": 0.89,
                    "margin_of_error": 0.01,
                },
                "bert_f1": {
                    "mean": 0.86,
                    "ci_lower": 0.85,
                    "ci_upper": 0.87,
                    "margin_of_error": 0.01,
                },
                "nli": {
                    "mean": 0.78,
                    "ci_lower": 0.74,
                    "ci_upper": 0.82,
                    "margin_of_error": 0.04,
                },
            },
        },
        "mistral": {
            "baseline": {
                "bleu": {
                    "mean": 0.04,
                    "ci_lower": 0.035,
                    "ci_upper": 0.045,
                    "margin_of_error": 0.005,
                },
                "rouge1": {
                    "mean": 0.22,
                    "ci_lower": 0.20,
                    "ci_upper": 0.24,
                    "margin_of_error": 0.02,
                },
                "rouge2": {
                    "mean": 0.095,
                    "ci_lower": 0.08,
                    "ci_upper": 0.11,
                    "margin_of_error": 0.015,
                },
                "rougeL": {
                    "mean": 0.1775,
                    "ci_lower": 0.16,
                    "ci_upper": 0.195,
                    "margin_of_error": 0.0175,
                },
                "bert_precision": {
                    "mean": 0.84,
                    "ci_lower": 0.83,
                    "ci_upper": 0.85,
                    "margin_of_error": 0.01,
                },
                "bert_recall": {
                    "mean": 0.87,
                    "ci_lower": 0.86,
                    "ci_upper": 0.88,
                    "margin_of_error": 0.01,
                },
                "bert_f1": {
                    "mean": 0.855,
                    "ci_lower": 0.85,
                    "ci_upper": 0.86,
                    "margin_of_error": 0.005,
                },
                "nli": {
                    "mean": 0.75,
                    "ci_lower": 0.71,
                    "ci_upper": 0.79,
                    "margin_of_error": 0.04,
                },
            },
            "rag": {
                "bleu": {
                    "mean": 0.064,
                    "ci_lower": 0.058,
                    "ci_upper": 0.070,
                    "margin_of_error": 0.006,
                },
                "rouge1": {
                    "mean": 0.28,
                    "ci_lower": 0.26,
                    "ci_upper": 0.30,
                    "margin_of_error": 0.02,
                },
                "rouge2": {
                    "mean": 0.122,
                    "ci_lower": 0.11,
                    "ci_upper": 0.134,
                    "margin_of_error": 0.012,
                },
                "rougeL": {
                    "mean": 0.227,
                    "ci_lower": 0.21,
                    "ci_upper": 0.244,
                    "margin_of_error": 0.017,
                },
                "bert_precision": {
                    "mean": 0.858,
                    "ci_lower": 0.85,
                    "ci_upper": 0.866,
                    "margin_of_error": 0.008,
                },
                "bert_recall": {
                    "mean": 0.895,
                    "ci_lower": 0.89,
                    "ci_upper": 0.90,
                    "margin_of_error": 0.005,
                },
                "bert_f1": {
                    "mean": 0.876,
                    "ci_lower": 0.87,
                    "ci_upper": 0.882,
                    "margin_of_error": 0.006,
                },
                "nli": {
                    "mean": 0.82,
                    "ci_lower": 0.78,
                    "ci_upper": 0.86,
                    "margin_of_error": 0.04,
                },
            },
        },
        "llama3.1": {
            "baseline": {
                "bleu": {
                    "mean": 0.008,
                    "ci_lower": 0.005,
                    "ci_upper": 0.011,
                    "margin_of_error": 0.003,
                },
                "rouge1": {
                    "mean": 0.05,
                    "ci_lower": 0.04,
                    "ci_upper": 0.06,
                    "margin_of_error": 0.01,
                },
                "rouge2": {
                    "mean": 0.023,
                    "ci_lower": 0.018,
                    "ci_upper": 0.028,
                    "margin_of_error": 0.005,
                },
                "rougeL": {
                    "mean": 0.044,
                    "ci_lower": 0.035,
                    "ci_upper": 0.053,
                    "margin_of_error": 0.009,
                },
                "bert_precision": {
                    "mean": 0.77,
                    "ci_lower": 0.76,
                    "ci_upper": 0.78,
                    "margin_of_error": 0.01,
                },
                "bert_recall": {
                    "mean": 0.87,
                    "ci_lower": 0.86,
                    "ci_upper": 0.88,
                    "margin_of_error": 0.01,
                },
                "bert_f1": {
                    "mean": 0.82,
                    "ci_lower": 0.81,
                    "ci_upper": 0.83,
                    "margin_of_error": 0.01,
                },
                "nli": {
                    "mean": 0.65,
                    "ci_lower": 0.60,
                    "ci_upper": 0.70,
                    "margin_of_error": 0.05,
                },
            },
            "rag": {
                "bleu": {
                    "mean": 0.026,
                    "ci_lower": 0.022,
                    "ci_upper": 0.030,
                    "margin_of_error": 0.004,
                },
                "rouge1": {
                    "mean": 0.14,
                    "ci_lower": 0.12,
                    "ci_upper": 0.16,
                    "margin_of_error": 0.02,
                },
                "rouge2": {
                    "mean": 0.058,
                    "ci_lower": 0.048,
                    "ci_upper": 0.068,
                    "margin_of_error": 0.01,
                },
                "rougeL": {
                    "mean": 0.108,
                    "ci_lower": 0.09,
                    "ci_upper": 0.126,
                    "margin_of_error": 0.018,
                },
                "bert_precision": {
                    "mean": 0.81,
                    "ci_lower": 0.80,
                    "ci_upper": 0.82,
                    "margin_of_error": 0.01,
                },
                "bert_recall": {
                    "mean": 0.886,
                    "ci_lower": 0.88,
                    "ci_upper": 0.892,
                    "margin_of_error": 0.006,
                },
                "bert_f1": {
                    "mean": 0.83,
                    "ci_lower": 0.82,
                    "ci_upper": 0.84,
                    "margin_of_error": 0.01,
                },
                "nli": {
                    "mean": 0.74,
                    "ci_lower": 0.69,
                    "ci_upper": 0.79,
                    "margin_of_error": 0.05,
                },
            },
        },
        "llama1-13b": {
            "baseline": {
                "bleu": {
                    "mean": 0.035,
                    "ci_lower": 0.030,
                    "ci_upper": 0.040,
                    "margin_of_error": 0.005,
                },
                "rouge1": {
                    "mean": 0.20,
                    "ci_lower": 0.18,
                    "ci_upper": 0.22,
                    "margin_of_error": 0.02,
                },
                "rouge2": {
                    "mean": 0.088,
                    "ci_lower": 0.075,
                    "ci_upper": 0.101,
                    "margin_of_error": 0.013,
                },
                "rougeL": {
                    "mean": 0.165,
                    "ci_lower": 0.15,
                    "ci_upper": 0.18,
                    "margin_of_error": 0.015,
                },
                "bert_precision": {
                    "mean": 0.83,
                    "ci_lower": 0.82,
                    "ci_upper": 0.84,
                    "margin_of_error": 0.01,
                },
                "bert_recall": {
                    "mean": 0.875,
                    "ci_lower": 0.87,
                    "ci_upper": 0.88,
                    "margin_of_error": 0.005,
                },
                "bert_f1": {
                    "mean": 0.852,
                    "ci_lower": 0.85,
                    "ci_upper": 0.854,
                    "margin_of_error": 0.002,
                },
                "nli": {
                    "mean": 0.73,
                    "ci_lower": 0.68,
                    "ci_upper": 0.78,
                    "margin_of_error": 0.05,
                },
            },
            "rag": {
                "bleu": {
                    "mean": 0.063,
                    "ci_lower": 0.057,
                    "ci_upper": 0.069,
                    "margin_of_error": 0.006,
                },
                "rouge1": {
                    "mean": 0.26,
                    "ci_lower": 0.24,
                    "ci_upper": 0.28,
                    "margin_of_error": 0.02,
                },
                "rouge2": {
                    "mean": 0.115,
                    "ci_lower": 0.10,
                    "ci_upper": 0.13,
                    "margin_of_error": 0.015,
                },
                "rougeL": {
                    "mean": 0.195,
                    "ci_lower": 0.18,
                    "ci_upper": 0.21,
                    "margin_of_error": 0.015,
                },
                "bert_precision": {
                    "mean": 0.845,
                    "ci_lower": 0.84,
                    "ci_upper": 0.85,
                    "margin_of_error": 0.005,
                },
                "bert_recall": {
                    "mean": 0.881,
                    "ci_lower": 0.877,
                    "ci_upper": 0.885,
                    "margin_of_error": 0.004,
                },
                "bert_f1": {
                    "mean": 0.856,
                    "ci_lower": 0.853,
                    "ci_upper": 0.859,
                    "margin_of_error": 0.003,
                },
                "nli": {
                    "mean": 0.80,
                    "ci_lower": 0.76,
                    "ci_upper": 0.84,
                    "margin_of_error": 0.04,
                },
            },
        },
    }

    # 生成所有图表
    generate_all_plots(mock_results, figure_dir="figures")
