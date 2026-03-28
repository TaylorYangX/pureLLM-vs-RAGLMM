"""
===========================================================
步骤 4: 评估实验结果 (Evaluate Results)
===========================================================

功能说明：
    加载 step3 生成的原始输出 CSV，与 Ground Truth 对比，
    计算所有 7 个评估指标并输出统计结果。

    前置条件：
    - results/raw_outputs_*.csv 已存在 (step3)
    - data/ground_truth.json 已存在 (step2)

    输出：
    - results/evaluation_stats_YYYYMMDD_HHMMSS.json
    - results/improvement_summary_YYYYMMDD_HHMMSS.csv

使用方式：
    python step4_evaluate.py
    python step4_evaluate.py --input results/raw_outputs_20260328.csv
"""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.model_config import EXPERIMENT_CONFIG
from data.dataset_loader import load_ground_truth
from evaluation.metrics import (
    compute_bleu,
    compute_rouge,
    compute_bertscore,
    compute_confidence_interval,
    compute_improvement,
)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="步骤 4: 评估实验结果"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="原始输出 CSV 文件路径（默认使用最新的）"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="评估结果输出目录"
    )
    parser.add_argument(
        "--confidence", type=float, default=None,
        help="置信区间水平（默认 0.90）"
    )
    return parser.parse_args()


def find_latest_raw_output(output_dir: str) -> str:
    """
    在结果目录中查找最新的原始输出文件。

    参数:
        output_dir (str): 结果目录

    返回:
        str: 最新文件的完整路径
    """
    files = [
        f for f in os.listdir(output_dir)
        if f.startswith("raw_outputs_") and f.endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError(
            f"❌ 在 {output_dir}/ 中未找到 raw_outputs_*.csv\n"
            f"请先运行 step3_run_experiments.py"
        )
    latest = sorted(files)[-1]
    return os.path.join(output_dir, latest)


def evaluate_from_csv(
    csv_path: str,
    ground_truth: list,
    confidence_level: float = 0.90
) -> dict:
    """
    从 CSV 文件加载原始输出并计算所有指标。

    参数:
        csv_path (str): 原始输出 CSV 路径
        ground_truth (list): Ground Truth 数据
        confidence_level (float): 置信水平

    返回:
        dict: 结构化评估结果
    """
    print(f"📊 加载原始输出: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"   共 {len(df)} 条记录")

    # 构建 Ground Truth 映射
    gt_map = {item["query_id"]: item["ground_truth"] for item in ground_truth}

    # 指标名称列表
    metric_names = ["bleu", "rouge1", "rouge2", "rougeL",
                    "bert_precision", "bert_recall", "bert_f1"]

    # 按模型和模式分组
    models = df["model"].unique()
    modes = df["mode"].unique()

    final_results = {}

    for model in models:
        print(f"\n--- 评估模型: {model} ---")
        final_results[model] = {}

        for mode in modes:
            # 筛选当前模型和模式的成功记录
            subset = df[
                (df["model"] == model) &
                (df["mode"] == mode) &
                (df["success"] == True)
            ]
            print(f"  [{mode}] {len(subset)} 条成功记录")

            if subset.empty:
                final_results[model][mode] = {
                    m: {"mean": 0.0, "std": 0.0, "ci_lower": 0.0,
                        "ci_upper": 0.0, "margin_of_error": 0.0, "n": 0}
                    for m in metric_names
                }
                continue

            # 配对参考文本和候选文本
            references = []
            candidates = []
            for _, row in subset.iterrows():
                qid = row.get("query_id")
                gt = gt_map.get(qid, "")
                resp = str(row.get("response", "")).strip()
                if gt and resp:
                    references.append(gt)
                    candidates.append(resp)

            if not references:
                print(f"  ⚠️  没有有效的参考/候选对")
                final_results[model][mode] = {
                    m: {"mean": 0.0, "std": 0.0, "ci_lower": 0.0,
                        "ci_upper": 0.0, "margin_of_error": 0.0, "n": 0}
                    for m in metric_names
                }
                continue

            # 计算词汇指标
            bleu_scores = []
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []

            for ref, cand in zip(references, candidates):
                bleu_scores.append(compute_bleu(ref, cand))
                rouge = compute_rouge(ref, cand)
                rouge1_scores.append(rouge["rouge1"])
                rouge2_scores.append(rouge["rouge2"])
                rougeL_scores.append(rouge["rougeL"])

            # 计算语义指标（批量）
            print(f"  计算 BERTScore ({len(references)} 个样本)...")
            bert_results = compute_bertscore(references, candidates)

            # 汇总
            all_scores = {
                "bleu": bleu_scores,
                "rouge1": rouge1_scores,
                "rouge2": rouge2_scores,
                "rougeL": rougeL_scores,
                "bert_precision": bert_results["precision"],
                "bert_recall": bert_results["recall"],
                "bert_f1": bert_results["f1"],
            }

            # 计算统计量
            mode_stats = {}
            for metric_name, scores in all_scores.items():
                ci = compute_confidence_interval(scores, confidence_level)
                mode_stats[metric_name] = ci

            final_results[model][mode] = mode_stats

            # 打印摘要
            for m in metric_names:
                s = mode_stats[m]
                print(f"    {m}: mean={s['mean']:.4f} [{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]")

    return final_results


def save_evaluation_results(results: dict, output_dir: str):
    """
    保存评估结果。

    参数:
        results (dict): 评估结果
        output_dir (str): 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 保存统计结果 (JSON)
    stats_path = os.path.join(output_dir, f"evaluation_stats_{timestamp}.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 统计结果: {stats_path}")

    # 2. 保存改进汇总 (CSV)
    metric_names = ["bleu", "rouge1", "rouge2", "rougeL",
                    "bert_precision", "bert_recall", "bert_f1"]
    rows = []
    for model_key in results.keys():
        row = {"model": model_key}
        for m in metric_names:
            bl = results[model_key].get("baseline", {}).get(m, {}).get("mean", 0)
            rg = results[model_key].get("rag", {}).get(m, {}).get("mean", 0)
            row[f"{m}_baseline"] = bl
            row[f"{m}_rag"] = rg
            row[f"{m}_improvement_%"] = compute_improvement(bl, rg)
        rows.append(row)

    if rows:
        imp_path = os.path.join(output_dir, f"improvement_summary_{timestamp}.csv")
        pd.DataFrame(rows).to_csv(imp_path, index=False, encoding="utf-8")
        print(f"💾 改进汇总: {imp_path}")


def main():
    args = parse_args()

    output_dir = args.output_dir or EXPERIMENT_CONFIG["output_dir"]
    confidence = args.confidence or EXPERIMENT_CONFIG["confidence_level"]
    gt_path = EXPERIMENT_CONFIG["ground_truth_path"]

    print("=" * 60)
    print("📊 步骤 4: 评估实验结果")
    print("=" * 60)

    # 确定输入文件
    if args.input:
        csv_path = args.input
    else:
        csv_path = find_latest_raw_output(output_dir)

    # 加载 Ground Truth
    ground_truth = load_ground_truth(gt_path)

    # 评估
    results = evaluate_from_csv(csv_path, ground_truth, confidence)

    # 保存
    save_evaluation_results(results, output_dir)

    # 打印改进摘要
    print(f"\n📊 RAG 改进幅度摘要:")
    for model_key in results.keys():
        print(f"\n  [{model_key}]")
        for m in ["bleu", "rouge1", "rougeL", "bert_f1"]:
            bl = results[model_key].get("baseline", {}).get(m, {}).get("mean", 0)
            rg = results[model_key].get("rag", {}).get(m, {}).get("mean", 0)
            imp = compute_improvement(bl, rg)
            print(f"    {m}: {bl:.4f} → {rg:.4f} ({imp:+.1f}%)")

    print(f"\n✅ 步骤 4 完成!")
    print(f"   下一步: python step5_visualize.py")


if __name__ == "__main__":
    main()
