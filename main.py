"""
===========================================================
主程序入口 (Main Entry Point)
===========================================================

功能说明：
    编排完整的实验流程，复现论文
    "Retrieval-Augmented Generation vs. Baseline LLMs:
     A Multi-Metric Evaluation" 的实验管道。

实验流程概述：
    1. 加载配置和数据
    2. 构建 FAISS 向量索引
    3. 对每个模型执行 Baseline 和 RAG-Augmented 生成
       - 11 个查询 × 11 次迭代 = 121 个输出/模型
    4. 计算所有 7 个评估指标
    5. 统计分析（均值 + 90% 置信区间）
    6. 生成可视化图表
    7. 保存结果

使用方式：
    # 完整实验
    python main.py

    # 快速测试（减少迭代次数）
    python main.py --quick

    # 仅生成图表（使用已有结果）
    python main.py --plot-only

    # 指定模型
    python main.py --models mistral llama3.1

    # 指定文档路径
    python main.py --document data/human_nutrition_2020.pdf
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# 导入项目模块
from config.model_config import (
    get_all_model_keys,
    get_llm_config,
    EXPERIMENT_CONFIG,
    RAG_CONFIG,
    EMBEDDING_CONFIG,
    print_config_summary,
)
from data.dataset_loader import prepare_experiment_data, load_ground_truth
from retrieval.retriever import FAISSRetriever
from models.llm_baseline import BaselineLLM
from models.rag_pipeline import RAGPipeline
from evaluation.metrics import (
    compute_bleu,
    compute_rouge,
    compute_bertscore,
    compute_confidence_interval,
    compute_improvement,
)
from visualization.plot_results import generate_all_plots


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="RAG vs Baseline LLM 多指标评估实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                              # 完整实验
  python main.py --quick                      # 快速测试（2次迭代）
  python main.py --plot-only                  # 仅生成图表
  python main.py --models mistral llama3.1    # 指定模型
  python main.py --iterations 5              # 自定义迭代次数
        """
    )

    parser.add_argument(
        "--quick", action="store_true",
        help="快速测试模式：仅使用 2 次迭代和 3 个查询"
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="仅生成图表（需要已有的结果文件）"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="指定要评估的模型（默认全部）"
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="每个查询的迭代次数（默认 11）"
    )
    parser.add_argument(
        "--queries", type=int, default=None,
        help="使用的查询数量（默认 11）"
    )
    parser.add_argument(
        "--document", type=str, default=None,
        help="知识文档路径（PDF）"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="结果输出目录"
    )
    parser.add_argument(
        "--figure-dir", type=str, default=None,
        help="图表输出目录"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="试运行：验证配置和数据但不执行实验"
    )

    return parser.parse_args()


def run_baseline_experiments(
    model_keys: list,
    queries: list,
    num_iterations: int
) -> dict:
    """
    运行所有 Baseline LLM 实验。

    对每个模型、每个查询执行多次迭代生成。

    参数:
        model_keys (list): 模型键名列表
        queries (list): 查询文本列表
        num_iterations (int): 每个查询的迭代次数

    返回:
        dict: {model_key: [result_dicts]}
    """
    print("\n" + "=" * 60)
    print("🤖 运行 Baseline LLM 实验")
    print("=" * 60)

    all_results = {}
    for model_key in model_keys:
        print(f"\n--- {model_key} ---")
        try:
            llm = BaselineLLM(model_key)
            results = llm.generate_batch(
                queries=queries,
                num_iterations=num_iterations
            )
            all_results[model_key] = results
            # 统计成功率
            success_count = sum(1 for r in results if r["success"])
            print(f"  ✅ 完成: {success_count}/{len(results)} 成功")
        except Exception as e:
            print(f"  ❌ 模型 {model_key} 执行失败: {e}")
            all_results[model_key] = []

    return all_results


def run_rag_experiments(
    model_keys: list,
    queries: list,
    num_iterations: int,
    retriever: FAISSRetriever
) -> dict:
    """
    运行所有 RAG-Augmented LLM 实验。

    参数:
        model_keys (list): 模型键名列表
        queries (list): 查询文本列表
        num_iterations (int): 每个查询的迭代次数
        retriever (FAISSRetriever): 共享的 FAISS 检索器

    返回:
        dict: {model_key: [result_dicts]}
    """
    print("\n" + "=" * 60)
    print("🔗 运行 RAG-Augmented LLM 实验")
    print("=" * 60)

    all_results = {}
    for model_key in model_keys:
        print(f"\n--- RAG-{model_key} ---")
        try:
            rag = RAGPipeline(
                model_key=model_key,
                retriever=retriever  # 共享索引，避免重复构建
            )
            results = rag.generate_batch(
                queries=queries,
                num_iterations=num_iterations
            )
            all_results[model_key] = results
            success_count = sum(1 for r in results if r["success"])
            print(f"  ✅ 完成: {success_count}/{len(results)} 成功")
        except Exception as e:
            print(f"  ❌ RAG-{model_key} 执行失败: {e}")
            all_results[model_key] = []

    return all_results


def evaluate_results(
    baseline_results: dict,
    rag_results: dict,
    ground_truth: list,
    num_iterations: int,
    confidence_level: float = 0.90
) -> dict:
    """
    使用所有 7 个指标评估实验结果。

    流程：
    1. 对每个模型的每个输出计算 BLEU 和 ROUGE
    2. 批量计算 BERTScore（更高效）
    3. 聚合统计：均值 + 90% 置信区间
    4. 计算 RAG 改进百分比

    参数:
        baseline_results (dict): Baseline 实验结果
        rag_results (dict): RAG 实验结果
        ground_truth (list): Ground Truth 数据
        num_iterations (int): 迭代次数
        confidence_level (float): 置信水平

    返回:
        dict: 结构化的评估结果
    """
    print("\n" + "=" * 60)
    print("📊 评估实验结果")
    print("=" * 60)

    # 构建查询 ID 到 ground truth 的映射
    gt_map = {item["query_id"]: item["ground_truth"] for item in ground_truth}

    # 指标名称列表
    metric_names = ["bleu", "rouge1", "rouge2", "rougeL",
                    "bert_precision", "bert_recall", "bert_f1"]

    # 存储最终结构化结果
    final_results = {}

    # 合并所有模式的结果进行统一处理
    all_model_data = {}
    for model_key in set(list(baseline_results.keys()) + list(rag_results.keys())):
        all_model_data[model_key] = {
            "baseline": baseline_results.get(model_key, []),
            "rag": rag_results.get(model_key, []),
        }

    for model_key, modes in all_model_data.items():
        print(f"\n--- 评估模型: {model_key} ---")
        final_results[model_key] = {}

        for mode, results_list in modes.items():
            print(f"  [{mode}] {len(results_list)} 个输出")

            if not results_list:
                # 如果没有结果，填充零值
                final_results[model_key][mode] = {
                    metric: {
                        "mean": 0.0, "std": 0.0,
                        "ci_lower": 0.0, "ci_upper": 0.0,
                        "margin_of_error": 0.0, "n": 0
                    }
                    for metric in metric_names
                }
                continue

            # 只处理成功的结果
            valid_results = [r for r in results_list if r.get("success", False)]
            if not valid_results:
                print(f"  ⚠️  没有成功的输出")
                final_results[model_key][mode] = {
                    metric: {
                        "mean": 0.0, "std": 0.0,
                        "ci_lower": 0.0, "ci_upper": 0.0,
                        "margin_of_error": 0.0, "n": 0
                    }
                    for metric in metric_names
                }
                continue

            # 收集参考文本和候选文本
            references = []
            candidates = []
            for r in valid_results:
                query_id = r.get("query_id", 1)
                gt = gt_map.get(query_id, "")
                if gt and r.get("response", ""):
                    references.append(gt)
                    candidates.append(r["response"])

            if not references:
                print(f"  ⚠️  没有有效的参考/候选文本对")
                final_results[model_key][mode] = {
                    metric: {
                        "mean": 0.0, "std": 0.0,
                        "ci_lower": 0.0, "ci_upper": 0.0,
                        "margin_of_error": 0.0, "n": 0
                    }
                    for metric in metric_names
                }
                continue

            n_samples = len(references)
            print(f"  计算 {n_samples} 个样本的指标...")

            # ---- 计算词汇指标（逐个） ----
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

            # ---- 计算语义指标（批量） ----
            print(f"  计算 BERTScore...")
            bert_results = compute_bertscore(references, candidates)

            # ---- 汇总所有分数 ----
            all_scores = {
                "bleu": bleu_scores,
                "rouge1": rouge1_scores,
                "rouge2": rouge2_scores,
                "rougeL": rougeL_scores,
                "bert_precision": bert_results["precision"],
                "bert_recall": bert_results["recall"],
                "bert_f1": bert_results["f1"],
            }

            # ---- 计算统计量和置信区间 ----
            mode_stats = {}
            for metric_name, scores in all_scores.items():
                ci = compute_confidence_interval(scores, confidence_level)
                mode_stats[metric_name] = ci

            final_results[model_key][mode] = mode_stats

            # 打印摘要
            for metric_name in metric_names:
                stats = mode_stats[metric_name]
                print(
                    f"    {metric_name}: "
                    f"mean={stats['mean']:.4f} "
                    f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
                )

    return final_results


def save_results(
    results: dict,
    baseline_raw: dict,
    rag_raw: dict,
    output_dir: str = "results"
):
    """
    保存实验结果到文件。

    保存内容：
    1. 结构化统计结果（JSON）
    2. 原始输出数据（CSV）
    3. 改进百分比汇总（CSV）

    参数:
        results (dict): 结构化评估结果
        baseline_raw (dict): 原始 Baseline 输出
        rag_raw (dict): 原始 RAG 输出
        output_dir (str): 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 保存统计结果（JSON）
    stats_path = os.path.join(output_dir, f"evaluation_stats_{timestamp}.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 统计结果已保存: {stats_path}")

    # 2. 保存原始输出（CSV）
    raw_rows = []
    for model_key, outputs in baseline_raw.items():
        for output in outputs:
            raw_rows.append({
                "model": model_key,
                "mode": "baseline",
                "query_id": output.get("query_id", ""),
                "query": output.get("query", ""),
                "iteration": output.get("iteration", ""),
                "response": output.get("response", ""),
                "latency": output.get("latency", ""),
                "success": output.get("success", False),
            })
    for model_key, outputs in rag_raw.items():
        for output in outputs:
            raw_rows.append({
                "model": model_key,
                "mode": "rag",
                "query_id": output.get("query_id", ""),
                "query": output.get("query", ""),
                "iteration": output.get("iteration", ""),
                "response": output.get("response", ""),
                "latency": output.get("latency", ""),
                "success": output.get("success", False),
            })

    if raw_rows:
        raw_df = pd.DataFrame(raw_rows)
        raw_path = os.path.join(output_dir, f"raw_outputs_{timestamp}.csv")
        raw_df.to_csv(raw_path, index=False, encoding="utf-8")
        print(f"💾 原始输出已保存: {raw_path}")

    # 3. 保存改进百分比汇总
    metric_names = ["bleu", "rouge1", "rouge2", "rougeL",
                    "bert_precision", "bert_recall", "bert_f1"]
    improvement_rows = []
    for model_key in results.keys():
        row = {"model": model_key}
        for metric in metric_names:
            bl_mean = results[model_key].get("baseline", {}).get(
                metric, {}).get("mean", 0)
            rag_mean = results[model_key].get("rag", {}).get(
                metric, {}).get("mean", 0)
            row[f"{metric}_baseline"] = bl_mean
            row[f"{metric}_rag"] = rag_mean
            row[f"{metric}_improvement"] = compute_improvement(bl_mean, rag_mean)
        improvement_rows.append(row)

    if improvement_rows:
        imp_df = pd.DataFrame(improvement_rows)
        imp_path = os.path.join(output_dir, f"improvement_summary_{timestamp}.csv")
        imp_df.to_csv(imp_path, index=False, encoding="utf-8")
        print(f"💾 改进汇总已保存: {imp_path}")


def main():
    """
    主函数：编排完整的实验流程。
    """
    args = parse_args()

    # =============================================
    # 步骤0: 打印欢迎信息和配置
    # =============================================
    print("=" * 60)
    print("🧪 RAG vs Baseline LLM 多指标评估实验")
    print("   论文复现: \"Retrieval-Augmented Generation vs.")
    print("   Baseline LLMs: A Multi-Metric Evaluation\"")
    print("=" * 60)
    print(f"\n⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 打印配置摘要
    print_config_summary()

    # =============================================
    # 步骤1: 确定实验参数
    # =============================================
    # 模型列表
    if args.models:
        model_keys = args.models
    else:
        model_keys = get_all_model_keys()

    # 迭代次数
    if args.quick:
        num_iterations = 2
        num_queries = 3
        print("\n⚡ 快速测试模式: 2 次迭代, 3 个查询")
    else:
        num_iterations = args.iterations or EXPERIMENT_CONFIG["num_iterations"]
        num_queries = args.queries or EXPERIMENT_CONFIG["num_queries"]

    confidence_level = EXPERIMENT_CONFIG["confidence_level"]
    output_dir = args.output_dir or EXPERIMENT_CONFIG["output_dir"]
    figure_dir = args.figure_dir or EXPERIMENT_CONFIG["figure_dir"]
    document_path = args.document or EXPERIMENT_CONFIG["document_path"]
    ground_truth_path = EXPERIMENT_CONFIG["ground_truth_path"]

    print(f"\n📋 实验参数:")
    print(f"   模型: {model_keys}")
    print(f"   迭代次数: {num_iterations}")
    print(f"   查询数量: {num_queries}")
    print(f"   置信水平: {confidence_level}")
    print(f"   文档路径: {document_path}")

    # =============================================
    # 步骤2: 仅绘图模式
    # =============================================
    if args.plot_only:
        print("\n📊 仅绘图模式：从已有结果生成图表")
        # 查找最新的结果文件
        result_files = [
            f for f in os.listdir(output_dir)
            if f.startswith("evaluation_stats_") and f.endswith(".json")
        ]
        if not result_files:
            print(f"❌ 未在 {output_dir}/ 中找到结果文件")
            sys.exit(1)

        latest_file = sorted(result_files)[-1]
        result_path = os.path.join(output_dir, latest_file)
        print(f"   使用结果文件: {result_path}")

        with open(result_path, "r") as f:
            results = json.load(f)

        generate_all_plots(results, figure_dir)
        print(f"\n✅ 图表已生成于: {figure_dir}/")
        return

    # =============================================
    # 步骤3: 加载数据
    # =============================================
    print("\n" + "=" * 60)
    print("📚 加载实验数据")
    print("=" * 60)

    # 加载 Ground Truth
    ground_truth = load_ground_truth(ground_truth_path)

    # 限制查询数量
    ground_truth = ground_truth[:num_queries]
    queries = [item["query"] for item in ground_truth]

    print(f"\n使用 {len(queries)} 个查询:")
    for i, q in enumerate(queries):
        print(f"  Q{i + 1}: {q[:60]}...")

    # =============================================
    # 步骤4: 构建 FAISS 索引
    # =============================================
    print("\n" + "=" * 60)
    print("🔧 构建 FAISS 向量索引")
    print("=" * 60)

    retriever = FAISSRetriever(
        model_name=EMBEDDING_CONFIG["model_name"],
        embedding_dim=EMBEDDING_CONFIG["embedding_dim"]
    )

    # 尝试加载已有索引
    index_dir = "data/faiss_index"
    if retriever.load_index(index_dir):
        print("📂 使用已有索引")
    else:
        # 需要从文档构建索引
        from data.dataset_loader import load_pdf_document, split_documents

        if not os.path.exists(document_path):
            # 尝试下载
            from data.dataset_loader import download_document
            document_url = EXPERIMENT_CONFIG.get("document_url", "")
            if document_url:
                download_document(document_url, document_path)
            else:
                print(f"❌ 文档未找到: {document_path}")
                print("请提供 PDF 文档路径或设置 DOCUMENT_PATH 环境变量")
                sys.exit(1)

        # 加载和切分文档
        full_text = load_pdf_document(document_path)
        chunks = split_documents(
            full_text,
            chunk_size=RAG_CONFIG["chunk_size"],
            chunk_overlap=RAG_CONFIG["chunk_overlap"]
        )

        # 构建索引
        retriever.encode_documents(chunks)
        retriever.build_index()
        retriever.save_index(index_dir)

    # =============================================
    # 步骤5: 试运行检查
    # =============================================
    if args.dry_run:
        print("\n" + "=" * 60)
        print("✅ 试运行完成！")
        print("=" * 60)
        print("配置和数据验证通过。")
        print(f"  模型数量: {len(model_keys)}")
        print(f"  查询数量: {len(queries)}")
        print(f"  迭代次数: {num_iterations}")
        print(f"  预计总输出: {len(model_keys) * len(queries) * num_iterations * 2}")
        print(f"  FAISS 索引: {retriever.index.ntotal} 个向量")
        return

    # =============================================
    # 步骤6: 运行 Baseline 实验
    # =============================================
    baseline_results = run_baseline_experiments(
        model_keys=model_keys,
        queries=queries,
        num_iterations=num_iterations
    )

    # =============================================
    # 步骤7: 运行 RAG 实验
    # =============================================
    rag_results = run_rag_experiments(
        model_keys=model_keys,
        queries=queries,
        num_iterations=num_iterations,
        retriever=retriever
    )

    # =============================================
    # 步骤8: 评估结果
    # =============================================
    evaluation_results = evaluate_results(
        baseline_results=baseline_results,
        rag_results=rag_results,
        ground_truth=ground_truth,
        num_iterations=num_iterations,
        confidence_level=confidence_level
    )

    # =============================================
    # 步骤9: 保存结果
    # =============================================
    print("\n" + "=" * 60)
    print("💾 保存实验结果")
    print("=" * 60)

    save_results(
        results=evaluation_results,
        baseline_raw=baseline_results,
        rag_raw=rag_results,
        output_dir=output_dir
    )

    # =============================================
    # 步骤10: 生成图表
    # =============================================
    generate_all_plots(evaluation_results, figure_dir)

    # =============================================
    # 完成
    # =============================================
    print("\n" + "=" * 60)
    print("🎉 实验完成！")
    print("=" * 60)
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📂 输出文件:")
    print(f"   结果目录: {output_dir}/")
    print(f"   图表目录: {figure_dir}/")

    # 打印改进幅度摘要
    print(f"\n📊 RAG 改进幅度摘要:")
    metric_names = ["bleu", "rouge1", "rouge2", "rougeL",
                    "bert_recall", "bert_f1"]
    for model_key in model_keys:
        if model_key in evaluation_results:
            print(f"\n  [{model_key}]")
            for metric in metric_names:
                bl = evaluation_results[model_key].get("baseline", {}).get(
                    metric, {}).get("mean", 0)
                rg = evaluation_results[model_key].get("rag", {}).get(
                    metric, {}).get("mean", 0)
                imp = compute_improvement(bl, rg)
                print(f"    {metric}: {bl:.4f} → {rg:.4f} ({imp:+.1f}%)")


if __name__ == "__main__":
    main()
