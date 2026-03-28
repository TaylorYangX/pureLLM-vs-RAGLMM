"""
===========================================================
步骤 3: 运行实验 (Run Experiments)
===========================================================

功能说明：
    运行 Baseline LLM 和 RAG-Augmented LLM 的生成实验。
    对每个模型、每个查询执行多次迭代，记录所有输出。

    前置条件：
    - VectorDB/ 中已有向量索引 (step1)
    - data/ground_truth.json 已存在 (step2 或手动创建)

    输出：
    - results/raw_outputs_YYYYMMDD_HHMMSS.csv

使用方式：
    python step3_run_experiments.py
    python step3_run_experiments.py --quick                    # 快速测试
    python step3_run_experiments.py --models mistral llama3.1  # 指定模型
    python step3_run_experiments.py --iterations 5             # 自定义迭代
"""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.model_config import (
    get_all_model_keys,
    EXPERIMENT_CONFIG,
    EMBEDDING_CONFIG,
)
from data.dataset_loader import load_ground_truth
from retrieval.retriever import FAISSRetriever
from models.llm_baseline import BaselineLLM
from models.rag_pipeline import RAGPipeline


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="步骤 3: 运行 Baseline + RAG 实验"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="快速测试模式：2 次迭代，3 个查询"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="指定要评估的模型（默认全部）"
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="每个查询的迭代次数"
    )
    parser.add_argument(
        "--queries", type=int, default=None,
        help="使用的查询数量"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="结果输出目录"
    )
    parser.add_argument(
        "--vector-dir", type=str, default=None,
        help="向量索引目录（默认 VectorDB）"
    )
    return parser.parse_args()


def run_baseline_experiments(model_keys, queries, num_iterations):
    """
    运行所有 Baseline LLM 实验。

    参数:
        model_keys (list): 模型键名列表
        queries (list): 查询列表（包含 query_id 和 query）
        num_iterations (int): 迭代次数

    返回:
        dict: {model_key: [输出结果列表]}
    """
    print("\n" + "=" * 60)
    print("🤖 运行 Baseline LLM 实验")
    print("=" * 60)

    all_results = {}
    for model_key in model_keys:
        print(f"\n--- {model_key} ---")
        try:
            llm = BaselineLLM(model_key)
            query_texts = [q["query"] for q in queries]
            results = llm.generate_batch(
                queries=query_texts,
                num_iterations=num_iterations
            )
            # 补充 query_id 信息
            idx = 0
            for q_idx, q in enumerate(queries):
                for it in range(num_iterations):
                    if idx < len(results):
                        results[idx]["query_id"] = q["query_id"]
                    idx += 1

            all_results[model_key] = results
            success_count = sum(1 for r in results if r.get("success", False))
            print(f"  ✅ 完成: {success_count}/{len(results)} 成功")
        except Exception as e:
            print(f"  ❌ 模型 {model_key} 执行失败: {e}")
            all_results[model_key] = []

    return all_results


def run_rag_experiments(model_keys, queries, num_iterations, retriever):
    """
    运行所有 RAG-Augmented LLM 实验。

    参数:
        model_keys (list): 模型键名列表
        queries (list): 查询列表
        num_iterations (int): 迭代次数
        retriever (FAISSRetriever): FAISS 检索器

    返回:
        dict: {model_key: [输出结果列表]}
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
                retriever=retriever,
            )
            query_texts = [q["query"] for q in queries]
            results = rag.generate_batch(
                queries=query_texts,
                num_iterations=num_iterations
            )
            # 补充 query_id 信息
            idx = 0
            for q_idx, q in enumerate(queries):
                for it in range(num_iterations):
                    if idx < len(results):
                        results[idx]["query_id"] = q["query_id"]
                    idx += 1

            all_results[model_key] = results
            success_count = sum(1 for r in results if r.get("success", False))
            print(f"  ✅ 完成: {success_count}/{len(results)} 成功")
        except Exception as e:
            print(f"  ❌ RAG-{model_key} 执行失败: {e}")
            all_results[model_key] = []

    return all_results


def save_raw_outputs(baseline_results, rag_results, output_dir):
    """
    保存原始实验输出为 CSV。

    参数:
        baseline_results (dict): Baseline 结果
        rag_results (dict): RAG 结果
        output_dir (str): 输出目录

    返回:
        str: 保存的文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_rows = []
    for model_key, outputs in baseline_results.items():
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
    for model_key, outputs in rag_results.items():
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

    raw_path = os.path.join(output_dir, f"raw_outputs_{timestamp}.csv")
    if raw_rows:
        df = pd.DataFrame(raw_rows)
        df.to_csv(raw_path, index=False, encoding="utf-8")
        print(f"💾 原始输出已保存: {raw_path}")
    else:
        print("⚠️  没有输出数据")

    return raw_path


def main():
    args = parse_args()

    # 读取配置
    model_keys = args.models or get_all_model_keys()
    output_dir = args.output_dir or EXPERIMENT_CONFIG["output_dir"]
    vector_dir = args.vector_dir or EXPERIMENT_CONFIG["vector_db_dir"]
    gt_path = EXPERIMENT_CONFIG["ground_truth_path"]

    if args.quick:
        num_iterations = 2
        num_queries = 3
        print("⚡ 快速测试模式: 2 次迭代, 3 个查询")
    else:
        num_iterations = args.iterations or EXPERIMENT_CONFIG["num_iterations"]
        num_queries = args.queries or None  # None 表示使用全部

    print("=" * 60)
    print("🧪 步骤 3: 运行实验")
    print("=" * 60)
    print(f"   模型: {model_keys}")
    print(f"   迭代次数: {num_iterations}")

    # 加载 Ground Truth
    ground_truth = load_ground_truth(gt_path)
    if num_queries:
        ground_truth = ground_truth[:num_queries]

    print(f"   查询数量: {len(ground_truth)}")

    # 加载向量索引
    print("\n--- 加载向量索引 ---")
    retriever = FAISSRetriever(
        model_name=EMBEDDING_CONFIG["model_name"],
        embedding_dim=EMBEDDING_CONFIG["embedding_dim"],
    )
    if not retriever.load_index(vector_dir):
        print(f"❌ 向量索引未找到: {vector_dir}/")
        print("   请先运行 step1_build_index.py")
        sys.exit(1)

    # 运行 Baseline 实验
    baseline_results = run_baseline_experiments(
        model_keys, ground_truth, num_iterations
    )

    # 运行 RAG 实验
    rag_results = run_rag_experiments(
        model_keys, ground_truth, num_iterations, retriever
    )

    # 保存原始输出
    raw_path = save_raw_outputs(baseline_results, rag_results, output_dir)

    print(f"\n✅ 步骤 3 完成!")
    print(f"   输出文件: {raw_path}")
    print(f"   下一步: python step4_evaluate.py")


if __name__ == "__main__":
    main()
