"""
===========================================================
步骤 2: 生成 Ground Truth (Generate Ground Truth)
===========================================================

功能说明：
    调用高级 LLM 基于文档内容自动生成 ground_truth.json。
    本脚本是 generate_ground_truth.py 的简化入口。

    可配置项：
    - 生成条目数量 (--num-entries)
    - 复杂度等级 (--complexity: simple/medium/complex)
    - 是否跳过 (--skip)

使用方式：
    python step2_generate_ground_truth.py
    python step2_generate_ground_truth.py --num-entries 20 --complexity complex
    python step2_generate_ground_truth.py --skip
"""

import os
import sys

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接复用 generate_ground_truth.py 的逻辑
# 该模块包含完整的 CLI 参数解析和执行流程
if __name__ == "__main__":
    # 导入并执行 generate_ground_truth 模块的主逻辑
    from generate_ground_truth import parse_args, generate_ground_truth
    from config.model_config import (
        GROUND_TRUTH_CONFIG,
        GROUND_TRUTH_LLM_CONFIG,
        EXPERIMENT_CONFIG,
        RAG_CONFIG,
    )
    from data.dataset_loader import load_all_documents, split_documents
    import json

    args = parse_args()

    # 合并命令行参数和配置文件
    num_entries = args.num_entries or GROUND_TRUTH_CONFIG["num_entries"]
    complexity = args.complexity or GROUND_TRUTH_CONFIG["complexity"]
    data_dir = args.data_dir or EXPERIMENT_CONFIG["data_dir"]
    output_path = args.output or GROUND_TRUTH_CONFIG["output_path"]

    print("=" * 60)
    print("📝 步骤 2: 生成 Ground Truth")
    print("=" * 60)

    # 检查是否跳过
    if args.skip or not GROUND_TRUTH_CONFIG["enabled"]:
        print("⏭️  跳过 Ground Truth 生成")
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            print(f"   已有文件: {output_path} ({len(existing)} 条)")
        else:
            print(f"   ⚠️  {output_path} 不存在")
            print(f"   请取消 --skip 运行生成，或手动创建文件")
        sys.exit(0)

    # 加载文档
    full_text = load_all_documents(data_dir)

    # 切分文档
    chunks = split_documents(
        full_text,
        chunk_size=RAG_CONFIG["chunk_size"],
        chunk_overlap=RAG_CONFIG["chunk_overlap"],
    )

    # 生成 Ground Truth
    ground_truth = generate_ground_truth(
        chunks=chunks,
        num_entries=num_entries,
        complexity=complexity,
        output_path=output_path,
        llm_config=GROUND_TRUTH_LLM_CONFIG,
    )

    print(f"\n✅ 步骤 2 完成! 共生成 {len(ground_truth)} 条数据")
