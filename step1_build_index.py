"""
===========================================================
步骤 1: 构建向量索引 (Build Vector Index)
===========================================================

功能说明：
    加载 data/ 目录中所有文档（PDF/XLSX），
    将文本切分为块并向量化，构建 FAISS 索引，
    保存到 VectorDB/ 目录。

    这是整个实验流程的第一步，后续步骤依赖此索引。

使用方式：
    python step1_build_index.py
    python step1_build_index.py --data-dir data --vector-dir VectorDB
    python step1_build_index.py --rebuild           # 强制重建
"""

import argparse
import os
import sys

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.model_config import EMBEDDING_CONFIG, RAG_CONFIG, EXPERIMENT_CONFIG
from data.dataset_loader import load_all_documents, split_documents
from retrieval.retriever import FAISSRetriever


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="步骤 1: 构建 FAISS 向量索引"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="文档目录路径（默认 data）"
    )
    parser.add_argument(
        "--vector-dir", type=str, default=None,
        help="向量索引保存目录（默认 VectorDB）"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="强制重建索引（忽略已有索引）"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="文档块大小（默认 1000）"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=None,
        help="文档块重叠大小（默认 200）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 读取配置
    data_dir = args.data_dir or EXPERIMENT_CONFIG["data_dir"]
    vector_dir = args.vector_dir or EXPERIMENT_CONFIG["vector_db_dir"]
    chunk_size = args.chunk_size or RAG_CONFIG["chunk_size"]
    chunk_overlap = args.chunk_overlap or RAG_CONFIG["chunk_overlap"]

    print("=" * 60)
    print("🔧 步骤 1: 构建 FAISS 向量索引")
    print("=" * 60)
    print(f"   文档目录: {data_dir}")
    print(f"   向量保存: {vector_dir}")
    print(f"   块大小: {chunk_size}, 重叠: {chunk_overlap}")

    # 检查是否已有索引且不需要重建
    retriever = FAISSRetriever(
        model_name=EMBEDDING_CONFIG["model_name"],
        embedding_dim=EMBEDDING_CONFIG["embedding_dim"],
    )

    if not args.rebuild and retriever.load_index(vector_dir):
        print(f"\n📂 已有索引加载成功 ({retriever.index.ntotal} 个向量)")
        print("   如需重建，请使用 --rebuild 参数")
        return

    # 步骤1: 加载所有文档
    print("\n--- 加载文档 ---")
    full_text = load_all_documents(data_dir)

    # 步骤2: 切分文档
    print("\n--- 切分文档 ---")
    chunks = split_documents(
        full_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 步骤3: 向量化
    print("\n--- 向量化文档 ---")
    retriever.encode_documents(chunks)

    # 步骤4: 构建索引
    print("\n--- 构建 FAISS 索引 ---")
    retriever.build_index()

    # 步骤5: 保存到 VectorDB/
    print("\n--- 保存索引 ---")
    retriever.save_index(vector_dir)

    print(f"\n✅ 步骤 1 完成!")
    print(f"   索引向量数: {retriever.index.ntotal}")
    print(f"   文档块数: {len(chunks)}")
    print(f"   保存位置: {vector_dir}/")


if __name__ == "__main__":
    main()
