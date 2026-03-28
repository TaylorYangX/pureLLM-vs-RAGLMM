"""
===========================================================
数据加载模块 (Dataset Loader Module)
===========================================================

功能说明：
    负责加载和预处理实验所需的所有数据，包括：
    1. PDF 文档加载 —— 从 PDF 文件中提取纯文本
    2. 文档切分 —— 将长文本切分为适合向量检索的小块
    3. Ground Truth 加载 —— 加载查询和标准答案数据
    4. 文档下载 —— 如果本地没有文档，从 URL 下载

设计说明：
    论文使用 "Human Nutrition: 2020 Edition" 教科书作为知识源。
    文档被切分为小块后进行向量化，存入 FAISS 索引。
    切分时使用重叠（overlap）确保跨块边界的信息不丢失。
"""

import json
import os
import urllib.request
from typing import Optional

import PyPDF2


def download_document(url: str, save_path: str) -> str:
    """
    从 URL 下载文档到本地。

    原理：
        论文使用在线教科书作为知识源。如果本地没有 PDF 文件，
        此函数会从指定 URL 下载。使用 urllib 而非 requests，
        减少外部依赖。

    参数:
        url (str): 文档的下载链接
        save_path (str): 本地保存路径

    返回:
        str: 下载后的文件路径
    """
    # 如果文件已存在，直接返回路径，避免重复下载
    if os.path.exists(save_path):
        print(f"📄 文档已存在: {save_path}")
        return save_path

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"⬇️  正在下载文档: {url}")
    print(f"   保存至: {save_path}")

    try:
        # 使用 urllib 下载文件，设置超时防止挂起
        urllib.request.urlretrieve(url, save_path)
        print(f"✅ 下载完成，文件大小: {os.path.getsize(save_path) / 1024 / 1024:.1f} MB")
    except Exception as e:
        raise RuntimeError(
            f"❌ 下载文档失败: {e}\n"
            f"请手动下载文档并放置在: {save_path}\n"
            f"下载链接: {url}"
        )

    return save_path


def load_pdf_document(pdf_path: str) -> str:
    """
    从 PDF 文件中提取全部文本内容。

    原理：
        使用 PyPDF2 逐页读取 PDF 内容。PyPDF2 是纯 Python 实现，
        无需外部系统依赖（如 poppler），适合跨平台使用。
        提取的文本将用于后续的文档切分和向量化。

    参数:
        pdf_path (str): PDF 文件的路径

    返回:
        str: 提取的全部文本内容（各页文本用换行符连接）

    异常:
        FileNotFoundError: 当指定的 PDF 文件不存在时
        RuntimeError: 当 PDF 解析失败时
    """
    # 验证文件是否存在
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"❌ PDF 文件未找到: {pdf_path}\n"
            f"请确认文件路径正确，或使用 download_document() 下载。"
        )

    print(f"📖 正在加载 PDF: {pdf_path}")

    try:
        # 打开 PDF 文件并逐页提取文本
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            print(f"   共 {total_pages} 页")

            # 存储每页提取的文本
            all_text = []
            for page_num, page in enumerate(reader.pages):
                # 提取单页文本
                page_text = page.extract_text()
                if page_text:
                    # 清理文本：去除多余空白，但保留段落结构
                    cleaned_text = page_text.strip()
                    all_text.append(cleaned_text)

            # 用换行符连接所有页面的文本
            full_text = "\n\n".join(all_text)
            print(f"   提取文本长度: {len(full_text)} 字符")

            return full_text

    except Exception as e:
        raise RuntimeError(f"❌ PDF 解析失败: {e}")


def split_documents(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separator: str = "\n"
) -> list:
    """
    将长文本切分为适合向量检索的小块。

    原理：
        RAG 的核心步骤之一。长文本无法直接输入 embedding 模型，
        需要切分为较小的"块"（chunk）。切分策略直接影响检索质量：

        1. chunk_size 控制每块的大小：
           - 太大：包含过多无关信息，降低检索精度
           - 太小：丢失上下文，导致信息碎片化

        2. chunk_overlap 控制相邻块的重叠：
           - 重叠确保跨块边界的完整句子不被截断
           - 提高检索时找到完整相关段落的概率

        3. separator 优先在自然边界（如换行符）处切分：
           - 避免在句子中间断开
           - 保持语义完整性

    参数:
        text (str): 待切分的完整文本
        chunk_size (int): 每个块的目标字符数，默认 1000
        chunk_overlap (int): 相邻块间重叠的字符数，默认 200
        separator (str): 优先切分的分隔符，默认 "\n"

    返回:
        list: 切分后的文本块列表，每个元素为一个字符串
    """
    # 参数验证
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})"
        )

    # 步骤1: 按分隔符初步分割文本
    # 这样可以优先在自然段落边界处切分
    splits = text.split(separator)

    chunks = []          # 最终的文本块列表
    current_chunk = []   # 当前正在构建的块（存储分割后的片段）
    current_length = 0   # 当前块的累计字符数

    for split in splits:
        split_length = len(split)

        # 判断是否需要开始新的块：
        # 当加入当前片段后超过 chunk_size 时，保存当前块并开始新块
        if current_length + split_length + 1 > chunk_size and current_chunk:
            # 将当前块的所有片段合并为一个字符串
            chunk_text = separator.join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

            # 实现重叠：保留当前块末尾的部分片段
            # 从后往前累计，直到达到 overlap 的字符数
            overlap_chunks = []
            overlap_length = 0
            for prev_split in reversed(current_chunk):
                if overlap_length + len(prev_split) + 1 <= chunk_overlap:
                    overlap_chunks.insert(0, prev_split)
                    overlap_length += len(prev_split) + 1
                else:
                    break

            # 新块从重叠部分开始
            current_chunk = overlap_chunks
            current_length = overlap_length

        # 将当前片段加入块
        current_chunk.append(split)
        current_length += split_length + 1  # +1 是分隔符的长度

    # 处理最后一个块
    if current_chunk:
        chunk_text = separator.join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

    print(f"📦 文档切分完成: {len(chunks)} 个文本块")
    print(f"   平均块大小: {sum(len(c) for c in chunks) / len(chunks):.0f} 字符")

    return chunks


def load_ground_truth(json_path: str) -> list:
    """
    加载 Ground Truth 数据（查询 + 标准答案）。

    原理：
        论文使用半自动方式生成 Ground Truth：
        1. 从教科书中提取有意义的段落
        2. 基于段落生成查询和答案
        3. 由两位领域专家审核正确性

        数据格式为包含三元组的列表：
        - query: 查询问题
        - ground_truth: 标准答案
        - source_passage: 答案来源的原文段落

    参数:
        json_path (str): Ground Truth JSON 文件的路径

    返回:
        list: 包含查询和答案的字典列表
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"❌ Ground Truth 文件未找到: {json_path}\n"
            f"请确认文件路径正确。"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"📋 加载 Ground Truth: {len(data)} 条查询")
    for item in data:
        print(f"   Q{item['query_id']}: {item['query'][:50]}...")

    return data


def prepare_experiment_data(
    document_path: str,
    ground_truth_path: str,
    document_url: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> tuple:
    """
    准备实验所需的全部数据：文档块和 Ground Truth。

    这是一个便捷函数，整合了以下步骤：
    1. 如果本地没有文档，尝试下载
    2. 加载 PDF 并提取文本
    3. 将文本切分为块
    4. 加载 Ground Truth

    参数:
        document_path (str): PDF 文档路径
        ground_truth_path (str): Ground Truth JSON 路径
        document_url (str, optional): 文档下载 URL
        chunk_size (int): 文档块大小
        chunk_overlap (int): 文档块重叠大小

    返回:
        tuple: (chunks, ground_truth_data)
            - chunks: 文档文本块列表
            - ground_truth_data: Ground Truth 数据列表
    """
    print("=" * 60)
    print("📚 准备实验数据")
    print("=" * 60)

    # 步骤1: 确保文档存在
    if not os.path.exists(document_path):
        if document_url:
            print(f"\n⚠️  本地文档不存在，尝试从 URL 下载...")
            download_document(document_url, document_path)
        else:
            raise FileNotFoundError(
                f"❌ 文档未找到: {document_path}\n"
                f"请提供文档路径或下载 URL。"
            )

    # 步骤2: 加载 PDF 文档
    print("\n--- 加载文档 ---")
    full_text = load_pdf_document(document_path)

    # 步骤3: 切分文档
    print("\n--- 切分文档 ---")
    chunks = split_documents(
        text=full_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # 步骤4: 加载 Ground Truth
    print("\n--- 加载 Ground Truth ---")
    ground_truth_data = load_ground_truth(ground_truth_path)

    print("\n✅ 数据准备完成!")
    print(f"   文档块数量: {len(chunks)}")
    print(f"   查询数量: {len(ground_truth_data)}")
    print("=" * 60)

    return chunks, ground_truth_data


# =============================================
# 单元测试 / 功能演示
# =============================================
if __name__ == "__main__":
    # 测试 Ground Truth 加载
    print("测试 Ground Truth 加载...")
    gt_path = os.path.join(os.path.dirname(__file__), "ground_truth.json")
    if os.path.exists(gt_path):
        data = load_ground_truth(gt_path)
        print(f"\n示例查询: {data[0]['query']}")
        print(f"示例答案: {data[0]['ground_truth'][:100]}...")

    # 测试文本切分
    print("\n\n测试文本切分...")
    sample_text = "This is a test.\n" * 100
    chunks = split_documents(sample_text, chunk_size=200, chunk_overlap=50)
    print(f"输入长度: {len(sample_text)}, 输出块数: {len(chunks)}")
