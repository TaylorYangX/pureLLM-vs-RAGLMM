"""
===========================================================
Ground Truth 自动生成模块
===========================================================

功能说明：
    使用高级 LLM 基于提供的知识文档自动生成 ground_truth.json。
    该脚本会：
    1. 加载 data/ 目录中的所有文档（PDF/XLSX）
    2. 对文档进行切分
    3. 从中采样代表性段落
    4. 调用高级 LLM 生成查询-答案对
    5. 输出标准化的 ground_truth.json 文件

可配置项（通过 config/model_config.py 或命令行参数）：
    - num_entries: 生成条目数量（默认 11）
    - complexity: 生成复杂度（simple / medium / complex）
    - enabled: 是否启用生成（False 时跳过）

使用方式：
    # 使用默认配置生成
    python generate_ground_truth.py

    # 指定条目数量和复杂度
    python generate_ground_truth.py --num-entries 20 --complexity complex

    # 指定文档目录
    python generate_ground_truth.py --data-dir data

    # 禁用生成（仅验证已有文件）
    python generate_ground_truth.py --skip
"""

import argparse
import json
import os
import random
import sys
import time

from openai import OpenAI


def get_complexity_prompt(complexity: str) -> str:
    """
    根据复杂度等级返回对应的生成指令。

    不同复杂度对 LLM 生成的查询和答案有不同要求：
    - simple:  直接的事实性问答
    - medium:  需要理解和解释的问答
    - complex: 需要综合多段落信息的分析性问答

    参数:
        complexity (str): 复杂度等级

    返回:
        str: 对应的提示指令文本
    """
    prompts = {
        "simple": """Generate simple, factual questions that can be answered directly from a single passage.
The answers should be short and extracted almost verbatim from the text.
Focus on "what", "who", "when", "where" type questions.""",

        "medium": """Generate questions that require understanding and explanation.
The answers should include context and reasoning, not just raw facts.
Focus on "how", "why", and "what is the significance of" type questions.
Answers should be 2-4 sentences long.""",

        "complex": """Generate complex, analytical questions that require synthesizing information from multiple parts of the text.
The answers should demonstrate deep understanding, include explanations, comparisons, or cause-and-effect relationships.
Focus on questions that require critical thinking, such as "Compare and contrast...", "Explain the relationship between...", "What are the implications of...".
Answers should be comprehensive, 3-6 sentences long.""",
    }

    if complexity not in prompts:
        print(f"⚠️  未知复杂度 '{complexity}'，使用默认 'medium'")
        complexity = "medium"

    return prompts[complexity]


def sample_representative_chunks(chunks: list, num_samples: int) -> list:
    """
    从文档块中采样代表性段落。

    策略：
    1. 如果块数量少于需要的样本数，全部使用
    2. 否则，均匀采样（避免集中在某个区域）
    3. 过滤掉太短的块（少于 100 字符的可能没有足够信息）

    参数:
        chunks (list): 所有文档块
        num_samples (int): 需要的样本数

    返回:
        list: 采样后的文档块列表
    """
    # 过滤太短的块
    valid_chunks = [c for c in chunks if len(c.strip()) >= 100]

    if not valid_chunks:
        print("⚠️  没有足够长的文档块，使用所有可用块")
        valid_chunks = chunks

    if len(valid_chunks) <= num_samples:
        return valid_chunks

    # 均匀采样：将文档分为 num_samples 个区间，每个区间取一个
    step = len(valid_chunks) / num_samples
    sampled = []
    for i in range(num_samples):
        # 在每个区间内随机选一个，增加多样性
        start = int(i * step)
        end = int((i + 1) * step)
        idx = random.randint(start, min(end - 1, len(valid_chunks) - 1))
        sampled.append(valid_chunks[idx])

    return sampled


def generate_qa_pair(
    client: OpenAI,
    model_name: str,
    passage: str,
    complexity: str,
    entry_id: int,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    max_retries: int = 3
) -> dict:
    """
    基于单个文档段落生成一对查询-答案。

    原理：
        将文档段落和生成指令一起发送给高级 LLM，
        要求它生成一个与段落内容相关的查询和对应的标准答案。
        使用 JSON 格式输出，便于自动解析。

    参数:
        client (OpenAI): API 客户端
        model_name (str): 模型名称
        passage (str): 文档段落文本
        complexity (str): 复杂度等级
        entry_id (int): 条目编号
        temperature (float): 生成温度
        max_tokens (int): 最大 token 数
        max_retries (int): 最大重试次数

    返回:
        dict: {"query_id": int, "query": str, "ground_truth": str, "source_passage": str}
    """
    complexity_instruction = get_complexity_prompt(complexity)

    # 构建提示：要求 LLM 生成 JSON 格式的查询-答案对
    prompt = f"""You are an expert question generator. Based on the following passage from a knowledge document, generate exactly ONE question-answer pair.

{complexity_instruction}

IMPORTANT RULES:
1. The question must be answerable using ONLY the information in the passage.
2. The answer must be accurate and grounded in the passage content.
3. Output ONLY valid JSON with no additional text, no markdown formatting, and no code blocks.
4. Use the exact format shown below.

Passage:
---
{passage[:3000]}
---

Output format (JSON only, no other text):
{{"query": "Your question here", "ground_truth": "Your answer here"}}"""

    # 带重试的 API 调用
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a precise question-answer pair generator. Always output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # 解析 LLM 输出
            raw_output = response.choices[0].message.content.strip()

            # 尝试清理输出（移除可能的 markdown 代码块标记）
            if raw_output.startswith("```"):
                # 移除 ```json 和 ``` 包裹
                lines = raw_output.split("\n")
                raw_output = "\n".join(
                    line for line in lines
                    if not line.strip().startswith("```")
                )

            # 解析 JSON
            qa_data = json.loads(raw_output)

            return {
                "query_id": entry_id,
                "query": qa_data.get("query", ""),
                "ground_truth": qa_data.get("ground_truth", ""),
                "source_passage": passage[:500],  # 保留来源段落的前500字符
            }

        except json.JSONDecodeError as e:
            print(f"    ⚠️  JSON 解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            print(f"    ⚠️  API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    # 所有重试都失败，返回空条目
    print(f"    ❌ 条目 {entry_id} 生成失败，跳过")
    return None


def generate_ground_truth(
    chunks: list,
    num_entries: int = 11,
    complexity: str = "medium",
    output_path: str = "data/ground_truth.json",
    llm_config: dict = None
) -> list:
    """
    完整的 Ground Truth 生成流程。

    参数:
        chunks (list): 文档块列表
        num_entries (int): 生成条目数量
        complexity (str): 复杂度等级 (simple/medium/complex)
        output_path (str): 输出 JSON 文件路径
        llm_config (dict): LLM 配置字典

    返回:
        list: 生成的 ground truth 条目列表
    """
    # 加载 LLM 配置
    if llm_config is None:
        from config.model_config import GROUND_TRUTH_LLM_CONFIG
        llm_config = GROUND_TRUTH_LLM_CONFIG

    print("=" * 60)
    print("🧠 Ground Truth 自动生成")
    print("=" * 60)
    print(f"   LLM 模型: {llm_config['model_name']}")
    print(f"   目标条目数: {num_entries}")
    print(f"   复杂度等级: {complexity}")
    print(f"   输出路径: {output_path}")

    # 初始化 API 客户端
    client = OpenAI(
        api_key=llm_config["api_key"],
        base_url=llm_config["base_url"],
    )

    # 采样代表性段落
    sampled_chunks = sample_representative_chunks(chunks, num_entries)
    print(f"\n📑 从 {len(chunks)} 个文档块中采样了 {len(sampled_chunks)} 个段落")

    # 逐个生成查询-答案对
    ground_truth_data = []
    for i, chunk in enumerate(sampled_chunks):
        print(f"\n--- 生成条目 {i + 1}/{len(sampled_chunks)} ---")
        print(f"   段落预览: {chunk[:80]}...")

        qa_pair = generate_qa_pair(
            client=client,
            model_name=llm_config["model_name"],
            passage=chunk,
            complexity=complexity,
            entry_id=i + 1,
            temperature=llm_config.get("temperature", 0.3),
            max_tokens=llm_config.get("max_tokens", 2048),
        )

        if qa_pair and qa_pair.get("query") and qa_pair.get("ground_truth"):
            ground_truth_data.append(qa_pair)
            print(f"   ✅ Q: {qa_pair['query'][:60]}...")
        else:
            print(f"   ❌ 生成失败，跳过")

        # 避免 API 速率限制
        time.sleep(0.5)

    # 保存结果
    if ground_truth_data:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ground_truth_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Ground Truth 生成完成!")
        print(f"   成功生成: {len(ground_truth_data)}/{len(sampled_chunks)} 条")
        print(f"   保存位置: {output_path}")
    else:
        print(f"\n❌ 没有成功生成任何条目")

    return ground_truth_data


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="基于知识文档自动生成 Ground Truth 数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python generate_ground_truth.py                           # 默认配置
  python generate_ground_truth.py --num-entries 20          # 生成 20 条
  python generate_ground_truth.py --complexity complex      # 高复杂度
  python generate_ground_truth.py --skip                    # 跳过生成
        """
    )

    parser.add_argument(
        "--num-entries", type=int, default=None,
        help="生成的条目数量（默认从配置读取，通常为 11）"
    )
    parser.add_argument(
        "--complexity", type=str, default=None,
        choices=["simple", "medium", "complex"],
        help="生成复杂度等级（默认 medium）"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="文档目录路径（默认 data）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出文件路径（默认 data/ground_truth.json）"
    )
    parser.add_argument(
        "--skip", action="store_true",
        help="跳过生成，仅验证已有的 ground_truth.json"
    )

    return parser.parse_args()


# =============================================
# 入口点
# =============================================
if __name__ == "__main__":
    # 确保可以导入项目模块
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from config.model_config import (
        GROUND_TRUTH_CONFIG,
        GROUND_TRUTH_LLM_CONFIG,
        EXPERIMENT_CONFIG,
        RAG_CONFIG,
    )
    from data.dataset_loader import load_all_documents, split_documents

    args = parse_args()

    # 合并命令行参数和配置文件
    num_entries = args.num_entries or GROUND_TRUTH_CONFIG["num_entries"]
    complexity = args.complexity or GROUND_TRUTH_CONFIG["complexity"]
    data_dir = args.data_dir or EXPERIMENT_CONFIG["data_dir"]
    output_path = args.output or GROUND_TRUTH_CONFIG["output_path"]

    # 检查是否跳过
    if args.skip or not GROUND_TRUTH_CONFIG["enabled"]:
        print("⏭️  跳过 Ground Truth 生成")
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            print(f"   已有文件: {output_path} ({len(existing)} 条)")
        else:
            print(f"   ⚠️  {output_path} 不存在，请手动创建或取消 --skip")
        sys.exit(0)

    # 步骤1: 加载文档
    print("\n" + "=" * 60)
    print("📚 加载文档")
    print("=" * 60)

    full_text = load_all_documents(data_dir)

    # 步骤2: 切分文档
    print("\n" + "=" * 60)
    print("📦 切分文档")
    print("=" * 60)

    chunks = split_documents(
        full_text,
        chunk_size=RAG_CONFIG["chunk_size"],
        chunk_overlap=RAG_CONFIG["chunk_overlap"],
    )

    # 步骤3: 生成 Ground Truth
    ground_truth = generate_ground_truth(
        chunks=chunks,
        num_entries=num_entries,
        complexity=complexity,
        output_path=output_path,
        llm_config=GROUND_TRUTH_LLM_CONFIG,
    )

    print(f"\n🎉 完成! 共生成 {len(ground_truth)} 条 Ground Truth 数据")
