"""
===========================================================
模型配置模块 (Model Configuration Module)
===========================================================

功能说明：
    统一管理所有模型的 API 配置，包括：
    1. Baseline LLM 的 API 调用配置
    2. RAG 中生成模型的 API 配置
    3. Embedding 模型配置
    4. RAG 流程参数配置
    5. Ground Truth 生成配置（使用高级 LLM）
    6. 实验参数配置

设计原则：
    - 所有 API 密钥和端点通过环境变量或此文件集中管理
    - 模型名称不硬编码在业务逻辑中
    - 支持 OpenAI 兼容格式的 API（如 Ollama、vLLM、Together AI）
    - 项目不绑定任何特定领域，支持任意知识文档

使用方式：
    from config.model_config import LLM_CONFIGS, EMBEDDING_CONFIG, RAG_CONFIG
"""

import os


# =============================================
# 1. LLM 模型配置（用于 Baseline 和 RAG 生成）
# =============================================
# 说明：
#   - 每个模型对应一个独立的配置字典
#   - api_key: API 密钥，优先从环境变量读取，默认为 "ollama"（本地服务无需密钥）
#   - base_url: API 基础地址，Ollama 默认为 http://localhost:11434/v1
#   - model_name: 模型在 API 端点中的注册名称
#   - temperature: 控制输出随机性，0 表示确定性输出，1 表示最大随机性
#   - max_tokens: 生成的最大 token 数量

LLM_CONFIGS = {
    # ---- TinyLlama 1.1B ----
    # 最小的模型，适合边缘设备部署，但语言理解和生成能力有限
    "tinyllama": {
        "api_key": os.environ.get("TINYLLAMA_API_KEY", "ollama"),
        "base_url": os.environ.get("TINYLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "model_name": os.environ.get("TINYLLAMA_MODEL", "tinyllama"),
        "temperature": 0.7,
        "max_tokens": 1024,
    },

    # ---- Mistral 7B ----
    # 高效推理能力，在同参数级别中表现出色
    "mistral": {
        "api_key": os.environ.get("MISTRAL_API_KEY", "ollama"),
        "base_url": os.environ.get("MISTRAL_BASE_URL", "http://localhost:11434/v1"),
        "model_name": os.environ.get("MISTRAL_MODEL", "mistral"),
        "temperature": 0.7,
        "max_tokens": 1024,
    },

    # ---- Llama 3.1 8B ----
    # 增强的推理能力和现代架构
    "llama3.1": {
        "api_key": os.environ.get("LLAMA31_API_KEY", "ollama"),
        "base_url": os.environ.get("LLAMA31_BASE_URL", "http://localhost:11434/v1"),
        "model_name": os.environ.get("LLAMA31_MODEL", "llama3.1"),
        "temperature": 0.7,
        "max_tokens": 1024,
    },

    # ---- Llama 1 13B ----
    # 强大的少样本学习能力，但基于较旧的架构
    "llama1-13b": {
        "api_key": os.environ.get("LLAMA1_API_KEY", "ollama"),
        "base_url": os.environ.get("LLAMA1_BASE_URL", "http://localhost:11434/v1"),
        "model_name": os.environ.get("LLAMA1_MODEL", "llama2:13b"),
        "temperature": 0.7,
        "max_tokens": 1024,
    },
}


# =============================================
# 2. Ground Truth 生成 LLM 配置
# =============================================
# 说明：
#   用于自动生成 Ground Truth 数据的高级 LLM 配置。
#   建议使用能力更强的模型（如 GPT-4、Claude、Qwen-72B 等），
#   以确保生成的查询和答案质量足够高。
#   该模型仅在 step2_generate_ground_truth.py 中使用。

GROUND_TRUTH_LLM_CONFIG = {
    "api_key": os.environ.get("GT_LLM_API_KEY", "ollama"),
    "base_url": os.environ.get("GT_LLM_BASE_URL", "http://localhost:11434/v1"),
    "model_name": os.environ.get("GT_LLM_MODEL", "llama3.1"),
    "temperature": 0.3,    # 较低温度，确保生成内容准确、稳定
    "max_tokens": 2048,    # Ground Truth 答案可能较长，需要更多 token
}


# =============================================
# 3. Embedding 模型配置
# =============================================
# 说明：
#   使用 sentence-transformers 的 "all-MiniLM-L6-v2" 模型
#   该模型在准确性和速度之间取得良好平衡：
#   - 与 BGE-Large 等大模型检索性能接近
#   - 但速度更快、体积更小
#
# 支持两种模式：
#   1. 本地模式（默认）：使用 sentence-transformers 库直接加载
#   2. API 模式：通过 OpenAI 兼容 API 调用

EMBEDDING_CONFIG = {
    # 使用模式: "local" 表示本地加载, "api" 表示通过 API 调用
    "mode": os.environ.get("EMBEDDING_MODE", "local"),

    # 本地模式配置
    "model_name": os.environ.get(
        "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    ),

    # API 模式配置（当 mode="api" 时使用）
    "api_key": os.environ.get("EMBEDDING_API_KEY", ""),
    "base_url": os.environ.get("EMBEDDING_BASE_URL", ""),
    "api_model_name": os.environ.get("EMBEDDING_API_MODEL", ""),

    # 模型维度：all-MiniLM-L6-v2 的输出维度为 384
    "embedding_dim": 384,
}


# =============================================
# 4. RAG 流程配置
# =============================================
# 说明：
#   控制 RAG 流程中文档处理和检索的关键参数
#   这些参数直接影响检索质量和生成效果

RAG_CONFIG = {
    # ---- 文档切分参数 ----
    "chunk_size": 1000,       # 每个文档块的字符数
    "chunk_overlap": 200,     # 相邻块之间的重叠字符数
    "separator": "\n",        # 文本分隔符

    # ---- 检索参数 ----
    "top_k": 5,               # 检索返回的最相似文档数量

    # ---- 提示模板 ----
    "prompt_template": """You are a knowledgeable assistant. Use the following retrieved context to answer the question accurately and comprehensively.

Context:
{context}

Question: {question}

Answer: Based on the provided context, """,
}


# =============================================
# 5. 实验配置
# =============================================
# 说明：
#   控制实验运行的核心参数

EXPERIMENT_CONFIG = {
    # 每个查询的重复执行次数（用于捕捉输出变异性）
    "num_iterations": 11,

    # 置信区间的置信水平
    "confidence_level": 0.90,

    # 输出目录
    "output_dir": "results",
    "figure_dir": "figures",

    # Ground Truth 数据文件路径
    "ground_truth_path": "data/ground_truth.json",

    # 知识文档目录（放置 PDF/XLSX 文件的目录）
    "data_dir": "data",

    # 向量索引保存目录
    "vector_db_dir": "VectorDB",
}


# =============================================
# 6. Ground Truth 生成配置
# =============================================
# 说明：
#   控制 step2_generate_ground_truth.py 的行为

GROUND_TRUTH_CONFIG = {
    # 生成条目数量（默认 11 条，与论文一致）
    "num_entries": int(os.environ.get("GT_NUM_ENTRIES", "11")),

    # 生成复杂度等级：
    #   "simple"   — 事实性问答，答案简短直接
    #   "medium"   — 需要一定推理，答案包含解释
    #   "complex"  — 综合性问题，需要多段落推理，答案详细
    "complexity": os.environ.get("GT_COMPLEXITY", "medium"),

    # 是否启用 Ground Truth 生成步骤
    # 设为 False 时 step2 将跳过生成，直接使用已有 ground_truth.json
    "enabled": os.environ.get("GT_ENABLED", "true").lower() == "true",

    # 输出路径
    "output_path": "data/ground_truth.json",
}


# =============================================
# 7. 辅助函数
# =============================================

def get_llm_config(model_key: str) -> dict:
    """
    根据模型键名获取对应的 LLM 配置。

    参数:
        model_key (str): 模型的键名，如 "tinyllama", "mistral" 等

    返回:
        dict: 包含该模型所有配置信息的字典

    异常:
        KeyError: 当指定的模型键名不存在时抛出
    """
    if model_key not in LLM_CONFIGS:
        available = ", ".join(LLM_CONFIGS.keys())
        raise KeyError(
            f"未找到模型 '{model_key}' 的配置。"
            f"可用模型: {available}"
        )
    return LLM_CONFIGS[model_key]


def get_all_model_keys() -> list:
    """
    获取所有已配置的模型键名列表。

    返回:
        list: 模型键名列表，如 ["tinyllama", "mistral", "llama3.1", "llama1-13b"]
    """
    return list(LLM_CONFIGS.keys())


def print_config_summary():
    """
    打印当前配置摘要，用于运行前确认配置正确。
    隐藏 API 密钥的具体值以确保安全。
    """
    print("=" * 60)
    print("📋 当前配置摘要")
    print("=" * 60)

    print("\n🤖 LLM 模型配置:")
    for key, config in LLM_CONFIGS.items():
        masked_key = config["api_key"][:4] + "****" if len(config["api_key"]) > 4 else "****"
        print(f"  [{key}]")
        print(f"    模型名称: {config['model_name']}")
        print(f"    API 地址: {config['base_url']}")
        print(f"    API 密钥: {masked_key}")
        print(f"    Temperature: {config['temperature']}")
        print(f"    Max Tokens: {config['max_tokens']}")

    print(f"\n🧠 Ground Truth 生成 LLM:")
    gt_key = GROUND_TRUTH_LLM_CONFIG["api_key"][:4] + "****" if len(GROUND_TRUTH_LLM_CONFIG["api_key"]) > 4 else "****"
    print(f"    模型: {GROUND_TRUTH_LLM_CONFIG['model_name']}")
    print(f"    API 地址: {GROUND_TRUTH_LLM_CONFIG['base_url']}")
    print(f"    API 密钥: {gt_key}")

    print(f"\n🔗 Embedding 配置:")
    print(f"    模式: {EMBEDDING_CONFIG['mode']}")
    print(f"    模型: {EMBEDDING_CONFIG['model_name']}")

    print(f"\n📐 RAG 配置:")
    print(f"    Chunk Size: {RAG_CONFIG['chunk_size']}")
    print(f"    Chunk Overlap: {RAG_CONFIG['chunk_overlap']}")
    print(f"    Top-K: {RAG_CONFIG['top_k']}")

    print(f"\n🧪 实验配置:")
    print(f"    迭代次数: {EXPERIMENT_CONFIG['num_iterations']}")
    print(f"    置信水平: {EXPERIMENT_CONFIG['confidence_level']}")
    print(f"    向量库目录: {EXPERIMENT_CONFIG['vector_db_dir']}")

    print(f"\n📝 Ground Truth 配置:")
    print(f"    条目数量: {GROUND_TRUTH_CONFIG['num_entries']}")
    print(f"    复杂度: {GROUND_TRUTH_CONFIG['complexity']}")
    print(f"    启用生成: {GROUND_TRUTH_CONFIG['enabled']}")
    print("=" * 60)


# 当直接运行此文件时，打印配置摘要
if __name__ == "__main__":
    print_config_summary()
