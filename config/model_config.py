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
    5. 实验参数配置

设计原则：
    - 所有 API 密钥和端点通过环境变量或此文件集中管理
    - 模型名称不硬编码在业务逻辑中
    - 支持 OpenAI 兼容格式的 API（如 Ollama、vLLM、Together AI）

使用方式：
    from config.model_config import LLM_CONFIGS, EMBEDDING_CONFIG, RAG_CONFIG
"""

import os


# =============================================
# 1. LLM 模型配置
# =============================================
# 说明：
#   - 每个模型对应一个独立的配置字典
#   - api_key: API 密钥，优先从环境变量读取，默认为 "ollama"（本地服务无需密钥）
#   - base_url: API 基础地址，Ollama 默认为 http://localhost:11434/v1
#   - model_name: 模型在 API 端点中的注册名称
#   - temperature: 控制输出随机性，0 表示确定性输出，1 表示最大随机性
#   - max_tokens: 生成的最大 token 数量
#
# 论文使用的4个模型：
#   TinyLlama 1.1B, Mistral 7B, Llama 3.1 8B, Llama 1 13B

LLM_CONFIGS = {
    # ---- TinyLlama 1.1B ----
    # 最小的模型，适合边缘设备部署，但语言理解和生成能力有限
    "tinyllama": {
        "api_key": os.environ.get("TINYLLAMA_API_KEY", "ollama"),
        "base_url": os.environ.get("TINYLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "model_name": os.environ.get("TINYLLAMA_MODEL", "tinyllama"),
        "temperature": 0.7,   # 论文中使用适度随机性以捕捉输出变异性
        "max_tokens": 1024,   # 限制输出长度，与论文设置一致
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
# 2. Embedding 模型配置
# =============================================
# 说明：
#   论文使用 sentence-transformers 的 "all-MiniLM-L6-v2" 模型
#   该模型在准确性和速度之间取得良好平衡：
#   - 与 BGE-Large 等大模型检索性能接近
#   - 但速度更快、体积更小
#   - 适合大规模多查询 RAG 评估
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
    ),  # 论文指定的 embedding 模型

    # API 模式配置（当 mode="api" 时使用）
    "api_key": os.environ.get("EMBEDDING_API_KEY", ""),
    "base_url": os.environ.get("EMBEDDING_BASE_URL", ""),
    "api_model_name": os.environ.get("EMBEDDING_API_MODEL", ""),

    # 模型维度：all-MiniLM-L6-v2 的输出维度为 384
    "embedding_dim": 384,
}


# =============================================
# 3. RAG 流程配置
# =============================================
# 说明：
#   控制 RAG 流程中文档处理和检索的关键参数
#   这些参数直接影响检索质量和生成效果

RAG_CONFIG = {
    # ---- 文档切分参数 ----
    # chunk_size: 每个文档块的字符数
    # 较大的 chunk 保留更多上下文，但可能引入噪声
    # 较小的 chunk 更精确，但可能丢失上下文
    "chunk_size": 1000,

    # chunk_overlap: 相邻块之间的重叠字符数
    # 重叠确保跨块边界的信息不丢失
    "chunk_overlap": 200,

    # separator: 文本分隔符，用于优先在自然边界处切分
    "separator": "\n",

    # ---- 检索参数 ----
    # top_k: 检索返回的最相似文档数量
    # 论文明确指定 top_k = 5
    "top_k": 5,

    # ---- 提示模板 ----
    # RAG 生成时使用的提示模板
    # 遵循 LangChain 风格：将检索到的上下文与查询组合
    "prompt_template": """You are a knowledgeable assistant. Use the following retrieved context to answer the question accurately and comprehensively.

Context:
{context}

Question: {question}

Answer: Based on the provided context, """,
}


# =============================================
# 4. 实验配置
# =============================================
# 说明：
#   控制实验运行的核心参数
#   论文中每个模型运行 11 个查询 × 11 次迭代 = 121 个输出

EXPERIMENT_CONFIG = {
    # num_queries: 评估使用的查询数量（论文使用 11 个）
    "num_queries": 11,

    # num_iterations: 每个查询的重复执行次数（论文使用 11 次）
    # 多次执行用于捕捉输出变异性，确保可重复性
    "num_iterations": 11,

    # confidence_level: 置信区间的置信水平（论文使用 90%）
    "confidence_level": 0.90,

    # output_dir: 结果输出目录
    "output_dir": "results",

    # figure_dir: 图表输出目录
    "figure_dir": "figures",

    # ground_truth_path: Ground Truth 数据文件路径
    "ground_truth_path": "data/ground_truth.json",

    # document_path: 知识文档路径（PDF）
    # 论文使用 "Human Nutrition: 2020 Edition"
    "document_path": os.environ.get(
        "DOCUMENT_PATH", "data/human_nutrition_2020.pdf"
    ),

    # document_url: 文档下载 URL（备用）
    "document_url": "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf",
}


# =============================================
# 5. 辅助函数
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
        # 隐藏 API 密钥，仅显示前4位
        masked_key = config["api_key"][:4] + "****" if len(config["api_key"]) > 4 else "****"
        print(f"  [{key}]")
        print(f"    模型名称: {config['model_name']}")
        print(f"    API 地址: {config['base_url']}")
        print(f"    API 密钥: {masked_key}")
        print(f"    Temperature: {config['temperature']}")
        print(f"    Max Tokens: {config['max_tokens']}")

    print(f"\n🔗 Embedding 配置:")
    print(f"    模式: {EMBEDDING_CONFIG['mode']}")
    print(f"    模型: {EMBEDDING_CONFIG['model_name']}")

    print(f"\n📐 RAG 配置:")
    print(f"    Chunk Size: {RAG_CONFIG['chunk_size']}")
    print(f"    Chunk Overlap: {RAG_CONFIG['chunk_overlap']}")
    print(f"    Top-K: {RAG_CONFIG['top_k']}")

    print(f"\n🧪 实验配置:")
    print(f"    查询数量: {EXPERIMENT_CONFIG['num_queries']}")
    print(f"    迭代次数: {EXPERIMENT_CONFIG['num_iterations']}")
    print(f"    置信水平: {EXPERIMENT_CONFIG['confidence_level']}")
    print("=" * 60)


# 当直接运行此文件时，打印配置摘要
if __name__ == "__main__":
    print_config_summary()
