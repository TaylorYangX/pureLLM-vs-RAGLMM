# RAG vs Baseline LLM: 多指标评估实验复现

> 论文复现: *"Retrieval-Augmented Generation vs. Baseline LLMs: A Multi-Metric Evaluation for Knowledge-Intensive Content"*
> (Information 2025, 16, 766)

## 📋 项目概述

本项目完整复现了论文中的实验管道，比较 4 个 Baseline LLM 与 RAG-Augmented LLM 在知识密集型任务上的表现。

### 评估模型
| 模型 | 参数量 | 最低显存 |
|------|-------|---------|
| TinyLlama | 1.1B | 1 GB |
| Mistral 7B | 7.3B | 7 GB |
| Llama 3.1 8B | 8B | 8 GB |
| Llama 1 13B | 13B | 13 GB |

### 评估指标（7个）
| 类别 | 指标 | 说明 |
|------|------|------|
| 词汇相似度 | BLEU | n-gram 精确度 |
| 词汇相似度 | ROUGE-1 | 单词召回率 |
| 词汇相似度 | ROUGE-2 | 双词召回率 |
| 词汇相似度 | ROUGE-L | 最长公共子序列 |
| 语义相似度 | BERTScore Precision | 语义精确度 |
| 语义相似度 | BERTScore Recall | 语义召回率 |
| 语义相似度 | BERTScore F1 | 综合语义得分 |

## 📁 项目结构

```
pureLLM-vs-RAGLMM/
├── config/
│   └── model_config.py        # 模型API配置（key, URL, 模型名）
├── data/
│   ├── dataset_loader.py      # 数据加载（PDF文档、Ground Truth）
│   └── ground_truth.json      # 11组查询+标准答案
├── models/
│   ├── llm_baseline.py        # Baseline LLM调用
│   └── rag_pipeline.py        # RAG完整流程（检索+生成）
├── retrieval/
│   └── retriever.py           # FAISS向量检索
├── evaluation/
│   └── metrics.py             # 7个评估指标实现
├── visualization/
│   └── plot_results.py        # 论文图表复现
├── results/                   # 实验结果输出
├── figures/                   # 图表输出
├── main.py                    # 主程序入口
├── requirements.txt           # Python依赖
└── README.md                  # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置模型 API

本项目默认使用 [Ollama](https://ollama.com/) 作为 LLM 推理后端。

#### 安装并启动 Ollama
```bash
# 安装 Ollama（Linux）
curl -fsSL https://ollama.com/install.sh | sh

# 启动服务
ollama serve

# 拉取所需模型
ollama pull tinyllama
ollama pull mistral
ollama pull llama3.1
ollama pull llama2:13b
```

#### 使用其他 API（如 Together AI、OpenRouter）
修改 `config/model_config.py` 中的配置，或通过环境变量设置：

```bash
export MISTRAL_API_KEY="your-api-key"
export MISTRAL_BASE_URL="https://api.together.xyz/v1"
export MISTRAL_MODEL="mistralai/Mistral-7B-Instruct-v0.1"
```

### 3. 准备知识文档

论文使用 "Human Nutrition: 2020 Edition" 教科书。程序会自动尝试下载，或手动放置：

```bash
# 下载文档并放到 data/ 目录
# 文件路径: data/human_nutrition_2020.pdf
```

### 4. 运行实验

```bash
# 完整实验（11查询 × 11迭代 × 4模型 × 2模式 = 968个输出）
python main.py

# 快速测试（3查询 × 2迭代，验证流程是否正确）
python main.py --quick

# 试运行（仅验证配置和数据，不执行实验）
python main.py --dry-run

# 指定模型
python main.py --models mistral llama3.1

# 自定义迭代次数
python main.py --iterations 5

# 仅生成图表（使用已有结果）
python main.py --plot-only
```

## 📊 输出说明

### 结果文件（results/）
- `evaluation_stats_YYYYMMDD_HHMMSS.json` — 结构化统计结果（均值+置信区间）
- `raw_outputs_YYYYMMDD_HHMMSS.csv` — 原始模型输出
- `improvement_summary_YYYYMMDD_HHMMSS.csv` — RAG改进百分比汇总

### 图表文件（figures/）
- `fig3_lexical_comparison.png` — 词汇相似度对比图
- `fig4_semantic_comparison.png` — 语义相似度对比图
- `table3_improvement_heatmap.png` — 改进百分比热力图
- `fig5_cross_model_comparison.png` — 跨模型对比图
- `all_metrics_overview.png` — 综合概览图

## 🔧 RAG 流程详解

```
1. 文档加载 (load_pdf_document)
   ↓ PDF → 纯文本
2. 文档切分 (split_documents)
   ↓ 长文本 → chunk_size=1000, overlap=200
3. 向量化 (encode_documents)
   ↓ 使用 all-MiniLM-L6-v2 生成 384维向量
4. 向量检索 (FAISS search)
   ↓ 查询 → Top-5 相似文档
5. 拼接上下文 (get_context_string)
   ↓ 5个段落 → 拼接为上下文
6. 输入LLM生成 (generate)
   ↓ context + query → LLM → answer
```

## 📝 示例输入输出

### 查询示例
```
Q1: What are the main functions of carbohydrates in the human body?
```

### Baseline LLM 输出（无检索）
```
Carbohydrates provide energy for the body. They are one of the three
macronutrients along with proteins and fats...
(基于模型参数知识的回答，可能不够准确或全面)
```

### RAG-Augmented LLM 输出（有检索增强）
```
Based on the provided context, carbohydrates serve several essential
functions: they are the primary source of energy, providing fuel for
the brain, central nervous system, and muscles during exercise...
(基于检索文档的回答，更准确和全面)
```

### 评估结果示例
```
BLEU:           0.04 → 0.064  (+60.5%)
ROUGE-L:        0.18 → 0.227  (+28.0%)
BERTScore F1:   0.855 → 0.876 (+2.8%)
```

## ⚙️ 自定义配置

### 修改 RAG 参数
在 `config/model_config.py` 中调整：
```python
RAG_CONFIG = {
    "chunk_size": 1000,    # 文档块大小
    "chunk_overlap": 200,  # 块间重叠
    "top_k": 5,            # 检索返回数量
}
```

### 修改实验参数
```python
EXPERIMENT_CONFIG = {
    "num_queries": 11,          # 查询数量
    "num_iterations": 11,       # 迭代次数
    "confidence_level": 0.90,  # 置信水平
}
```

### 自定义 Ground Truth
编辑 `data/ground_truth.json`，格式：
```json
[
  {
    "query_id": 1,
    "query": "你的问题",
    "ground_truth": "标准答案",
    "source_passage": "来源段落"
  }
]
```

## 📖 文件详细说明

| 文件 | 作用 | 关键类/函数 |
|------|------|------------|
| `config/model_config.py` | API配置管理 | `LLM_CONFIGS`, `EMBEDDING_CONFIG`, `RAG_CONFIG` |
| `data/dataset_loader.py` | 数据加载 | `load_pdf_document()`, `split_documents()`, `load_ground_truth()` |
| `retrieval/retriever.py` | FAISS检索 | `FAISSRetriever` 类 |
| `models/llm_baseline.py` | Baseline LLM | `BaselineLLM` 类 |
| `models/rag_pipeline.py` | RAG流程 | `RAGPipeline` 类 |
| `evaluation/metrics.py` | 评估指标 | `compute_bleu()`, `compute_rouge()`, `compute_bertscore()` |
| `visualization/plot_results.py` | 图表生成 | `generate_all_plots()` |
| `main.py` | 主程序 | `main()` |

## 📚 参考文献

- Papineni et al. "BLEU: A Method for Automatic Evaluation of Machine Translation" (ACL 2002)
- Lin & Och. "ROUGE and its Evaluation" (NTCIR 2004)
- Zhang et al. "BERTScore: Evaluating Text Generation with BERT" (ICLR 2020)
- Douze et al. "The Faiss Library" (2024)
- Yin & Zhang. "Sentence Similarity Based on all-MiniLM-L6-v2" (ICIAAI 2024)
