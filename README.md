# RAG vs Baseline LLM: 多指标评估实验

> 基于论文 *"Retrieval-Augmented Generation vs. Baseline LLMs: A Multi-Metric Evaluation"* 的方法论实现。
> 本项目已泛化为通用的 RAG 评估框架，支持任意领域的知识文档。

## 📋 项目概述

本项目实现了完整的 RAG (检索增强生成) 评估管道，比较 Baseline LLM 与 RAG-Augmented LLM 在知识密集型任务上的表现。

### 核心特性

- ✅ **通用化**：支持任意 PDF/XLSX 知识文档，不限于特定领域
- ✅ **自动化**：一键构建索引、生成 Ground Truth、运行实验、评估、可视化
- ✅ **模块化**：每个步骤独立执行，便于调试和定制
- ✅ **7 个评估指标**：BLEU、ROUGE-1/2/L、BERTScore P/R/F1
- ✅ **统计分析**：90% 置信区间、百分比改进

### 评估指标

| 类别       | 指标                | 说明                                      |
| ---------- | ------------------- | ----------------------------------------- |
| 词汇相似度 | BLEU                | n-gram 精确度                             |
| 词汇相似度 | ROUGE-1             | 单词召回率                                |
| 词汇相似度 | ROUGE-2             | 双词召回率                                |
| 词汇相似度 | ROUGE-L             | 最长公共子序列                            |
| 语义相似度 | BERTScore Precision | 语义精确度                                |
| 语义相似度 | BERTScore Recall    | 语义召回率                                |
| 语义相似度 | BERTScore F1        | 综合语义得分                              |
| 语义       | NLI                 | the following three, we select entailment |

| 标签          | 含义                                         |
| ------------- | -------------------------------------------- |
| entailment    | 模型输出支持参考答案 → 语义一致             |
| contradiction | 模型输出与参考答案语义相反                   |
| neutral       | 模型输出与参考答案没有直接关系（或部分一致） |

## 📁 项目结构

```
pureLLM-vs-RAGLMM/
│
│  # ===== 分步执行脚本（推荐） =====
├── step1_build_index.py            # 步骤1：构建 FAISS 向量索引
├── step2_generate_ground_truth.py  # 步骤2：生成 Ground Truth
├── step3_run_experiments.py        # 步骤3：运行 Baseline + RAG 实验
├── step4_evaluate.py               # 步骤4：计算评估指标
├── step5_visualize.py              # 步骤5：生成可视化图表
│
│  # ===== 可选的一键入口 =====
├── main.py                         # 一键运行所有步骤
│
│  # ===== 核心模块 =====
├── config/
│   └── model_config.py             # 统一配置（模型、RAG参数、实验参数）
├── data/
│   ├── dataset_loader.py           # 文档加载器（PDF + XLSX）
│   └── ground_truth.json           # Ground Truth 数据（自动/手动生成）
├── models/
│   ├── llm_baseline.py             # Baseline LLM 调用
│   └── rag_pipeline.py             # RAG 完整流程（检索+生成）
├── retrieval/
│   └── retriever.py                # FAISS 向量检索器
├── evaluation/
│   └── metrics.py                  # 7 个评估指标实现
├── visualization/
│   └── plot_results.py             # 论文图表复现
├── generate_ground_truth.py        # Ground Truth 生成器（核心逻辑）
│
│  # ===== 输出目录 =====
├── VectorDB/                       # FAISS 向量索引存储
├── results/                        # 实验结果输出
├── figures/                        # 图表输出
│
│  # ===== 配置 =====
├── requirements.txt                # Python 依赖
└── README.md                       # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
python3 -m pip install -r requirements.txt
```

### 2. 准备文档

将你的知识文档放入 `data/` 目录：

```bash
# 支持的格式：PDF 和 XLSX
cp your_document.pdf data/
cp your_data.xlsx data/
```

> **重要**：无需手动下载特定文档，项目会自动扫描 `data/` 目录中的所有 PDF 和 XLSX 文件。

### 3. 配置 LLM

本项目默认使用 [Ollama](https://ollama.com/) 作为推理后端：

```bash
# 安装并启动 Ollama
ollama serve

# 拉取模型
ollama pull tinyllama
ollama pull mistral
ollama pull llama3.1
ollama pull llama2:13b
```

如需使用其他 API（如 OpenAI、Together AI），修改 `config/model_config.py` 或设置环境变量。

### 4. 运行实验

#### 方式一：分步执行（推荐）

```bash
# 步骤1：构建向量索引 → 保存到 VectorDB/
python step1_build_index.py

# 步骤2：自动生成 Ground Truth → data/ground_truth.json
python step2_generate_ground_truth.py

# 步骤3：运行 Baseline + RAG 实验 → results/raw_outputs_*.csv
python step3_run_experiments.py

# 步骤4：计算所有评估指标 → results/evaluation_stats_*.json
python step4_evaluate.py

# 步骤5：生成论文图表 → figures/*.png
python step5_visualize.py
```

#### 方式二：一键执行

```bash
python main.py                     # 执行所有步骤
python main.py --quick             # 快速测试
python main.py --skip-gt           # 跳过 Ground Truth 生成
python main.py --skip-index        # 跳过索引构建
```

## 📝 各步骤详细说明

### Step 1: 构建向量索引 (`step1_build_index.py`)

- **输入**：`data/` 目录中的 PDF 和 XLSX 文件
- **输出**：`VectorDB/` 目录（包含 FAISS 索引和文档数据）
- **功能**：加载文档 → 提取文本 → 切分为块 → 向量化 → 构建 FAISS 索引

```bash
python step1_build_index.py
python step1_build_index.py --rebuild           # 强制重建
python step1_build_index.py --chunk-size 500    # 自定义块大小
```

### Step 2: 生成 Ground Truth (`step2_generate_ground_truth.py`)

- **输入**：`data/` 中的文档 + 高级 LLM API
- **输出**：`data/ground_truth.json`
- **功能**：从文档中采样段落 → 调用 LLM 生成查询-答案对

```bash
python step2_generate_ground_truth.py
python step2_generate_ground_truth.py --num-entries 20        # 生成 20 条
python step2_generate_ground_truth.py --complexity complex    # 高复杂度
python step2_generate_ground_truth.py --skip                  # 跳过生成
```

**复杂度等级说明**：

| 等级        | 说明                              | 答案长度 |
| ----------- | --------------------------------- | -------- |
| `simple`  | 事实性问答（what/who/when/where） | 1-2 句   |
| `medium`  | 理解性问答（how/why）             | 2-4 句   |
| `complex` | 分析性问答（比较、因果、综合）    | 3-6 句   |

### Step 3: 运行实验 (`step3_run_experiments.py`)

- **输入**：`VectorDB/` + `data/ground_truth.json` + LLM API
- **输出**：`results/raw_outputs_YYYYMMDD_HHMMSS.csv`
- **功能**：对每个模型运行 Baseline 和 RAG 模式的生成

```bash
python step3_run_experiments.py
python step3_run_experiments.py --quick                    # 快速测试
python step3_run_experiments.py --models mistral llama3.1  # 指定模型
python step3_run_experiments.py --iterations 5             # 自定义迭代
```

### Step 4: 评估结果 (`step4_evaluate.py`)

- **输入**：`results/raw_outputs_*.csv` + `data/ground_truth.json`
- **输出**：
  - `results/evaluation_stats_YYYYMMDD_HHMMSS.json`（统计结果）
  - `results/improvement_summary_YYYYMMDD_HHMMSS.csv`（改进汇总）
- **功能**：计算 7 个指标 + 90% 置信区间

```bash
python step4_evaluate.py
python step4_evaluate.py --input results/raw_outputs_20260328.csv
```

### Step 5: 生成图表 (`step5_visualize.py`)

- **输入**：`results/evaluation_stats_*.json`
- **输出**：`figures/*.png`（5 张图表）
- **功能**：复现论文中的 Figure 3/4/5 和 Table 3

```bash
python step5_visualize.py
python step5_visualize.py --input results/evaluation_stats_20260328.json
```

## ⚙️ 配置说明

所有配置集中在 `config/model_config.py`：

### LLM 模型配置

```python
# 修改模型或切换 API 提供商
LLM_CONFIGS = {
    "mistral": {
        "api_key": os.environ.get("MISTRAL_API_KEY", "ollama"),
        "base_url": os.environ.get("MISTRAL_BASE_URL", "http://localhost:11434/v1"),
        "model_name": "mistral",
        "temperature": 0.7,
        "max_tokens": 1024,
    },
    # ... 其他模型
}
```

### Ground Truth 生成 LLM（建议使用更强的模型）

```python
GROUND_TRUTH_LLM_CONFIG = {
    "api_key": os.environ.get("GT_LLM_API_KEY", "ollama"),
    "base_url": os.environ.get("GT_LLM_BASE_URL", "http://localhost:11434/v1"),
    "model_name": os.environ.get("GT_LLM_MODEL", "llama3.1"),
    "temperature": 0.3,     # 低温度，确保准确性
    "max_tokens": 2048,
}
```

### Ground Truth 生成参数

```python
GROUND_TRUTH_CONFIG = {
    "num_entries": 11,       # 生成条目数
    "complexity": "medium",  # simple/medium/complex
    "enabled": True,         # 是否启用自动生成
}
```

### RAG 参数

```python
RAG_CONFIG = {
    "chunk_size": 1000,      # 文档块大小
    "chunk_overlap": 200,    # 块间重叠
    "top_k": 5,              # 检索返回数量
}
```

### 环境变量覆盖

```bash
# Ground Truth 生成 LLM（推荐使用 GPT-4 或 Claude）
export GT_LLM_API_KEY="your-key"
export GT_LLM_BASE_URL="https://api.openai.com/v1"
export GT_LLM_MODEL="gpt-4"

# 条目数量和复杂度
export GT_NUM_ENTRIES=20
export GT_COMPLEXITY=complex

# 禁用自动生成
export GT_ENABLED=false
```

## 📊 输出说明

### 结果文件（results/）

| 文件                          | 说明                                   |
| ----------------------------- | -------------------------------------- |
| `raw_outputs_*.csv`         | 原始模型输出（查询、迭代、响应、延迟） |
| `evaluation_stats_*.json`   | 结构化统计结果（均值+置信区间）        |
| `improvement_summary_*.csv` | RAG 改进百分比汇总                     |

### 图表文件（figures/）

| 文件                                | 对应论文 | 说明             |
| ----------------------------------- | -------- | ---------------- |
| `fig3_lexical_comparison.png`     | Figure 3 | BLEU/ROUGE 对比  |
| `fig4_semantic_comparison.png`    | Figure 4 | BERTScore 对比   |
| `table3_improvement_heatmap.png`  | Table 3  | 改进百分比热力图 |
| `fig5_cross_model_comparison.png` | Figure 5 | 跨模型对比       |
| `all_metrics_overview.png`        | —       | 综合概览         |

### 向量索引（VectorDB/）

| 文件              | 说明                 |
| ----------------- | -------------------- |
| `index.faiss`   | FAISS 向量索引文件   |
| `documents.pkl` | 原始文档块和向量矩阵 |

## 📖 文件详细说明

| 文件                              | 作用                | 核心 API                                                             |
| --------------------------------- | ------------------- | -------------------------------------------------------------------- |
| `config/model_config.py`        | 统一配置管理        | `LLM_CONFIGS`, `EMBEDDING_CONFIG`, `GROUND_TRUTH_CONFIG`       |
| `data/dataset_loader.py`        | PDF + XLSX 文档加载 | `load_document()`, `load_all_documents()`, `split_documents()` |
| `generate_ground_truth.py`      | Ground Truth 生成器 | `generate_ground_truth()`, `generate_qa_pair()`                  |
| `retrieval/retriever.py`        | FAISS 向量检索      | `FAISSRetriever` 类                                                |
| `models/llm_baseline.py`        | Baseline LLM        | `BaselineLLM` 类                                                   |
| `models/rag_pipeline.py`        | RAG 流程            | `RAGPipeline` 类                                                   |
| `evaluation/metrics.py`         | 评估指标            | `compute_bleu()`, `compute_rouge()`, `compute_bertscore()`     |
| `visualization/plot_results.py` | 图表生成            | `generate_all_plots()`                                             |
