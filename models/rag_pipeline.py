"""
===========================================================
RAG Pipeline 模块 (Retrieval-Augmented Generation Module)
===========================================================

功能说明：
    完整的 RAG（检索增强生成）流程实现。

    论文中 RAG 的工作流程分为三个阶段：
    1. 数据预处理和索引 (Data Preprocessing & Indexing)
       - PDF 文档加载
       - 文本切分（chunking）
       - 向量化（embedding）→ 存入 FAISS 索引

    2. 上下文检索 (Contextual Retrieval)
       - 接收查询
       - 在 FAISS 中搜索 top-5 最相似段落
       - 按相似度排序

    3. 内容生成 (Content Generation)
       - 将检索到的段落与查询组合成 prompt
       - LangChain 风格：query → retriever → context + query → LLM → answer
       - 输出解析器确保返回纯文本

    RAG 模式 vs Baseline 模式的关键区别：
    - Baseline: query → LLM → answer（仅靠参数知识）
    - RAG:      query → retriever → context + query → LLM → answer（有外部知识增强）
"""

import time
from typing import Optional

from openai import OpenAI

# 导入项目模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import get_llm_config, RAG_CONFIG
from retrieval.retriever import FAISSRetriever
from data.dataset_loader import load_pdf_document, split_documents


class RAGPipeline:
    """
    RAG (Retrieval-Augmented Generation) 完整流程。

    实现论文中描述的 LangChain 风格 RAG 管道：
    1. 文档加载 → 切分 → 向量化 → 建立 FAISS 索引
    2. 查询 → 向量检索 Top-5 → 拼接上下文
    3. 上下文 + 查询 → LLM 生成 → 解析输出

    属性:
        retriever: FAISS 向量检索器实例
        model_key: 使用的 LLM 模型键名
        config: LLM 配置
        client: OpenAI 兼容的 API 客户端
        top_k: 检索返回的文档数量
        prompt_template: RAG 提示模板
        is_index_built: 索引是否已构建
    """

    def __init__(
        self,
        model_key: str,
        retriever: Optional[FAISSRetriever] = None,
        top_k: int = None,
        prompt_template: str = None
    ):
        """
        初始化 RAG Pipeline。

        参数:
            model_key (str): LLM 模型的配置键名
            retriever (FAISSRetriever, optional): 预构建的检索器
                如果为 None，将在 build_index 时创建
            top_k (int, optional): 检索返回的文档数量
                默认使用 RAG_CONFIG 中的设置（论文中为 5）
            prompt_template (str, optional): 自定义的 RAG 提示模板
                默认使用 RAG_CONFIG 中的模板
        """
        self.model_key = model_key
        self.config = get_llm_config(model_key)

        # 创建 OpenAI 兼容 API 客户端
        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["base_url"]
        )

        # 检索器（可以共享同一个检索器实例，避免重复构建索引）
        self.retriever = retriever

        # RAG 参数
        self.top_k = top_k or RAG_CONFIG["top_k"]  # 默认 5

        # 提示模板
        # 论文中实现 LangChain 风格：检索上下文与查询组合
        self.prompt_template = prompt_template or RAG_CONFIG["prompt_template"]

        # 状态标志
        self.is_index_built = retriever is not None and retriever.index is not None

        print(f"🔗 RAG Pipeline 初始化: {model_key}")
        print(f"   模型: {self.config['model_name']}")
        print(f"   Top-K: {self.top_k}")
        print(f"   索引状态: {'已构建' if self.is_index_built else '未构建'}")

    def build_index_from_documents(
        self,
        documents: list,
        embedding_model: str = "all-MiniLM-L6-v2",
        save_path: Optional[str] = None
    ):
        """
        从文档块列表构建 FAISS 索引。

        原理：
            这是 RAG 流程的第一阶段（数据预处理和索引）：
            1. 创建 embedding 模型实例
            2. 将所有文档块转换为向量
            3. 构建 FAISS 索引
            4. 可选：将索引保存到磁盘

        参数:
            documents (list): 文档块列表
            embedding_model (str): embedding 模型名称
                论文使用 "all-MiniLM-L6-v2"
            save_path (str, optional): 索引保存路径
        """
        print(f"\n📦 构建 RAG 索引...")

        # 如果还没有检索器，创建一个新的
        if self.retriever is None:
            self.retriever = FAISSRetriever(model_name=embedding_model)

        # 步骤1: 向量化所有文档块
        self.retriever.encode_documents(documents)

        # 步骤2: 构建 FAISS 索引
        self.retriever.build_index()

        # 可选步骤: 保存索引到磁盘
        if save_path:
            self.retriever.save_index(save_path)

        self.is_index_built = True
        print(f"✅ RAG 索引构建完成")

    def build_index_from_pdf(
        self,
        pdf_path: str,
        chunk_size: int = None,
        chunk_overlap: int = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        save_path: Optional[str] = None
    ) -> list:
        """
        从 PDF 文件构建完整的 RAG 索引。

        将数据预处理和索引构建合并为一个便捷方法：
        PDF → 提取文本 → 切分 → 向量化 → 建立 FAISS 索引

        参数:
            pdf_path (str): PDF 文件路径
            chunk_size (int, optional): 文档块大小
            chunk_overlap (int, optional): 重叠大小
            embedding_model (str): embedding 模型名称
            save_path (str, optional): 索引保存路径

        返回:
            list: 切分后的文档块列表
        """
        # 使用 RAG_CONFIG 中的默认值
        chunk_size = chunk_size or RAG_CONFIG["chunk_size"]
        chunk_overlap = chunk_overlap or RAG_CONFIG["chunk_overlap"]

        # 步骤1: 从 PDF 中提取文本
        print(f"\n📖 从 PDF 构建 RAG 索引: {pdf_path}")
        full_text = load_pdf_document(pdf_path)

        # 步骤2: 切分文本为块
        chunks = split_documents(
            text=full_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # 步骤3: 构建索引
        self.build_index_from_documents(
            documents=chunks,
            embedding_model=embedding_model,
            save_path=save_path
        )

        return chunks

    def retrieve(self, query: str) -> list:
        """
        为给定查询检索最相关的文档。

        对应论文中 RAG 的第二阶段：上下文检索（Contextual Retrieval）。
        使用 FAISS 搜索引擎找到与查询语义最相似的 top-k 个文档块。

        参数:
            query (str): 查询文本

        返回:
            list: 检索结果列表，按相似度降序排列
        """
        if not self.is_index_built:
            raise RuntimeError(
                "❌ FAISS 索引未构建。"
                "请先调用 build_index_from_documents() 或 build_index_from_pdf()"
            )

        # 调用 FAISS 检索器执行搜索
        results = self.retriever.search(query, top_k=self.top_k)
        return results

    def _build_rag_prompt(self, query: str, context: str) -> str:
        """
        构建 RAG 提示（将检索上下文与查询组合）。

        原理：
            论文中实现 LangChain 风格的 RAG 管道：
            - 检索器获取相关上下文
            - 将上下文与查询组合成结构化 prompt
            - 确保 LLM 基于检索到的信息生成答案

            prompt 模板的设计直接影响生成质量：
            - 明确告知模型使用提供的上下文
            - 引导模型给出准确、全面的回答
            - 通过 "Based on the provided context" 限制幻觉

        参数:
            query (str): 用户查询
            context (str): 检索到的上下文文本

        返回:
            str: 组合后的完整提示
        """
        # 使用模板格式化 prompt
        prompt = self.prompt_template.format(
            context=context,
            question=query
        )
        return prompt

    def generate(
        self,
        query: str,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> dict:
        """
        RAG 完整流程：检索 + 生成。

        这是 RAG 的核心方法，实现论文中描述的完整流程：
        1. 接收查询
        2. 检索 top-k 个相关文档
        3. 拼接检索到的文档作为上下文
        4. 将上下文 + 查询组合成 prompt
        5. 调用 LLM 生成回答
        6. 解析输出，返回纯文本

        参数:
            query (str): 用户查询
            max_retries (int): API 调用失败时的最大重试次数
            retry_delay (float): 重试间隔（秒）

        返回:
            dict: 包含以下字段：
                - "response": 模型生成的回答文本
                - "model": 使用的模型名称
                - "mode": "rag"
                - "retrieved_docs": 检索到的文档列表
                - "context": 拼接的上下文文本
                - "latency": 总耗时（秒）
                - "retrieval_latency": 检索耗时
                - "generation_latency": 生成耗时
                - "success": 是否成功
                - "error": 错误信息（如果失败）
        """
        total_start = time.time()

        # ========== 阶段1: 上下文检索 ==========
        retrieval_start = time.time()
        try:
            # 调用 FAISS 检索器搜索最相关的 top-k 个文档
            retrieved_docs = self.retrieve(query)
        except Exception as e:
            return {
                "response": "",
                "model": self.model_key,
                "mode": "rag",
                "retrieved_docs": [],
                "context": "",
                "latency": time.time() - total_start,
                "retrieval_latency": 0,
                "generation_latency": 0,
                "success": False,
                "error": f"检索失败: {e}"
            }
        retrieval_latency = time.time() - retrieval_start

        # 将检索结果拼接为上下文字符串
        context = self.retriever.get_context_string(retrieved_docs)

        # ========== 阶段2: 内容生成 ==========
        # 构建 RAG prompt：上下文 + 查询
        rag_prompt = self._build_rag_prompt(query, context)

        # 构建消息列表
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable assistant. "
                    "Answer questions based on the provided context. "
                    "Be accurate and comprehensive."
                )
            },
            {
                "role": "user",
                "content": rag_prompt
            }
        ]

        # 带重试的 API 调用
        generation_start = time.time()
        for attempt in range(max_retries):
            try:
                # 调用 LLM API
                response = self.client.chat.completions.create(
                    model=self.config["model_name"],
                    messages=messages,
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_tokens"],
                )

                generation_latency = time.time() - generation_start
                total_latency = time.time() - total_start

                # 提取并清理回答文本（输出解析器：确保纯文本）
                answer = response.choices[0].message.content.strip()

                return {
                    "response": answer,
                    "model": self.model_key,
                    "mode": "rag",
                    "retrieved_docs": retrieved_docs,
                    "context": context,
                    "latency": total_latency,
                    "retrieval_latency": retrieval_latency,
                    "generation_latency": generation_latency,
                    "success": True,
                    "error": None
                }

            except Exception as e:
                print(
                    f"⚠️  RAG 生成失败 (尝试 {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"   等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)

        # 所有重试都失败
        return {
            "response": "",
            "model": self.model_key,
            "mode": "rag",
            "retrieved_docs": retrieved_docs,
            "context": context,
            "latency": time.time() - total_start,
            "retrieval_latency": retrieval_latency,
            "generation_latency": 0,
            "success": False,
            "error": f"LLM API 调用在 {max_retries} 次重试后失败"
        }

    def generate_batch(
        self,
        queries: list,
        num_iterations: int = 1
    ) -> list:
        """
        批量 RAG 生成：对每个查询执行多次迭代。

        原理：
            论文中每个查询执行 11 次迭代以：
            1. 捕捉模型输出的变异性
            2. 为统计分析提供样本
            3. 确保可重复性

            每次迭代使用相同的查询，但 LLM 由于 temperature > 0
            会产生不同的输出。

        参数:
            queries (list): 查询列表
            num_iterations (int): 每个查询的迭代次数

        返回:
            list: 结果列表
        """
        results = []
        total = len(queries) * num_iterations
        count = 0

        for q_idx, query_text in enumerate(queries):
            for iteration in range(num_iterations):
                count += 1
                print(
                    f"\r  [RAG-{self.model_key}] "
                    f"进度: {count}/{total} "
                    f"(Q{q_idx + 1}, Iter {iteration + 1})",
                    end=""
                )

                # 执行完整的 RAG 流程
                result = self.generate(query_text)

                # 附加实验元数据
                result["query_id"] = q_idx + 1
                result["query"] = query_text
                result["iteration"] = iteration + 1

                results.append(result)

        print()  # 换行
        return results


# =============================================
# 单元测试 / 功能演示
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("RAG Pipeline 功能测试")
    print("=" * 60)
    print("\n⚠️  注意：此测试需要运行中的 Ollama 服务")
    print("   启动 Ollama: ollama serve")
    print("   拉取模型: ollama pull mistral\n")

    # 创建测试文档
    test_documents = [
        "Carbohydrates are the body's primary source of energy. They are broken down into glucose, which fuels cellular activity.",
        "Proteins are essential macronutrients for building and repairing tissues. They consist of amino acids linked by peptide bonds.",
        "Vitamins are organic compounds required in small quantities. They are classified as water-soluble or fat-soluble.",
        "Iron is an essential mineral for oxygen transport. Hemoglobin in red blood cells contains iron.",
        "Dietary fiber promotes digestive health. Soluble fiber forms a gel-like substance, while insoluble fiber adds bulk to stool.",
        "Calcium is the most abundant mineral in the body. About 99% is stored in bones and teeth.",
        "Water is essential for all cellular processes. It regulates body temperature and transports nutrients.",
        "Lipids are important for energy storage and cell membrane structure. They include triglycerides, phospholipids, and sterols.",
    ]

    try:
        # 初始化 RAG Pipeline
        rag = RAGPipeline(model_key="mistral")

        # 构建索引
        rag.build_index_from_documents(test_documents)

        # 测试检索
        query = "What is the role of iron in the human body?"
        print(f"\n🔍 测试查询: {query}")

        retrieved = rag.retrieve(query)
        print(f"\n📋 检索到 {len(retrieved)} 个文档:")
        for doc in retrieved:
            print(f"  [{doc['rank']}] score={doc['score']:.4f}: {doc['text'][:60]}...")

        # 测试完整 RAG 生成
        result = rag.generate(query)

        if result["success"]:
            print(f"\n✅ RAG 生成成功!")
            print(f"   检索耗时: {result['retrieval_latency']:.2f}s")
            print(f"   生成耗时: {result['generation_latency']:.2f}s")
            print(f"   总耗时: {result['latency']:.2f}s")
            print(f"   回答: {result['response'][:200]}...")
        else:
            print(f"\n❌ RAG 生成失败: {result['error']}")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("请确认 Ollama 服务已启动并部署了对应模型。")
