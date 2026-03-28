"""
===========================================================
向量检索模块 (FAISS Vector Retriever Module)
===========================================================

功能说明：
    实现基于 FAISS 的向量检索系统，用于 RAG 流程中的文档检索。

    论文中的关键设计：
    1. 使用 "all-MiniLM-L6-v2" 模型将文本转换为 384 维向量
    2. 使用 FAISS 向量数据库存储和检索向量
    3. 每个查询返回 Top-5 最相似的文档块
    4. 基于余弦相似度（通过 L2 距离近似）排序

    FAISS (Facebook AI Similarity Search) 选择理由：
    - 支持高效的大规模向量搜索
    - 支持向量分类和索引分组，提高检索准确性
    - 内存效率高，适合处理大量文档块
"""

import os
import pickle
from typing import Optional

import faiss
import numpy as np

# sentence-transformers 用于文本向量化
from sentence_transformers import SentenceTransformer


class FAISSRetriever:
    """
    基于 FAISS 的向量检索器。

    工作流程：
    1. 初始化时加载 embedding 模型
    2. 对文档块进行向量化（encode_documents）
    3. 构建 FAISS 索引（build_index）
    4. 接收查询，返回最相似的 top-k 个文档块（search）

    属性:
        model: sentence-transformers 模型实例
        index: FAISS 索引对象
        documents: 原始文档块列表（用于返回检索结果的文本）
        embeddings: 文档的向量表示矩阵
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        use_api: bool = False,
        api_config: Optional[dict] = None
    ):
        """
        初始化检索器。

        参数:
            model_name (str): embedding 模型名称
                论文使用 "all-MiniLM-L6-v2"，该模型：
                - 输出 384 维向量
                - 速度快、体积小
                - 检索性能接近 BGE-Large 等大模型
            embedding_dim (int): 向量维度，默认 384
            use_api (bool): 是否使用 API 模式
            api_config (dict): API 模式的配置
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.use_api = use_api
        self.api_config = api_config

        # 存储原始文档和向量
        self.documents = []
        self.embeddings = None
        self.index = None

        # 加载 embedding 模型
        if not use_api:
            print(f"🔄 正在加载 Embedding 模型: {model_name}")
            # SentenceTransformer 会自动从 HuggingFace 下载模型
            # 首次运行需要网络连接，后续从缓存加载
            self.model = SentenceTransformer(model_name)
            print(f"✅ Embedding 模型加载完成")
        else:
            self.model = None
            print(f"🔄 使用 API 模式进行 Embedding")

    def encode_documents(self, documents: list, batch_size: int = 64) -> np.ndarray:
        """
        将文档块列表转换为向量矩阵。

        原理：
            使用 sentence-transformers 的 encode 方法将每个文档块
            映射到一个 384 维的稠密向量空间。在这个空间中，
            语义相似的文本会被映射到相近的位置。

            这是 RAG 流程的核心步骤之一：
            查询 → 向量化 → 在向量空间中找到最近的文档向量
            → 返回对应的原始文档文本

        参数:
            documents (list): 文档块列表，每个元素为字符串
            batch_size (int): 批量编码的批次大小
                较大的 batch_size 加速编码但占用更多内存

        返回:
            np.ndarray: 文档向量矩阵，形状为 (文档数, 向量维度)
        """
        # 保存原始文档，后续检索时需要返回文本
        self.documents = documents

        print(f"🔢 正在向量化 {len(documents)} 个文档块...")

        if not self.use_api:
            # 本地模式：使用 sentence-transformers 编码
            # show_progress_bar=True 显示编码进度
            # normalize_embeddings=True 对向量进行 L2 归一化
            # 归一化后，L2 距离等价于余弦距离
            self.embeddings = self.model.encode(
                documents,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,  # 归一化，使 L2 距离 ≈ 余弦距离
                convert_to_numpy=True
            )
        else:
            # API 模式：通过 API 获取 embeddings
            self.embeddings = self._encode_via_api(documents)

        # 确保数据类型为 float32（FAISS 要求）
        self.embeddings = self.embeddings.astype(np.float32)

        print(f"✅ 向量化完成，矩阵形状: {self.embeddings.shape}")
        return self.embeddings

    def _encode_via_api(self, documents: list) -> np.ndarray:
        """
        通过 API 获取文档的 embedding 向量。

        参数:
            documents (list): 文档块列表

        返回:
            np.ndarray: 文档向量矩阵
        """
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_config.get("api_key", ""),
            base_url=self.api_config.get("base_url", "")
        )

        all_embeddings = []
        # 分批处理，避免 API 请求过大
        batch_size = 32
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            response = client.embeddings.create(
                model=self.api_config.get("api_model_name", ""),
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def build_index(self, embeddings: Optional[np.ndarray] = None) -> faiss.Index:
        """
        构建 FAISS 索引。

        原理：
            FAISS 支持多种索引类型：
            - IndexFlatL2: 精确的 L2 距离搜索（暴力搜索）
            - IndexIVFFlat: 倒排文件索引，更快但近似
            - IndexHNSW: 层次化可导航小世界图，平衡速度和精度

            论文使用的数据规模（教科书级别）适合使用 IndexFlatL2，
            因为文档块数量不太大（通常几千个），精确搜索的延迟可接受。

            对于更大的数据集，可以切换到 IndexIVFFlat 或 IndexHNSW。

        参数:
            embeddings (np.ndarray, optional): 文档向量矩阵
                如果未提供，使用之前 encode_documents 生成的向量

        返回:
            faiss.Index: 构建好的 FAISS 索引
        """
        if embeddings is not None:
            self.embeddings = embeddings.astype(np.float32)
        elif self.embeddings is None:
            raise ValueError("❌ 没有可用的向量数据，请先调用 encode_documents()")

        # 获取向量维度
        dim = self.embeddings.shape[1]
        print(f"🏗️  正在构建 FAISS 索引 (维度: {dim}, 文档数: {self.embeddings.shape[0]})")

        # 使用 IndexFlatIP (内积)：因为向量已归一化，
        # 内积等价于余弦相似度，值越大越相似
        self.index = faiss.IndexFlatIP(dim)

        # 将所有文档向量添加到索引中
        self.index.add(self.embeddings)

        print(f"✅ FAISS 索引构建完成，包含 {self.index.ntotal} 个向量")
        return self.index

    def search(self, query: str, top_k: int = 5) -> list:
        """
        根据查询检索最相似的 top-k 个文档块。

        原理：
            1. 将查询文本向量化
            2. 在 FAISS 索引中搜索最近邻
            3. 返回 top-k 个最相似的文档块及其相似度得分

            论文中 top_k = 5，即每次检索返回最相似的 5 个段落。
            这些段落将被拼接成上下文，与查询一起输入 LLM 生成答案。

        参数:
            query (str): 查询文本
            top_k (int): 返回的文档数量，默认为 5（论文设定）

        返回:
            list: 检索结果列表，每个元素为字典：
                {
                    "text": 文档文本,
                    "score": 相似度得分 (越高越相似),
                    "index": 在文档列表中的索引
                }
        """
        if self.index is None:
            raise RuntimeError("❌ FAISS 索引未构建，请先调用 build_index()")

        # 步骤1: 将查询文本向量化
        if not self.use_api:
            query_embedding = self.model.encode(
                [query],
                normalize_embeddings=True,  # 与文档向量保持一致的归一化
                convert_to_numpy=True
            ).astype(np.float32)
        else:
            query_embedding = self._encode_via_api([query])

        # 步骤2: 在 FAISS 索引中搜索
        # distances: 相似度得分数组 (使用 IP 索引时，值越大越相似)
        # indices: 最近邻的索引数组
        distances, indices = self.index.search(query_embedding, top_k)

        # 步骤3: 组装检索结果
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents) and idx >= 0:
                results.append({
                    "text": self.documents[idx],     # 原始文档文本
                    "score": float(dist),            # 相似度得分
                    "index": int(idx),               # 在文档列表中的索引
                    "rank": i + 1                    # 排名（1-based）
                })

        return results

    def search_batch(self, queries: list, top_k: int = 5) -> list:
        """
        批量检索多个查询。

        参数:
            queries (list): 查询文本列表
            top_k (int): 每个查询返回的文档数量

        返回:
            list: 每个查询的检索结果列表的列表
        """
        if self.index is None:
            raise RuntimeError("❌ FAISS 索引未构建，请先调用 build_index()")

        # 批量向量化所有查询
        if not self.use_api:
            query_embeddings = self.model.encode(
                queries,
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype(np.float32)
        else:
            query_embeddings = self._encode_via_api(queries)

        # 批量搜索
        distances, indices = self.index.search(query_embeddings, top_k)

        # 组装结果
        all_results = []
        for q_idx in range(len(queries)):
            results = []
            for rank, (dist, doc_idx) in enumerate(
                zip(distances[q_idx], indices[q_idx])
            ):
                if doc_idx < len(self.documents) and doc_idx >= 0:
                    results.append({
                        "text": self.documents[doc_idx],
                        "score": float(dist),
                        "index": int(doc_idx),
                        "rank": rank + 1
                    })
            all_results.append(results)

        return all_results

    def save_index(self, save_dir: str = "VectorDB"):
        """
        将 FAISS 索引和文档数据持久化到磁盘。

        保存后可以避免每次运行都重新构建索引，节省时间。

        参数:
            save_dir (str): 保存目录路径
        """
        os.makedirs(save_dir, exist_ok=True)

        # 保存 FAISS 索引
        index_path = os.path.join(save_dir, "index.faiss")
        faiss.write_index(self.index, index_path)
        print(f"💾 FAISS 索引已保存: {index_path}")

        # 保存文档列表和向量矩阵
        data_path = os.path.join(save_dir, "documents.pkl")
        with open(data_path, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "embeddings": self.embeddings
            }, f)
        print(f"💾 文档数据已保存: {data_path}")

    def load_index(self, save_dir: str = "VectorDB") -> bool:
        """
        从磁盘加载已保存的 FAISS 索引和文档数据。

        参数:
            save_dir (str): 索引保存目录

        返回:
            bool: 加载成功返回 True，文件不存在返回 False
        """
        index_path = os.path.join(save_dir, "index.faiss")
        data_path = os.path.join(save_dir, "documents.pkl")

        if not os.path.exists(index_path) or not os.path.exists(data_path):
            print(f"⚠️  索引文件未找到: {save_dir}")
            return False

        # 加载 FAISS 索引
        self.index = faiss.read_index(index_path)
        print(f"📂 FAISS 索引已加载: {self.index.ntotal} 个向量")

        # 加载文档数据
        with open(data_path, "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.embeddings = data["embeddings"]
        print(f"📂 文档数据已加载: {len(self.documents)} 个文档块")

        return True

    def get_context_string(self, results: list, separator: str = "\n\n---\n\n") -> str:
        """
        将检索结果拼接为上下文字符串，用于输入 LLM。

        论文中，检索到的 top-5 段落被拼接后与查询一起
        作为 prompt 输入 LLM。

        参数:
            results (list): search() 返回的检索结果列表
            separator (str): 段落间的分隔符

        返回:
            str: 拼接后的上下文字符串
        """
        passages = [
            f"[Passage {r['rank']}] (relevance score: {r['score']:.4f})\n{r['text']}"
            for r in results
        ]
        return separator.join(passages)


# =============================================
# 单元测试 / 功能演示
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("FAISS 检索器功能测试")
    print("=" * 60)

    # 创建测试文档
    test_documents = [
        "Carbohydrates are the body's primary source of energy. They are broken down into glucose.",
        "Proteins are essential for building and repairing tissues. They are made of amino acids.",
        "Vitamins are organic compounds needed in small quantities for proper body function.",
        "Iron is an essential mineral that plays a key role in oxygen transport via hemoglobin.",
        "Dietary fiber aids digestion and helps maintain bowel health. It comes from plant foods.",
        "Calcium is crucial for strong bones and teeth. It also supports muscle function.",
        "Water makes up about 60% of body weight and is essential for all cellular processes.",
        "Lipids serve as energy storage and are important components of cell membranes.",
    ]

    # 初始化检索器
    retriever = FAISSRetriever(model_name="all-MiniLM-L6-v2")

    # 向量化文档
    embeddings = retriever.encode_documents(test_documents)

    # 构建索引
    retriever.build_index()

    # 执行检索测试
    test_query = "What is the role of iron in the human body?"
    results = retriever.search(test_query, top_k=3)

    print(f"\n🔍 查询: {test_query}")
    print(f"\n📋 检索结果:")
    for r in results:
        print(f"  [{r['rank']}] score={r['score']:.4f}: {r['text'][:80]}...")

    # 测试上下文拼接
    context = retriever.get_context_string(results)
    print(f"\n📝 拼接上下文 (前200字符):")
    print(context[:200])
