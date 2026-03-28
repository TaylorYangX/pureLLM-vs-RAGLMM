"""
===========================================================
Baseline LLM 调用模块 (Baseline LLM Module)
===========================================================

功能说明：
    实现 Baseline LLM（无检索增强）的 API 调用。

    论文中的 Baseline LLM 直接接收用户查询，
    仅依靠模型自身的参数知识生成答案，
    不使用任何外部检索信息。

    这与 RAG-Augmented LLM 形成对比：
    - Baseline: query → LLM → answer
    - RAG:      query → retriever → context + query → LLM → answer

    支持 OpenAI 兼容的 API 格式，可对接：
    - Ollama（本地部署）
    - vLLM（自托管推理服务）
    - Together AI / OpenRouter（云端 API）
"""

import time
from typing import Optional

from openai import OpenAI

# 导入配置模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.model_config import get_llm_config


class BaselineLLM:
    """
    Baseline LLM 调用器。

    功能：
        - 通过 OpenAI 兼容 API 调用 LLM
        - 支持多种模型配置
        - 内置重试机制和错误处理
        - 记录每次调用的耗时

    设计原则：
        - 与 RAG 管道使用相同的 API 接口
        - 输出格式统一，便于后续评估比较
    """

    def __init__(self, model_key: str):
        """
        初始化 Baseline LLM。

        参数:
            model_key (str): 模型配置的键名
                如 "tinyllama", "mistral", "llama3.1", "llama1-13b"
        """
        self.model_key = model_key

        # 从配置模块获取该模型的配置
        self.config = get_llm_config(model_key)

        # 创建 OpenAI 兼容的客户端
        # 通过修改 base_url，可以连接到任何 OpenAI 兼容的 API
        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["base_url"]
        )

        print(f"🤖 Baseline LLM 初始化: {model_key}")
        print(f"   模型: {self.config['model_name']}")
        print(f"   API: {self.config['base_url']}")

    def generate(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> dict:
        """
        使用 Baseline LLM 生成回答。

        原理：
            Baseline 模式下，LLM 仅依靠自身训练阶段学到的知识
            （参数知识/parametric knowledge）来回答问题。
            不提供任何外部上下文或检索信息。

            这是论文实验的对照组，用于衡量 RAG 增强的效果。

        参数:
            query (str): 用户查询
            system_prompt (str, optional): 系统提示词
                定义模型的角色和行为规范
            max_retries (int): API 调用失败时的最大重试次数
            retry_delay (float): 重试间隔（秒）

        返回:
            dict: 包含以下字段：
                - "response": 模型生成的回答文本
                - "model": 使用的模型名称
                - "latency": API 调用耗时（秒）
                - "success": 是否成功
                - "error": 错误信息（如果失败）
        """
        # 构建消息列表
        messages = []

        # 系统提示词：定义模型的角色
        if system_prompt is None:
            system_prompt = (
                "You are a knowledgeable assistant. "
                "Answer the following question accurately and comprehensively "
                "based on your knowledge."
            )
        messages.append({"role": "system", "content": system_prompt})

        # 用户查询
        messages.append({"role": "user", "content": query})

        # 带重试的 API 调用
        for attempt in range(max_retries):
            try:
                # 记录开始时间，用于计算延迟
                start_time = time.time()

                # 调用 OpenAI 兼容 API
                response = self.client.chat.completions.create(
                    model=self.config["model_name"],
                    messages=messages,
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_tokens"],
                )

                # 计算调用耗时
                latency = time.time() - start_time

                # 提取回答文本
                answer = response.choices[0].message.content.strip()

                return {
                    "response": answer,
                    "model": self.model_key,
                    "latency": latency,
                    "success": True,
                    "error": None
                }

            except Exception as e:
                print(
                    f"⚠️  API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    # 指数退避策略：每次重试等待更长时间
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"   等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)

        # 所有重试都失败
        return {
            "response": "",
            "model": self.model_key,
            "latency": 0,
            "success": False,
            "error": f"API 调用在 {max_retries} 次重试后仍然失败"
        }

    def generate_batch(
        self,
        queries: list,
        num_iterations: int = 1,
        system_prompt: Optional[str] = None
    ) -> list:
        """
        批量生成回答：对每个查询执行多次迭代。

        原理：
            论文中每个查询执行 11 次迭代，以：
            1. 捕捉模型输出的变异性（variability）
            2. 确保实验结果的可重复性
            3. 为统计分析（如置信区间）提供足够的样本

        参数:
            queries (list): 查询列表
            num_iterations (int): 每个查询的迭代次数（论文中为 11）
            system_prompt (str, optional): 系统提示词

        返回:
            list: 结果列表，每个元素包含：
                - query_id: 查询编号
                - query: 查询文本
                - iteration: 迭代编号
                - response: 模型回答
                - latency: 延迟
                - success: 是否成功
        """
        results = []
        total = len(queries) * num_iterations
        count = 0

        for q_idx, query_text in enumerate(queries):
            for iteration in range(num_iterations):
                count += 1
                print(
                    f"\r  [{self.model_key}] "
                    f"进度: {count}/{total} "
                    f"(Q{q_idx + 1}, Iter {iteration + 1})",
                    end=""
                )

                # 调用 LLM 生成回答
                result = self.generate(query_text, system_prompt=system_prompt)

                # 附加元数据
                result["query_id"] = q_idx + 1
                result["query"] = query_text
                result["iteration"] = iteration + 1
                result["mode"] = "baseline"

                results.append(result)

        print()  # 换行
        return results


# =============================================
# 单元测试 / 功能演示
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("Baseline LLM 功能测试")
    print("=" * 60)
    print("\n⚠️  注意：此测试需要运行中的 Ollama 服务")
    print("   启动 Ollama: ollama serve")
    print("   拉取模型: ollama pull mistral\n")

    try:
        # 使用 Mistral 进行测试
        llm = BaselineLLM("mistral")

        # 单次调用测试
        result = llm.generate("What are the main macronutrients?")

        if result["success"]:
            print(f"\n✅ 生成成功!")
            print(f"   耗时: {result['latency']:.2f}s")
            print(f"   回答: {result['response'][:200]}...")
        else:
            print(f"\n❌ 生成失败: {result['error']}")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("请确认 Ollama 服务已启动并部署了对应模型。")
