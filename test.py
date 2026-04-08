from transformers import pipeline

# 1️⃣ 初始化 NLI pipeline
nli = pipeline("text-classification", model="roberta-large-mnli")

# 2️⃣ 示例
premise = "you should shut down the computer"      # 模型输出
hypothesis = "you should not shut down the computer"  # 参考答案

# 3️⃣ 调用 NLI 模型
result = nli(f"{premise} </s> {hypothesis}")

print(result)