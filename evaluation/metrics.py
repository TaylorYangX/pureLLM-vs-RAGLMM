"""
===========================================================
评估指标模块 (Evaluation Metrics Module)
===========================================================

功能说明：
    实现论文中使用的所有 7 个评估指标，分为两类：

    词汇相似度指标 (Lexical Similarity):
        1. BLEU  — n-gram 精确度，衡量翻译/生成质量
        2. ROUGE-1 — 单字（unigram）召回率
        3. ROUGE-2 — 双字（bigram）召回率
        4. ROUGE-L — 最长公共子序列召回率

    语义相似度指标 (Semantic Similarity):
        5. BERTScore Precision — 语义精确度
        6. BERTScore Recall — 语义召回率
        7. BERTScore F1 — 精确度和召回率的调和平均

    每个指标的值域为 [0, 1]：
        0 = 完全不匹配
        1 = 完全匹配

实现原则：
    - 使用开源库实现，不调用黑盒 API
    - 每个函数清楚标注输入、输出、计算逻辑
    - 包含 90% 置信区间计算（论文要求）
"""

import numpy as np
from scipy import stats

# ---- BLEU 相关 ----
# nltk 的 BLEU 实现支持 n-gram 精确度和平滑方法
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ---- ROUGE 相关 ----
# google 的 rouge-score 库提供工业级 ROUGE 实现
from rouge_score import rouge_scorer

# ---- BERTScore 相关 ----
# bert-score 库使用预训练 BERT 模型计算语义相似度
from bert_score import score as bert_score_fn


# 确保 NLTK 数据已下载
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


# =============================================
# 1. BLEU Score 计算
# =============================================

def compute_bleu(reference: str, candidate: str) -> float:
    """
    计算 BLEU（Bilingual Evaluation Understudy）分数。

    原理：
        BLEU 衡量候选文本与参考文本之间的 n-gram 精确度。
        具体计算步骤：
        1. 将参考文本和候选文本分词
        2. 计算 1-gram, 2-gram, 3-gram, 4-gram 的精确度
        3. 取几何平均（各 n-gram 精确度的权重相等，默认每项 0.25）
        4. 应用简短惩罚（brevity penalty）：
           如果候选文本比参考文本短，则降低分数

        论文指出：
        - BLEU 精确衡量 n-gram 重叠
        - 惩罚缺失或错误的片段
        - 是机器翻译中的标准指标
        - 缺点：可能奖励过短的输出（忽略召回率）

    输入:
        reference (str): 参考文本（ground truth答案）
        candidate (str): 候选文本（模型生成的答案）

    输出:
        float: BLEU 分数，范围 [0, 1]

    计算逻辑:
        BLEU = BP × exp( Σ(wn × log(pn)) )
        其中:
            BP = min(1, exp(1 - len(ref)/len(cand)))  # 简短惩罚
            pn = n-gram 精确度（裁剪后）
            wn = 各 n-gram 的权重（默认均为 0.25）
    """
    # 步骤1: 分词
    # 使用简单的空格分词，因为我们处理的是英文文本
    reference_tokens = reference.lower().split()
    candidate_tokens = candidate.lower().split()

    # 处理边界情况：空文本
    if not candidate_tokens or not reference_tokens:
        return 0.0

    # 步骤2: 使用平滑方法避免 n-gram 计数为 0 导致的零分数
    # SmoothingFunction().method1 添加平滑常数 epsilon
    # 这对于短文本尤其重要，因为高阶 n-gram 可能完全不匹配
    smoother = SmoothingFunction().method1

    # 步骤3: 计算 BLEU 分数
    # 参考文本需要是列表的列表（支持多个参考）
    # 权重 (0.25, 0.25, 0.25, 0.25) 表示 1-4 gram 等权
    try:
        bleu = sentence_bleu(
            [reference_tokens],      # 参考文本（包裹在列表中）
            candidate_tokens,         # 候选文本
            weights=(0.25, 0.25, 0.25, 0.25),  # 1-4 gram 等权
            smoothing_function=smoother  # 平滑方法
        )
    except Exception:
        # 如果计算失败（例如文本过短），返回 0
        bleu = 0.0

    return bleu


# =============================================
# 2. ROUGE Score 计算
# =============================================

def compute_rouge(reference: str, candidate: str) -> dict:
    """
    计算 ROUGE（Recall-Oriented Understudy for Gisting Evaluation）分数。

    原理：
        ROUGE 是面向召回率的评估指标，衡量参考文本中有多少内容
        被候选文本覆盖。论文使用三个变体：

        ROUGE-1 (Unigram):
            计算单词级别的召回率
            ROUGE-1 = |参考词 ∩ 候选词| / |参考词|
            意义：衡量模型覆盖了参考答案中多少关键词

        ROUGE-2 (Bigram):
            计算连续两个词组合的召回率
            ROUGE-2 = |参考双词组 ∩ 候选双词组| / |参考双词组|
            意义：捕捉局部流畅性和短语级别重叠

        ROUGE-L (Longest Common Subsequence):
            基于最长公共子序列的召回率
            意义：衡量保持正确顺序的最长匹配词序列
            不要求连续，但要求顺序一致

    输入:
        reference (str): 参考文本（ground truth答案）
        candidate (str): 候选文本（模型生成的答案）

    输出:
        dict: 包含以下键的字典：
            - "rouge1": ROUGE-1 F1 分数
            - "rouge2": ROUGE-2 F1 分数
            - "rougeL": ROUGE-L F1 分数
            每个值范围为 [0, 1]

    计算逻辑:
        对于 ROUGE-N:
            Precision = |匹配的 n-grams| / |候选 n-grams|
            Recall = |匹配的 n-grams| / |参考 n-grams|
            F1 = 2 × P × R / (P + R)
        论文中使用 F-measure（F1），平衡精确度和召回率
    """
    # 创建 ROUGE 评分器
    # 指定要计算的 ROUGE 变体
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True  # 使用词干化，"running" 和 "run" 视为匹配
    )

    # 处理边界情况
    if not reference.strip() or not candidate.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    # 计算 ROUGE 分数
    scores = scorer.score(reference, candidate)

    # 提取 F-measure (F1 分数)
    # rouge_scorer 返回 precision, recall, fmeasure 三个值
    # 这里使用 fmeasure，与论文的评估方式一致
    return {
        "rouge1": scores['rouge1'].fmeasure,  # ROUGE-1 F1
        "rouge2": scores['rouge2'].fmeasure,  # ROUGE-2 F1
        "rougeL": scores['rougeL'].fmeasure,  # ROUGE-L F1
    }


# =============================================
# 3. BERTScore 计算
# =============================================

def compute_bertscore(
    references: list,
    candidates: list,
    model_type: str = "microsoft/deberta-xlarge-mnli",
    lang: str = "en",
    batch_size: int = 32
) -> dict:
    """
    计算 BERTScore（基于 BERT 的语义相似度评分）。

    原理：
        BERTScore 使用预训练的上下文嵌入模型（如 BERT/DeBERTa）
        来计算语义相似度，超越了表面词汇匹配。

        计算步骤：
        1. 对参考文本和候选文本中的每个 token 生成上下文嵌入
        2. 计算所有 token 对之间的余弦相似度矩阵
        3. 通过贪心匹配计算精确度、召回率和 F1

        BERTScore Precision:
            候选文本中的每个 token 与参考文本中最相似 token 的
            平均余弦相似度。衡量"生成的内容是否语义正确"。

        BERTScore Recall:
            参考文本中的每个 token 与候选文本中最相似 token 的
            平均余弦相似度。衡量"参考内容是否被充分覆盖"。

        BERTScore F1:
            精确度和召回率的调和平均。提供综合评估。

        论文指出：
        - BER Precision 检查 LLM 的答案是否语义正确
        - BERT Recall 确保 LLM 覆盖了参考答案的完整范围
        - BERT F1 平衡了完整性和准确性

    输入:
        references (list): 参考文本列表
        candidates (list): 候选文本列表
        model_type (str): 使用的 BERT 模型
            默认 "microsoft/deberta-xlarge-mnli"（bert_score 推荐）
        lang (str): 语言代码
        batch_size (int): 批处理大小

    输出:
        dict: 包含以下键：
            - "precision": 精确度列表（每个样本一个值）
            - "recall": 召回率列表
            - "f1": F1 分数列表

    计算逻辑:
        对于每对 (candidate, reference):
            P_BERT = (1/|c|) Σ_max_j cos(c_i, r_j)  ∀ c_i ∈ candidate
            R_BERT = (1/|r|) Σ_max_i cos(c_i, r_j)  ∀ r_j ∈ reference
            F1_BERT = 2 × P_BERT × R_BERT / (P_BERT + R_BERT)
    """
    # 验证输入
    if len(references) != len(candidates):
        raise ValueError(
            f"参考文本和候选文本数量不匹配: "
            f"{len(references)} vs {len(candidates)}"
        )

    if not references:
        return {"precision": [], "recall": [], "f1": []}

    print(f"🧠 正在计算 BERTScore ({len(references)} 个样本)...")

    # 调用 bert_score 库计算
    # verbose=True 显示进度条
    # rescale_with_baseline=False: 不进行基线校正
    #   论文中直接使用原始分数，不进行额外校正
    P, R, F1 = bert_score_fn(
        candidates,
        references,
        model_type=model_type,
        lang=lang,
        batch_size=batch_size,
        verbose=True,
        rescale_with_baseline=False  # 使用原始分数
    )

    # 将 PyTorch tensor 转换为 Python list
    precision_list = P.tolist()
    recall_list = R.tolist()
    f1_list = F1.tolist()

    print(f"✅ BERTScore 计算完成")
    print(f"   平均 Precision: {np.mean(precision_list):.4f}")
    print(f"   平均 Recall: {np.mean(recall_list):.4f}")
    print(f"   平均 F1: {np.mean(f1_list):.4f}")

    return {
        "precision": precision_list,
        "recall": recall_list,
        "f1": f1_list
    }


def compute_bertscore_single(
    reference: str,
    candidate: str,
    model_type: str = "microsoft/deberta-xlarge-mnli",
    lang: str = "en"
) -> dict:
    """
    计算单个样本的 BERTScore。

    参数:
        reference (str): 参考文本
        candidate (str): 候选文本
        model_type (str): BERT 模型类型
        lang (str): 语言

    返回:
        dict: {"precision": float, "recall": float, "f1": float}
    """
    result = compute_bertscore(
        [reference], [candidate],
        model_type=model_type,
        lang=lang
    )
    return {
        "precision": result["precision"][0],
        "recall": result["recall"][0],
        "f1": result["f1"][0]
    }


# =============================================
# 4. 综合指标计算
# =============================================

def compute_all_metrics(reference: str, candidate: str) -> dict:
    """
    计算单个样本的所有 7 个评估指标。

    将 BLEU、ROUGE（3个变体）和 BERTScore（3个维度）
    统一计算并返回。

    输入:
        reference (str): 参考文本（ground truth答案）
        candidate (str): 候选文本（模型生成的答案）

    输出:
        dict: 包含 7 个指标的字典：
            {
                "bleu": float,
                "rouge1": float,
                "rouge2": float,
                "rougeL": float,
                "bert_precision": float,
                "bert_recall": float,
                "bert_f1": float
            }
    """
    # 计算词汇相似度指标
    bleu = compute_bleu(reference, candidate)
    rouge = compute_rouge(reference, candidate)

    # 计算语义相似度指标
    bert_scores = compute_bertscore_single(reference, candidate)

    return {
        "bleu": bleu,
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "bert_precision": bert_scores["precision"],
        "bert_recall": bert_scores["recall"],
        "bert_f1": bert_scores["f1"],
    }


def compute_all_metrics_batch(
    references: list,
    candidates: list
) -> list:
    """
    批量计算所有指标。

    对于 BERTScore 使用批量计算以提高效率
    （避免重复加载模型）。

    输入:
        references (list): 参考文本列表
        candidates (list): 候选文本列表

    输出:
        list: 每个样本的指标字典列表
    """
    if len(references) != len(candidates):
        raise ValueError("参考文本和候选文本数量不匹配")

    n = len(references)
    print(f"\n📊 批量计算 {n} 个样本的评估指标...")

    # ---- 词汇指标：逐个计算 ----
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for i, (ref, cand) in enumerate(zip(references, candidates)):
        # 计算 BLEU
        bleu_scores.append(compute_bleu(ref, cand))

        # 计算 ROUGE
        rouge = compute_rouge(ref, cand)
        rouge1_scores.append(rouge["rouge1"])
        rouge2_scores.append(rouge["rouge2"])
        rougeL_scores.append(rouge["rougeL"])

    # ---- 语义指标：批量计算 ----
    # 批量计算更高效，因为 BERT 模型只需加载一次
    bert_results = compute_bertscore(references, candidates)

    # ---- 组装结果 ----
    results = []
    for i in range(n):
        results.append({
            "bleu": bleu_scores[i],
            "rouge1": rouge1_scores[i],
            "rouge2": rouge2_scores[i],
            "rougeL": rougeL_scores[i],
            "bert_precision": bert_results["precision"][i],
            "bert_recall": bert_results["recall"][i],
            "bert_f1": bert_results["f1"][i],
        })

    # 打印汇总统计
    print(f"\n📊 评估指标汇总:")
    metric_names = ["bleu", "rouge1", "rouge2", "rougeL",
                    "bert_precision", "bert_recall", "bert_f1"]
    for name in metric_names:
        values = [r[name] for r in results]
        print(f"   {name}: mean={np.mean(values):.4f}, "
              f"std={np.std(values):.4f}")

    return results


# =============================================
# 5. 统计分析：置信区间
# =============================================

def compute_confidence_interval(
    scores: list,
    confidence: float = 0.90
) -> dict:
    """
    计算给定分数列表的置信区间。

    原理：
        论文使用 90% 置信区间来评估指标估计值的可靠性。
        表示有 90% 的概率真实值落在此区间内。

        使用 t 分布（而非正态分布），因为：
        1. 样本量有限（每个模型 121 个输出）
        2. t 分布在小样本下提供更保守的估计
        3. 当样本量增大时，t 分布趋近于正态分布

    输入:
        scores (list): 评分列表
        confidence (float): 置信水平，默认 0.90

    输出:
        dict: {
            "mean": 均值,
            "std": 标准差,
            "ci_lower": 置信区间下界,
            "ci_upper": 置信区间上界,
            "margin_of_error": 误差范围,
            "n": 样本数量
        }

    计算逻辑:
        1. 计算样本均值 x̄ 和标准差 s
        2. 标准误差 SE = s / √n
        3. t 临界值 = t_α/2, n-1
        4. 置信区间 = x̄ ± t × SE
    """
    scores_array = np.array(scores)
    n = len(scores_array)

    if n < 2:
        # 样本太少，无法计算置信区间
        mean_val = float(np.mean(scores_array)) if n > 0 else 0.0
        return {
            "mean": mean_val,
            "std": 0.0,
            "ci_lower": mean_val,
            "ci_upper": mean_val,
            "margin_of_error": 0.0,
            "n": n
        }

    # 步骤1: 计算均值和标准差
    mean_val = float(np.mean(scores_array))
    std_val = float(np.std(scores_array, ddof=1))  # ddof=1 使用 Bessel 校正

    # 步骤2: 计算标准误差
    # SE = 标准差 / √样本数
    se = std_val / np.sqrt(n)

    # 步骤3: 获取 t 分布的临界值
    # 双尾检验，自由度 = n-1
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)

    # 步骤4: 计算误差范围和置信区间
    margin_of_error = t_critical * se
    ci_lower = mean_val - margin_of_error
    ci_upper = mean_val + margin_of_error

    return {
        "mean": mean_val,
        "std": std_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "margin_of_error": margin_of_error,
        "n": n
    }


def compute_improvement(baseline_mean: float, rag_mean: float) -> float:
    """
    计算 RAG 相对于 Baseline 的百分比改进。

    论文中的 Table 3 使用此公式计算改进幅度。

    输入:
        baseline_mean (float): Baseline LLM 的指标均值
        rag_mean (float): RAG-Augmented LLM 的指标均值

    输出:
        float: 百分比改进，如 60.5 表示 60.5% 的改进

    计算逻辑:
        improvement = ((rag_mean - baseline_mean) / baseline_mean) × 100
    """
    if baseline_mean == 0:
        return float("inf") if rag_mean > 0 else 0.0

    return ((rag_mean - baseline_mean) / baseline_mean) * 100


# =============================================
# 单元测试 / 功能演示
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("评估指标功能测试")
    print("=" * 60)

    # 测试数据
    reference = (
        "Carbohydrates are the body's primary source of energy. "
        "They are broken down into glucose, which fuels cellular activity. "
        "The brain relies primarily on glucose for its energy needs."
    )
    candidate = (
        "Carbohydrates serve as the main energy source for the body. "
        "They get converted into glucose that powers cells. "
        "The brain depends heavily on glucose for energy."
    )

    # 测试 BLEU
    print("\n--- BLEU Score ---")
    bleu = compute_bleu(reference, candidate)
    print(f"  BLEU: {bleu:.4f}")

    # 测试 ROUGE
    print("\n--- ROUGE Scores ---")
    rouge = compute_rouge(reference, candidate)
    for key, value in rouge.items():
        print(f"  {key}: {value:.4f}")

    # 测试 BERTScore
    print("\n--- BERTScore ---")
    bert = compute_bertscore_single(reference, candidate)
    for key, value in bert.items():
        print(f"  BERT {key}: {value:.4f}")

    # 测试综合指标
    print("\n--- 综合指标 ---")
    all_metrics = compute_all_metrics(reference, candidate)
    for key, value in all_metrics.items():
        print(f"  {key}: {value:.4f}")

    # 测试置信区间
    print("\n--- 置信区间 (90%) ---")
    sample_scores = [0.85, 0.87, 0.82, 0.89, 0.84, 0.86, 0.88, 0.83, 0.87, 0.85, 0.86]
    ci = compute_confidence_interval(sample_scores, confidence=0.90)
    print(f"  均值: {ci['mean']:.4f}")
    print(f"  标准差: {ci['std']:.4f}")
    print(f"  90% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    print(f"  误差范围: ±{ci['margin_of_error']:.4f}")

    # 测试改进百分比
    print("\n--- 改进百分比 ---")
    improvement = compute_improvement(0.04, 0.064)
    print(f"  Baseline=0.04, RAG=0.064 → improvement={improvement:.1f}%")
