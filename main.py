"""
===========================================================
主程序入口 (Main Entry Point)
===========================================================

功能说明：
    提供一键运行所有步骤的便捷入口。
    也可以单独运行每个步骤脚本。

    推荐工作流程（分步执行）：
        python step1_build_index.py           # 构建向量索引
        python step2_generate_ground_truth.py  # 生成 Ground Truth
        python step3_run_experiments.py        # 运行实验
        python step4_evaluate.py              # 评估结果
        python step5_visualize.py             # 生成图表

    一键执行（本文件）：
        python main.py                        # 全部步骤
        python main.py --skip-gt              # 跳过 Ground Truth 生成
        python main.py --quick                # 快速测试

使用方式：
    python main.py
    python main.py --quick                    # 快速测试
    python main.py --skip-index               # 跳过索引构建（使用已有）
    python main.py --skip-gt                  # 跳过 Ground Truth 生成
"""

import argparse
import os
import subprocess
import sys


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="RAG vs Baseline LLM 多指标评估实验（一键运行）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
推荐分步执行:
  python step1_build_index.py           # 步骤1：构建向量索引
  python step2_generate_ground_truth.py  # 步骤2：生成 Ground Truth
  python step3_run_experiments.py        # 步骤3：运行实验
  python step4_evaluate.py              # 步骤4：评估结果
  python step5_visualize.py             # 步骤5：生成图表
        """
    )

    parser.add_argument("--quick", action="store_true",
                        help="快速测试模式")
    parser.add_argument("--skip-index", action="store_true",
                        help="跳过步骤1（使用已有向量索引）")
    parser.add_argument("--skip-gt", action="store_true",
                        help="跳过步骤2（使用已有 Ground Truth）")
    parser.add_argument("--models", nargs="+", default=None,
                        help="指定要评估的模型")
    parser.add_argument("--iterations", type=int, default=None,
                        help="每个查询的迭代次数")
    return parser.parse_args()


def run_step(script_name: str, extra_args: list = None, description: str = ""):
    """
    运行一个步骤脚本。

    参数:
        script_name (str): 脚本文件名
        extra_args (list): 额外的命令行参数
        description (str): 步骤描述
    """
    print(f"\n{'='*60}")
    print(f"▶️  {description}")
    print(f"    运行: python {script_name} {' '.join(extra_args or [])}")
    print(f"{'='*60}\n")

    cmd = [sys.executable, script_name] + (extra_args or [])
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print(f"\n❌ {script_name} 执行失败 (返回码: {result.returncode})")
        sys.exit(result.returncode)


def main():
    args = parse_args()

    print("=" * 60)
    print("🧪 RAG vs Baseline LLM 多指标评估实验")
    print("=" * 60)

    # ---- 步骤 1: 构建向量索引 ----
    if not args.skip_index:
        run_step(
            "step1_build_index.py",
            description="步骤 1/5: 构建向量索引"
        )
    else:
        print("\n⏭️  跳过步骤 1 (使用已有索引)")

    # ---- 步骤 2: 生成 Ground Truth ----
    if not args.skip_gt:
        run_step(
            "step2_generate_ground_truth.py",
            description="步骤 2/5: 生成 Ground Truth"
        )
    else:
        print("\n⏭️  跳过步骤 2 (使用已有 Ground Truth)")

    # ---- 步骤 3: 运行实验 ----
    step3_args = []
    if args.quick:
        step3_args.append("--quick")
    if args.models:
        step3_args.extend(["--models"] + args.models)
    if args.iterations:
        step3_args.extend(["--iterations", str(args.iterations)])

    run_step(
        "step3_run_experiments.py",
        extra_args=step3_args,
        description="步骤 3/5: 运行实验"
    )

    # ---- 步骤 4: 评估结果 ----
    run_step(
        "step4_evaluate.py",
        description="步骤 4/5: 评估结果"
    )

    # ---- 步骤 5: 生成图表 ----
    run_step(
        "step5_visualize.py",
        description="步骤 5/5: 生成图表"
    )

    print("\n" + "=" * 60)
    print("🎉 所有步骤执行完成!")
    print("=" * 60)
    print("📂 输出:")
    print("   向量索引: VectorDB/")
    print("   实验结果: results/")
    print("   可视化:   figures/")


if __name__ == "__main__":
    main()
