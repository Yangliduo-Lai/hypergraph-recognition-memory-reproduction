# 文件：src/evaluations/roc_recognition_test.py
"""
仿论文的识别实验（old/new + ROC）

流程：
  1. 从 events.csv 读取事件（若失败则在线生成 synthetic 事件）。
  2. 按时间顺序将前一部分事件作为“学习阶段”（encoding）输入 HyperMemory。
  3. 构建测试集：
       - old items: 从学习阶段的事件中随机抽若干条
       - new items: 从后半段未见过的事件中随机抽若干条
  4. 对测试集中的每条事件：
       - 调用 HyperMemory.familiarity_judgment(hg)（不更新记忆）
       - 记录 familiarity_score 作为连续熟悉度
  5. 基于 (score, label) 计算 ROC 曲线和 AUC，并画图保存。

命令行示例（从项目根目录）：

  # 默认参数
  python src/evaluations/roc_recognition_test.py

  # 使用 60% 事件作为训练，old/new 各取 300 条
  python src/evaluations/roc_recognition_test.py --train-ratio 0.6 --n-test-old 300 --n-test-new 300 --seed 0
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# 把 src 加入 sys.path，方便 import
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from sampling import SamplingConfig, build_event_hypergraph, load_events_from_csv  # type: ignore
from connection import HyperMemory  # type: ignore
from synthetic_events import EventGeneratorConfig, generate_synthetic_events  # type: ignore


# ============== 计算 ROC & AUC ==============

def compute_roc(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    简单实现 ROC 和 AUC。

    参数：
        scores : 形状 (N,)，越大表示越“old”
        labels : 形状 (N,)，0 = new, 1 = old

    返回：
        fprs, tprs, auc
    """
    assert scores.shape == labels.shape
    N = scores.shape[0]

    # 按分数从高到低排序
    order = np.argsort(scores)[::-1]
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    P = float(labels_sorted.sum())          # old 的数量
    N_neg = float(len(labels_sorted) - P)   # new 的数量

    # 从“阈值 = +∞（全判 new）”开始逐步降低
    tprs = [0.0]
    fprs = [0.0]
    tp = 0.0
    fp = 0.0

    for y in labels_sorted:
        if y == 1:
            tp += 1.0
        else:
            fp += 1.0

        tprs.append(tp / P if P > 0 else 0.0)
        fprs.append(fp / N_neg if N_neg > 0 else 0.0)

    fprs_arr = np.array(fprs)
    tprs_arr = np.array(tprs)
    # 梯形公式近似 AUC
    auc = float(np.trapz(tprs_arr, fprs_arr))

    return fprs_arr, tprs_arr, auc


# ============== 主实验逻辑 ==============

def run_roc_experiment(
    train_ratio: float = 0.5,
    n_test_old: int = 200,
    n_test_new: int = 200,
    seed: int = 0,
):
    """
    运行一次 ROC 识别实验。
    """
    rng = np.random.default_rng(seed)

    # ---- 1. 加载或生成事件 ----
    try:
        events = load_events_from_csv()
        print(f"[数据] 从 events.csv 读取到 {events.shape[0]} 条事件。")
    except Exception as e:
        print("[警告] 读取 events.csv 失败，将在线生成 synthetic 事件。")
        print("       错误信息:", repr(e))
        gen_cfg = EventGeneratorConfig(n_events=2000, seed=seed)
        events = generate_synthetic_events(gen_cfg)
        print(f"       已生成 {events.shape[0]} 条 synthetic 事件。")

    n_total, n_attrs = events.shape
    train_size = max(1, int(n_total * train_ratio))
    train_size = min(train_size, n_total - 1)  # 至少留 1 条做 new

    print(f"[划分] 总事件数 = {n_total}")
    print(f"       训练(encoding) 使用前 {train_size} 条")
    print(f"       测试候选池大小 = {n_total - train_size} 条 (作为 new 候选)")

    # ---- 2. 构建 HyperMemory & sampling 配置 ----
    cfg = SamplingConfig(
        n_attrs=n_attrs,
        mode="ring",
        edge_type="random_order",
        k_min=2,
        k_max=5,
    )

    mem = HyperMemory(n_attrs=n_attrs)
    rng_sampling = np.random.default_rng(seed)

    # 为训练阶段的每条事件保留它的 EventHypergraph，
    # 以便测试时对 old items 复用同一个 hypergraph（避免重新采样带来的偏差）
    train_hgs: List = []

    print("\n[学习阶段] 开始顺序编码训练事件...")
    for idx in range(train_size):
        hg = build_event_hypergraph(events[idx], cfg, event_id=idx, rng=rng_sampling)
        train_hgs.append(hg)
        mem.process_event(hg)  # 正常执行 old/new 判断 + 若 new 则写入记忆

    print(f"[学习阶段] 结束。当前记忆中超边数 = {len(mem.edges)}, links = {len(mem.link_weights)}")

    # ---- 3. 构建测试集 ----
    #   old: 从 0..train_size-1 里随机抽
    #   new: 从 train_size..n_total-1 里随机抽
    n_test_old = min(n_test_old, train_size)
    n_new_pool = n_total - train_size
    n_test_new = min(n_test_new, n_new_pool)

    old_indices = rng.choice(train_size, size=n_test_old, replace=False)
    new_pool_indices = rng.choice(n_new_pool, size=n_test_new, replace=False) + train_size
    print(f"\n[测试集] old items = {n_test_old} 条, new items = {n_test_new} 条")

    test_events = []
    test_hgs = []
    labels = []

    # old items：直接复用训练时的超图
    for idx in old_indices:
        test_events.append(events[idx])
        test_hgs.append(train_hgs[idx])
        labels.append(1)  # 1 = old

    # new items：需要新建超图（这些事件在训练阶段没出现过）
    for idx in new_pool_indices:
        x = events[idx]
        hg = build_event_hypergraph(x, cfg, event_id=idx, rng=rng_sampling)
        test_events.append(x)
        test_hgs.append(hg)
        labels.append(0)  # 0 = new

    test_events = np.stack(test_events, axis=0)
    test_hgs = list(test_hgs)
    labels_arr = np.array(labels, dtype=int)

    # 为了不让 old/new 顺序太整齐，打乱一下
    perm = rng.permutation(len(labels_arr))
    test_events = test_events[perm]
    test_hgs = [test_hgs[i] for i in perm]
    labels_arr = labels_arr[perm]

    # ---- 4. 在固定记忆上计算 familiarity_score（不更新记忆） ----
    scores = []

    print("\n[测试阶段] 对每个测试事件计算 familiarity_score ...")
    for i, hg in enumerate(test_hgs):
        fam = mem.familiarity_judgment(hg)  # 不调用 process_event，记忆保持不变
        score = float(fam.get("familiarity_score", 0.0))
        scores.append(score)

    scores_arr = np.array(scores, dtype=float)

    # 简单输出一下 old/new 的分数统计
    old_scores = scores_arr[labels_arr == 1]
    new_scores = scores_arr[labels_arr == 0]
    print("\n[分数统计]")
    print(f"  old items: mean={old_scores.mean():.3f}, std={old_scores.std():.3f}")
    print(f"  new items: mean={new_scores.mean():.3f}, std={new_scores.std():.3f}")

    # ---- 5. 计算 ROC & AUC，并画图 ----
    fprs, tprs, auc = compute_roc(scores_arr, labels_arr)
    print(f"\n[ROC] AUC = {auc:.4f}")

    plot_roc(
        fprs=fprs,
        tprs=tprs,
        auc=auc,
        old_scores=old_scores,
        new_scores=new_scores,
        n_old=n_test_old,
        n_new=n_test_new,
    )


# ============== 画 ROC 图 & 分数直方图 ==============

def plot_roc(
    fprs: np.ndarray,
    tprs: np.ndarray,
    auc: float,
    old_scores: np.ndarray,
    new_scores: np.ndarray,
    n_old: int,
    n_new: int,
):
    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir = project_root / "data" / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "roc_recognition_test.png"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 左：ROC 曲线
    ax = axes[0]
    ax.plot(fprs, tprs, marker="o", label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Old/New Recognition ROC")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.5)

    # 右：old/new familiarity 分数直方图
    ax = axes[1]
    ax.hist(new_scores, bins=20, alpha=0.6, label=f"new (n={n_new})", density=True)
    ax.hist(old_scores, bins=20, alpha=0.6, label=f"old (n={n_old})", density=True)
    ax.set_xlabel("familiarity_score")
    ax.set_ylabel("density")
    ax.set_title("Score distribution")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[可视化] ROC 图已保存到: {out_path}")

    try:
        plt.show()
    except Exception as e:
        print("[可视化] plt.show() 失败，可能是无图形界面环境。错误信息:", repr(e))


# ============== 命令行入口 ==============

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="HyperMemory 仿论文 old/new 识别实验（计算 ROC & AUC）"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.5,
        help="用于学习阶段的事件比例（0~1，默认 0.5）",
    )
    parser.add_argument(
        "--n-test-old",
        type=int,
        default=200,
        help="测试集中 old items 的数量（默认 200）",
    )
    parser.add_argument(
        "--n-test-new",
        type=int,
        default=200,
        help="测试集中 new items 的数量（默认 200）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子（默认 0）",
    )

    args = parser.parse_args()

    run_roc_experiment(
        train_ratio=args.train_ratio,
        n_test_old=args.n_test_old,
        n_test_new=args.n_test_new,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
