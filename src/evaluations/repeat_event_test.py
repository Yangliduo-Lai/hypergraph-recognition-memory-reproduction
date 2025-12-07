# 文件：src/evaluations/repeat_event_test.py
"""
5.1.1 重复事件 + 分组扰动测试（Repeat Event Test with Grouped Perturbations）

新的实验设计：

    - 从事件库中随机抽取 5 条不同的事件 E_1,...,E_5
    - 对第 i 条事件 E_i：
        * 先输入原始事件 E_i（k=0）
        * 然后对其做 n_repeats 次 k=i 维扰动：
              - i=1 → 1 维扰动
              - i=2 → 2 维扰动
              - ...
              - i=5 → 5 维扰动
        * 每一次扰动都重新随机选维度和新值（不会重复同一个 E'）

    - 整个过程中 HyperMemory 是同一个，按时间顺序依次接收所有输入：
        E_1,  n_repeats 个 E'_1(k=1),
        E_2,  n_repeats 个 E'_2(k=2),
        ...
        E_5,  n_repeats 个 E'_5(k=5)

    - 观察：
        * is_old           : old/new 判断
        * edge_coverage    : 事件超边被匹配覆盖的比例
        * link_has_cycle   : 激活的记忆超边中是否存在环
        * n_active_edges   : 被激活的记忆超边数量
        * memory_edges     : 记忆中的超边数量（全局）
        * memory_links     : 记忆中的 link 数量（全局）
        * k                : 当前步扰动了多少个维度（0 表示原始 E_i）
        * group            : 第几组基准事件（1..5），也对应 k

可视化：
    - 按扰动维度 k，将 5 组实验分别画成 5 张图：
        repeat_event_test_k1.png, ..., repeat_event_test_k5.png

使用方法（从项目根目录）：

    # 默认：每个 k=1..5 的事件，各做 50 次扰动
    python src/evaluations/repeat_event_test.py

    # 自定义每个 k 的重复次数 / 设置随机种子
    python src/evaluations/repeat_event_test.py --n-repeats 30 --seed 0
"""

import sys
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import matplotlib.pyplot as plt

# ---- 为了能从 src/ 下 import sampling / connection / synthetic_events，这里手动把 src 加到 sys.path ----
SRC_DIR = Path(__file__).resolve().parent.parent  # .../src
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from sampling import SamplingConfig, build_event_hypergraph, load_events_from_csv  # type: ignore
from connection import HyperMemory  # type: ignore
from synthetic_events import EventGeneratorConfig, generate_synthetic_events  # type: ignore


# ============== 工具：构造扰动后的事件 ==============

def make_perturbed_event(
    x: np.ndarray,
    n_perturb_dims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    从事件 x 构造一个轻微变形版本 x'：
      - 随机选 n_perturb_dims 个维度（不放回）
      - 在每个维度上改成一个“不同于原值”的新值。

    为了保证新值在合法取值范围内，优先使用 synthetic_events.EventGeneratorConfig
    的 domain_sizes；如果长度不匹配，就用“当前值 + 2”估一个上界。
    """
    x = np.asarray(x)
    n_attrs = x.shape[0]
    k = min(max(n_perturb_dims, 0), n_attrs)
    if k == 0:
        return x.copy()

    # 从 synthetic_events 的配置中拿 domain_sizes（如果能对齐）
    default_cfg = EventGeneratorConfig()
    domain_sizes = list(default_cfg.domain_sizes)
    if len(domain_sizes) == 1:
        domain_sizes = domain_sizes * n_attrs

    if len(domain_sizes) != n_attrs:
        # 兜底：用当前事件值 + 2 估计取值空间大小
        domain_sizes = [int(v) + 2 for v in x]

    x_prime = x.copy()
    dims = rng.choice(n_attrs, size=k, replace=False)

    for d in dims:
        dom = int(domain_sizes[d])
        if dom <= 1:
            continue
        old_val = int(x_prime[d])
        if dom == 2:
            x_prime[d] = 1 - old_val
        else:
            candidates = list(range(dom))
            if old_val in candidates:
                candidates.remove(old_val)
            x_prime[d] = rng.choice(candidates)

    return x_prime


# ============== 核心测试逻辑 ==============

def run_repeat_event_test(
    n_repeats: int = 50,
    seed: int = 0,
):
    """
    分组重复 + 扰动事件测试核心函数。

    参数：
        n_repeats   : 对每个扰动维度 k（1..5）生成多少次不同的 E'
        seed        : 随机种子（用于抽事件 / sampling / 扰动）
    """
    rng = np.random.default_rng(seed)

    # 1. 读取事件库
    try:
        events = load_events_from_csv()
        print(f"[数据] 从 events.csv 中读取 {events.shape[0]} 条事件。")
    except Exception as e:
        # 如果没有 events.csv，就直接在线生成若干事件
        print("[警告] 读取 events.csv 失败，将在线生成事件。")
        print("       错误信息:", repr(e))
        gen_cfg = EventGeneratorConfig(n_events=1000, seed=seed)
        events = generate_synthetic_events(gen_cfg)
        print(f"       已生成 {events.shape[0]} 条 synthetic 事件。")

    n_total, n_attrs = events.shape
    if n_total < 5:
        raise RuntimeError(f"事件数量不足（{n_total} < 5），无法为 k=1..5 各选一条基准事件。")

    # 2. 随机抽取 5 条不同的事件作为 E_1,...,E_5
    base_indices = rng.choice(n_total, size=5, replace=False)
    print(f"[采样] 作为基准事件的 indices (对应 k=1..5)：{base_indices.tolist()}")

    # 3. Sampling 配置 & HyperMemory
    cfg = SamplingConfig(
        n_attrs=n_attrs,
        mode="ring",
        edge_type="random_order",
        k_min=2,
        k_max=5,
    )

    mem = HyperMemory(n_attrs=n_attrs)
    rng_sampling = np.random.default_rng(seed + 1000)   # 用于构建事件超图
    rng_perturb = np.random.default_rng(seed + 2000)    # 用于扰动

    print("\n[测试配置]")
    print(f"  n_repeats_per_k = {n_repeats}")
    print(f"  n_attrs         = {n_attrs}")
    print(f"  seed            = {seed}")
    print("\n[分组重复 + 扰动事件测试开始]\n")

    header = (
        f"{'step':>4}  "
        f"{'group':>5}  "
        f"{'k':>3}  "
        f"{'is_old':>6}  "
        f"{'edge_cov':>8}  "
        f"{'has_cycle':>9}  "
        f"{'n_active':>9}  "
        f"{'mem_edges':>10}  "
        f"{'mem_links':>9}"
    )
    print(header)
    print("-" * len(header))

    # 记录每一步的指标，方便后面画图
    steps: List[int] = []
    group_list: List[int] = []   # 1..5，对应 k
    k_list: List[int] = []
    is_old_list: List[int] = []
    edge_cov_list: List[float] = []
    has_cycle_list: List[int] = []
    n_active_list: List[int] = []
    mem_edges_list: List[int] = []
    mem_links_list: List[int] = []

    step = 0

    # ===== 对每个 k=1..5，选一条基准事件，做扰动 =====
    for group_k, base_idx in enumerate(base_indices, start=1):
        x_base = events[base_idx]
        print(f"\n[分组 {group_k}] 基准事件 index={base_idx}, 扰动维度 k={group_k}")
        print("          x_base =", x_base)

        # --- 3.1 先输入原始事件 E_i（k=0） ---
        hg_E = build_event_hypergraph(
            x_base, cfg, event_id=step, rng=rng_sampling
        )
        step += 1
        result_E = mem.process_event(hg_E)

        activated_edges: Set[int] = set()
        matches_per_edge: Dict[int, List[Dict]] = result_E["matches_per_edge"]
        for cands in matches_per_edge.values():
            for c in cands:
                activated_edges.add(c["mem_edge_id"])
        n_active = len(activated_edges)

        is_old = result_E["is_old"]
        edge_cov = result_E["edge_coverage"]
        has_cycle = result_E["link_has_cycle"]
        mem_edges = len(mem.edges)
        mem_links = len(mem.link_weights)
        k_perturb = 0  # 原始事件

        print(
            f"{step:4d}  "
            f"{group_k:5d}  "
            f"{k_perturb:3d}  "
            f"{str(is_old):>6}  "
            f"{edge_cov:8.2f}  "
            f"{str(has_cycle):>9}  "
            f"{n_active:9d}  "
            f"{mem_edges:10d}  "
            f"{mem_links:9d}"
        )

        steps.append(step)
        group_list.append(group_k)
        k_list.append(k_perturb)
        is_old_list.append(1 if is_old else 0)
        edge_cov_list.append(edge_cov)
        has_cycle_list.append(1 if has_cycle else 0)
        n_active_list.append(n_active)
        mem_edges_list.append(mem_edges)
        mem_links_list.append(mem_links)

        # --- 3.2 对该事件做 n_repeats 次 group_k 维的扰动 ---
        for _ in range(n_repeats):
            x_prime = make_perturbed_event(
                x_base, n_perturb_dims=group_k, rng=rng_perturb
            )

            hg_Ep = build_event_hypergraph(
                x_prime, cfg, event_id=step, rng=rng_sampling
            )
            step += 1
            result = mem.process_event(hg_Ep)

            activated_edges = set()
            matches_per_edge = result["matches_per_edge"]
            for cands in matches_per_edge.values():
                for c in cands:
                    activated_edges.add(c["mem_edge_id"])
            n_active = len(activated_edges)

            is_old = result["is_old"]
            edge_cov = result["edge_coverage"]
            has_cycle = result["link_has_cycle"]
            mem_edges = len(mem.edges)
            mem_links = len(mem.link_weights)
            k_perturb = group_k

            print(
                f"{step:4d}  "
                f"{group_k:5d}  "
                f"{k_perturb:3d}  "
                f"{str(is_old):>6}  "
                f"{edge_cov:8.2f}  "
                f"{str(has_cycle):>9}  "
                f"{n_active:9d}  "
                f"{mem_edges:10d}  "
                f"{mem_links:9d}"
            )

            steps.append(step)
            group_list.append(group_k)
            k_list.append(k_perturb)
            is_old_list.append(1 if is_old else 0)
            edge_cov_list.append(edge_cov)
            has_cycle_list.append(1 if has_cycle else 0)
            n_active_list.append(n_active)
            mem_edges_list.append(mem_edges)
            mem_links_list.append(mem_links)

    print("\n[测试结束]")
    print(f"  总步数: {step}")
    print(f"  最终记忆中的超边数量: {len(mem.edges)}")
    print(f"  最终记忆中的 links 数量: {len(mem.link_weights)}")

    # 画图可视化（按 k 分成 5 张图）
    plot_repeat_event_results(
        steps=steps,
        k=k_list,
        group=group_list,
        is_old=is_old_list,
        edge_cov=edge_cov_list,
        has_cycle=has_cycle_list,
        n_active=n_active_list,
        mem_edges=mem_edges_list,
        mem_links=mem_links_list,
    )


# ============== 可视化 ==============

def plot_repeat_event_results(
    steps: List[int],
    k: List[int],
    group: List[int],
    is_old: List[int],
    edge_cov: List[float],
    has_cycle: List[int],
    n_active: List[int],
    mem_edges: List[int],
    mem_links: List[int],
):
    """
    将分组重复 + 扰动事件测试的结果，按 k=1..5 分别画成 5 张图，
    并保存到 data/evaluations/repeat_event_test_k{1..5}.png。
    """
    project_root = Path(__file__).resolve().parent.parent.parent  # 到项目根
    out_dir = project_root / "data" / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)

    steps_arr = np.array(steps)
    k_arr = np.array(k)
    group_arr = np.array(group)
    is_old_arr = np.array(is_old)
    edge_cov_arr = np.array(edge_cov)
    has_cycle_arr = np.array(has_cycle)
    n_active_arr = np.array(n_active)
    mem_edges_arr = np.array(mem_edges)
    mem_links_arr = np.array(mem_links)

    # 对每个 group_k (1..5) 单独画一张图
    for g in range(1, 6):
        mask = (group_arr == g)
        if not mask.any():
            continue

        s = steps_arr[mask]
        k_g = k_arr[mask]
        is_old_g = is_old_arr[mask]
        edge_cov_g = edge_cov_arr[mask]
        has_cycle_g = has_cycle_arr[mask]
        n_active_g = n_active_arr[mask]
        mem_edges_g = mem_edges_arr[mask]
        mem_links_g = mem_links_arr[mask]

        out_path = out_dir / f"repeat_event_test_k{g}.png"

        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

        # --- 图 1：is_old & has_cycle & k(扰动维度数) ---
        ax = axes[0]
        ax.plot(s, is_old_g, marker="o", label="is_old (1=True,0=False)")
        ax.plot(s, has_cycle_g, marker="s", linestyle="--", label="has_cycle")
        ax.set_ylabel("Binary")
        ax.set_yticks([0, 1])
        ax.set_title(f"Group {g}: base event + perturbed E' (k={g})")
        ax.grid(True, linestyle=":", alpha=0.5)

        ax2 = ax.twinx()
        ax2.plot(s, k_g, marker="^", linestyle=":", label="k_perturb")
        ax2.set_ylabel("Perturbed dims k")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper right")

        # 用竖线标出这一组的基准事件（k=0 的那一步）
        base_mask = (k_g == 0)
        if base_mask.any():
            base_step = s[base_mask][0]
            ax.axvline(x=base_step, linestyle=":", linewidth=0.8, color="gray")
            ax.text(base_step, 1.05, f"E{g}", ha="center", va="bottom", fontsize=8)

        # --- 图 2：edge_coverage & n_active_edges ---
        ax = axes[1]
        ax.plot(s, edge_cov_g, marker="o", label="edge_coverage")
        ax.set_ylabel("Edge coverage")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.5)

        ax2 = ax.twinx()
        ax2.plot(s, n_active_g, marker="^", linestyle="--", label="n_active_edges")
        ax2.set_ylabel("Active memory edges")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper right")

        # --- 图 3：memory_edges & memory_links ---
        ax = axes[2]
        ax.plot(s, mem_edges_g, marker="o", label="memory_edges")
        ax.plot(s, mem_links_g, marker="s", linestyle="--", label="memory_links")
        ax.set_xlabel("Step")
        ax.set_ylabel("Count")
        ax.set_title("Memory size over inputs (this group)")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"[可视化] Group {g} 图像已保存到: {out_path}")

        try:
            plt.show()
        except Exception as e:
            print(f"[可视化] Group {g} plt.show() 失败，可能是无图形界面环境。错误信息:", repr(e))

        plt.close(fig)


# ============== 命令行入口 ==============

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "5.1.1 分组重复 + 扰动事件测试："
            "从事件库中抽 5 条基准事件，分别进行 1..5 维的多次扰动，"
            "并按扰动维度 k 分别输出 5 张图。"
        )
    )
    parser.add_argument(
        "--n-repeats",
        "-r",
        type=int,
        default=50,
        help="对每个扰动维度 k(1..5) 生成多少个不同的 E'（默认 50）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="用于抽基准事件、sampling 和扰动的随机种子（默认 0）",
    )

    args = parser.parse_args()

    run_repeat_event_test(
        n_repeats=args.n_repeats,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
