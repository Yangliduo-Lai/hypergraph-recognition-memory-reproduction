# 文件：src/evaluations/perturbed_event_test.py
"""
5.1.x 轻微变形事件测试（Perturbed Event Test, multi-E version）

目的：
    - 对多条事件 E_1, E_2, ... 逐个进行“逐步加大扰动”的测试。
    - 对每个 E_i：
        * Step 1:  输入原始事件 E_i          （扰动 0 个维度）
        * Step 2:  输入新的 E'_i,1           （扰动 1 个维度）
        * Step 3:  输入新的 E'_i,2           （扰动 2 个维度）
        * ...
        * Step t:  输入新的 E'_i,t-1         （扰动 min(t-1, max_perturb_dims) 个维度）
    - 每个 E_i 的 HyperMemory 都从空开始单独测试，互不干扰。
    - 最后对多个 E 的结果做平均曲线（+ 标准差），
      观察模型在不同扰动强度下的行为（old/new, 覆盖率等）。

可视化：
    - 随 step 的平均 is_old / has_cycle / edge_coverage / n_active 等曲线；
    - 保存到 data/evaluations/perturbed_event_test.png。

命令行用法（从项目根目录）示例：

    # 默认：从 events.csv 选前 5 条事件，各自测试；
    #       max_perturb_dims = 5，扰动步数 10，seed = 0
    python src/evaluations/perturbed_event_test.py

    # 只测试第 3 条事件（index=2）
    python src/evaluations/perturbed_event_test.py --event-index 2

    # 测试前 10 条事件，每条事件之后 15 步扰动，最大扰动 5 维
    python src/evaluations/perturbed_event_test.py \
        --n-events-test 10 \
        --max-perturb-dims 5 \
        --n-repeats-perturb 15 \
        --seed 0
"""

import sys
from pathlib import Path
from typing import Dict, List, Set, Optional

import numpy as np
import matplotlib.pyplot as plt

# ---- 为了能从 src/ 下 import sampling / connection / synthetic_events，这里手动把 src 加到 sys.path ----
SRC_DIR = Path(__file__).resolve().parent.parent  # .../src
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from sampling import SamplingConfig, build_event_hypergraph, load_events_from_csv  # type: ignore
from connection import HyperMemory  # type: ignore
from synthetic_events import EventGeneratorConfig, generate_synthetic_events  # type: ignore


# ============== 工具函数：构造轻微变形的事件 E' ==============

def make_perturbed_event(
    x: np.ndarray,
    n_perturb_dims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    从事件 x 构造一个轻微变形版本 x'：
      - 随机选 n_perturb_dims 个维度（不放回）
      - 在每个维度上改成一个“不同于原值”的新值。

    为了保证新值在合法取值范围内，优先尝试使用 synthetic_events.EventGeneratorConfig
    里的 domain_sizes；如果长度不匹配，就用 “该维度当前值 + 2” 估一个上界。
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
        # 兜底：用当前事件中的值 + 2 估计每个维度的取值上界
        max_vals = x
        domain_sizes = [int(v) + 2 for v in max_vals]

    x_prime = x.copy()
    dims = rng.choice(n_attrs, size=k, replace=False)

    for d in dims:
        dom = int(domain_sizes[d])
        if dom <= 1:
            continue
        old_val = int(x_prime[d])

        if dom == 2:
            # 二值情况，直接 flip
            x_prime[d] = 1 - old_val
        else:
            # 从 0..dom-1 里选一个 != old_val 的值
            candidates = list(range(dom))
            if old_val in candidates:
                candidates.remove(old_val)
            x_prime[d] = rng.choice(candidates)

    return x_prime


# ============== 单个事件的测试逻辑 ==============

def run_single_event_perturbed(
    x: np.ndarray,
    event_index: int,
    max_perturb_dims: int,
    n_repeats_perturb: int,
    seed: int,
    verbose: bool = True,
) -> Dict[str, List]:
    """
    对单个事件 x 做“逐步加大扰动”的测试。

    流程：
      step=1 : 输入原始事件 E       （扰动 0 维）
      step>=2: 每步构造新的 E'，扰动维度数
               k_step = min(step-1, max_perturb_dims)

    返回：
      {
        "steps":      [step_1, step_2, ...],
        "k":          [k_1, k_2, ...],
        "is_old":     [0/1,...],
        "edge_cov":   [...],
        "has_cycle":  [0/1,...],
        "n_active":   [...],
        "mem_edges":  [...],
        "mem_links":  [...],
      }
    """
    x = np.asarray(x)
    n_attrs = x.shape[0]

    # Sampling 配置（固定）
    cfg = SamplingConfig(
        n_attrs=n_attrs,
        mode="ring",
        edge_type="random_order",
        k_min=2,
        k_max=5,
    )

    # 为了可比性：E 和每个 E' 使用相同的 sampling 随机性（超边结构相同）
    rng_sampling_E = np.random.default_rng(seed)
    hg_E = build_event_hypergraph(x, cfg, event_id=0, rng=rng_sampling_E)
    n_edges_E = len(hg_E.hyperedges)

    # HyperMemory 从空开始
    mem = HyperMemory(n_attrs=n_attrs)

    if verbose:
        print(f"\n===== Event index {event_index} 测试开始 =====")
        print(f"  x = {x}")
        print(f"  n_attrs          = {n_attrs}")
        print(f"  n_edges_E        = {n_edges_E}")
        print(f"  max_perturb_dims = {max_perturb_dims}")
        print(f"  n_repeats_perturb= {n_repeats_perturb}")
        print(f"  seed             = {seed}\n")

        header = (
            f"{'step':>4}  "
            f"{'which':>5}  "
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

    steps: List[int] = []
    which_list: List[int] = []   # 0=E, 1=E'
    k_list: List[int] = []
    is_old_list: List[int] = []
    edge_cov_list: List[float] = []
    has_cycle_list: List[int] = []
    n_active_list: List[int] = []
    mem_edges_list: List[int] = []
    mem_links_list: List[int] = []

    step_id = 1

    # --- step 1: 原始 E（扰动 0 维） ---
    result_E = mem.process_event(hg_E)
    activated_edges: Set[int] = set()
    for cands in result_E["matches_per_edge"].values():
        for c in cands:
            activated_edges.add(c["mem_edge_id"])
    n_active = len(activated_edges)

    is_old = result_E["is_old"]
    edge_cov = result_E["edge_coverage"]
    has_cycle = result_E["link_has_cycle"]
    mem_edges = len(mem.edges)
    mem_links = len(mem.link_weights)
    k_perturb = 0

    if verbose:
        print(
            f"{step_id:4d}  "
            f"{'E':>5}  "
            f"{k_perturb:3d}  "
            f"{str(is_old):>6}  "
            f"{edge_cov:8.2f}  "
            f"{str(has_cycle):>9}  "
            f"{n_active:9d}  "
            f"{mem_edges:10d}  "
            f"{mem_links:9d}"
        )

    steps.append(step_id)
    which_list.append(0)
    k_list.append(k_perturb)
    is_old_list.append(1 if is_old else 0)
    edge_cov_list.append(edge_cov)
    has_cycle_list.append(1 if has_cycle else 0)
    n_active_list.append(n_active)
    mem_edges_list.append(mem_edges)
    mem_links_list.append(mem_links)

    # --- 后续 step: 逐步加大扰动的 E' ---
    rng_perturb = np.random.default_rng(seed + 9999 + event_index)  # 不同事件用不同扰动 RNG

    for i in range(n_repeats_perturb):
        step_id += 1
        k_perturb = min(i + 1, max_perturb_dims)

        # 生成新的 E'（每一步都重新扰动）
        x_prime = make_perturbed_event(x, n_perturb_dims=k_perturb, rng=rng_perturb)

        # 保持与 E 一致的 sampling 结构（同一个 seed）
        rng_sampling_Ep = np.random.default_rng(seed)
        hg_Ep = build_event_hypergraph(x_prime, cfg, event_id=step_id, rng=rng_sampling_Ep)

        result = mem.process_event(hg_Ep)

        activated_edges = set()
        for cands in result["matches_per_edge"].values():
            for c in cands:
                activated_edges.add(c["mem_edge_id"])
        n_active = len(activated_edges)

        is_old = result["is_old"]
        edge_cov = result["edge_coverage"]
        has_cycle = result["link_has_cycle"]
        mem_edges = len(mem.edges)
        mem_links = len(mem.link_weights)

        if verbose:
            print(
                f"{step_id:4d}  "
                f"{'E\'':>5}  "
                f"{k_perturb:3d}  "
                f"{str(is_old):>6}  "
                f"{edge_cov:8.2f}  "
                f"{str(has_cycle):>9}  "
                f"{n_active:9d}  "
                f"{mem_edges:10d}  "
                f"{mem_links:9d}"
            )

        steps.append(step_id)
        which_list.append(1)
        k_list.append(k_perturb)
        is_old_list.append(1 if is_old else 0)
        edge_cov_list.append(edge_cov)
        has_cycle_list.append(1 if has_cycle else 0)
        n_active_list.append(n_active)
        mem_edges_list.append(mem_edges)
        mem_links_list.append(mem_links)

    if verbose:
        print(f"===== Event index {event_index} 测试结束 =====")
        print(f"  最终记忆中的超边数量: {len(mem.edges)}")
        print(f"  最终记忆中的 links 数量: {len(mem.link_weights)}\n")

    return {
        "steps": steps,
        "k": k_list,
        "is_old": is_old_list,
        "edge_cov": edge_cov_list,
        "has_cycle": has_cycle_list,
        "n_active": n_active_list,
        "mem_edges": mem_edges_list,
        "mem_links": mem_links_list,
    }


# ============== 聚合多个事件的结果并画图 ==============

def plot_perturbed_event_results_aggregated(
    steps: List[int],
    k: List[int],
    all_is_old: np.ndarray,    # shape: (n_events, T)
    all_edge_cov: np.ndarray,
    all_has_cycle: np.ndarray,
    all_n_active: np.ndarray,
    all_mem_edges: np.ndarray,
    all_mem_links: np.ndarray,
):
    """
    将多个事件的结果聚合成平均曲线，并保存到 data/evaluations/ 下。
    """
    project_root = Path(__file__).resolve().parent.parent.parent  # 到项目根
    out_dir = project_root / "data" / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "perturbed_event_test.png"

    steps_arr = np.array(steps)
    k_arr = np.array(k)
    n_events, T = all_is_old.shape

    # 计算均值和标准差
    def mean_std(arr: np.ndarray):
        return arr.mean(axis=0), arr.std(axis=0)

    m_is_old, s_is_old = mean_std(all_is_old)
    m_has_cycle, s_has_cycle = mean_std(all_has_cycle)
    m_edge_cov, s_edge_cov = mean_std(all_edge_cov)
    m_n_active, s_n_active = mean_std(all_n_active)
    m_mem_edges, s_mem_edges = mean_std(all_mem_edges)
    m_mem_links, s_mem_links = mean_std(all_mem_links)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # --- 图 1：is_old & has_cycle & k_perturb ---
    ax = axes[0]
    ax.plot(steps_arr, m_is_old, marker="o", label="mean is_old")
    ax.fill_between(steps_arr, m_is_old - s_is_old, m_is_old + s_is_old,
                    alpha=0.2, linewidth=0)

    ax.plot(steps_arr, m_has_cycle, marker="s", linestyle="--", label="mean has_cycle")
    ax.fill_between(steps_arr, m_has_cycle - s_has_cycle, m_has_cycle + s_has_cycle,
                    alpha=0.2, linewidth=0)

    ax.set_ylabel("Prob / fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Original E (step 1) vs progressively perturbed E' (n_events={n_events})")
    ax.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(steps_arr, k_arr, marker="^", linestyle=":", label="k_perturb")
    ax2.set_ylabel("Perturbed dims k")

    # 合并图例
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    ax.axvline(x=1, linestyle=":", linewidth=1.0)
    ax.text(1, 1.05, "E", ha="center", va="bottom")

    # --- 图 2：edge_coverage & n_active_edges ---
    ax = axes[1]
    ax.plot(steps_arr, m_edge_cov, marker="o", label="mean edge_coverage")
    ax.fill_between(steps_arr, m_edge_cov - s_edge_cov, m_edge_cov + s_edge_cov,
                    alpha=0.2, linewidth=0)
    ax.set_ylabel("Edge coverage")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(steps_arr, m_n_active, marker="^", linestyle="--", label="mean n_active_edges")
    ax2.fill_between(steps_arr, m_n_active - s_n_active, m_n_active + s_n_active,
                     alpha=0.2, linewidth=0)
    ax2.set_ylabel("Active memory edges")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    # --- 图 3：memory_edges & memory_links ---
    ax = axes[2]
    ax.plot(steps_arr, m_mem_edges, marker="o", label="mean memory_edges")
    ax.fill_between(steps_arr, m_mem_edges - s_mem_edges, m_mem_edges + s_mem_edges,
                    alpha=0.2, linewidth=0)
    ax.plot(steps_arr, m_mem_links, marker="s", linestyle="--", label="mean memory_links")
    ax.fill_between(steps_arr, m_mem_links - s_mem_links, m_mem_links + s_mem_links,
                    alpha=0.2, linewidth=0)

    ax.set_xlabel("Step (1 = E, >=2 = perturbed E')")
    ax.set_ylabel("Count")
    ax.set_title("Memory size over inputs")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[可视化] 图像已保存到: {out_path}")

    try:
        plt.show()
    except Exception as e:
        print("[可视化] plt.show() 失败，可能是无图形界面环境。错误信息:", repr(e))


# ============== 顶层：多事件测试 ==============

def run_perturbed_event_test_multi(
    events: np.ndarray,
    event_indices: List[int],
    max_perturb_dims: int = 5,
    n_repeats_perturb: int = 10,
    seed: int = 0,
):
    """
    对多个事件 indices 逐个做“逐步加大扰动”的测试，并聚合结果。
    """
    n_events_total = events.shape[0]
    print(f"[多事件测试] 总共有 {n_events_total} 条事件，将测试 indices = {event_indices}")

    # 先用第一条事件跑一下，拿到 step 数 T
    first_idx = event_indices[0]
    res0 = run_single_event_perturbed(
        x=events[first_idx],
        event_index=first_idx,
        max_perturb_dims=max_perturb_dims,
        n_repeats_perturb=n_repeats_perturb,
        seed=seed + first_idx * 100,
        verbose=True,
    )

    steps = res0["steps"]
    k = res0["k"]
    T = len(steps)
    nE = len(event_indices)

    # 初始化聚合数组
    all_is_old = np.zeros((nE, T), dtype=float)
    all_edge_cov = np.zeros((nE, T), dtype=float)
    all_has_cycle = np.zeros((nE, T), dtype=float)
    all_n_active = np.zeros((nE, T), dtype=float)
    all_mem_edges = np.zeros((nE, T), dtype=float)
    all_mem_links = np.zeros((nE, T), dtype=float)

    def fill_row(idx_row: int, res: Dict[str, List]):
        all_is_old[idx_row, :] = np.array(res["is_old"], dtype=float)
        all_edge_cov[idx_row, :] = np.array(res["edge_cov"], dtype=float)
        all_has_cycle[idx_row, :] = np.array(res["has_cycle"], dtype=float)
        all_n_active[idx_row, :] = np.array(res["n_active"], dtype=float)
        all_mem_edges[idx_row, :] = np.array(res["mem_edges"], dtype=float)
        all_mem_links[idx_row, :] = np.array(res["mem_links"], dtype=float)

    fill_row(0, res0)

    # 其余事件
    for row_i, idx in enumerate(event_indices[1:], start=1):
        res = run_single_event_perturbed(
            x=events[idx],
            event_index=idx,
            max_perturb_dims=max_perturb_dims,
            n_repeats_perturb=n_repeats_perturb,
            seed=seed + idx * 100,
            verbose=False,  # 可以改成 True 看详细日志
        )
        fill_row(row_i, res)

    # 画聚合图
    plot_perturbed_event_results_aggregated(
        steps=steps,
        k=k,
        all_is_old=all_is_old,
        all_edge_cov=all_edge_cov,
        all_has_cycle=all_has_cycle,
        all_n_active=all_n_active,
        all_mem_edges=all_mem_edges,
        all_mem_links=all_mem_links,
    )


# ============== 命令行入口 ==============

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="轻微变形事件测试：对多个事件 E 做逐步扰动，观察 HyperMemory 的行为。"
    )
    parser.add_argument(
        "--event-index",
        "-i",
        type=int,
        default=None,
        help="只测试某一条事件的 index（从 0 开始）。如果不指定，则测试前 n-events-test 条事件。",
    )
    parser.add_argument(
        "--n-events-test",
        "-N",
        type=int,
        default=5,
        help="在未指定 --event-index 时，测试前 N 条事件（默认 5）。",
    )
    parser.add_argument(
        "--max-perturb-dims",
        "-k",
        type=int,
        default=5,
        help="扰动维度数的最大值（从 1 递增到该值后保持，默认 5）。",
    )
    parser.add_argument(
        "--n-repeats-perturb",
        "-r",
        type=int,
        default=10,
        help="扰动版 E' 的总步数（不包括原始 E 的那一步，默认 10）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="用于 sampling 和变形的随机种子（默认 0）。",
    )

    args = parser.parse_args()

    # ---- 加载或生成事件集合 ----
    # 需要至少 required_n 条事件
    if args.event_index is None:
        required_n = args.n_events_test
    else:
        required_n = args.event_index + 1

    try:
        events = load_events_from_csv()
        if events.shape[0] < required_n:
            print(
                f"[警告] events.csv 只有 {events.shape[0]} 条事件，小于需要的 {required_n}，"
                f"将只使用前 {events.shape[0]} 条。"
            )
            required_n = events.shape[0]
    except Exception as e:
        print("[警告] 读取 events.csv 失败，将在线生成合成事件。")
        print("       错误信息:", repr(e))
        gen_cfg = EventGeneratorConfig(n_events=required_n, seed=args.seed)
        events = generate_synthetic_events(gen_cfg)
        print(f"       已生成 {events.shape[0]} 条合成事件。")

    # ---- 决定测试哪些事件 ----
    if args.event_index is not None:
        if args.event_index < 0 or args.event_index >= events.shape[0]:
            raise IndexError(
                f"--event-index={args.event_index} 超出范围，"
                f"当前事件数为 {events.shape[0]}。"
            )
        event_indices = [args.event_index]
    else:
        n_use = min(args.n_events_test, events.shape[0])
        event_indices = list(range(n_use))

    # ---- 跑多事件测试 ----
    run_perturbed_event_test_multi(
        events=events,
        event_indices=event_indices,
        max_perturb_dims=args.max_perturb_dims,
        n_repeats_perturb=args.n_repeats_perturb,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
