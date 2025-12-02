# 文件：src/evaluations/repeat_event_test.py
"""
5.1.1 重复事件测试（Repeat Event Test）

目的：
    - 检查 HyperMemory 在面对“同一事件反复出现”时的行为：
        * 第一次看到 → 一定是 new（写入记忆）
        * 之后再看到同一个事件 → 应该被判为 old
    - 同时观察：
        * is_old           : old/new 判断
        * edge_coverage    : 事件超边被匹配覆盖的比例
        * link_has_cycle   : 激活的记忆超边中是否存在环
        * n_active_edges   : 被激活的记忆超边数量
        * memory_edges     : 记忆中的超边数量（全局）
        * memory_links     : 记忆中的 link 数量（全局）

可视化：
    - 在重复次数维度上画出上述量的变化曲线
    - 保存到 data/evaluations/repeat_event_test.png

使用方法（从项目根目录）：

    # 默认：重复同一个事件 10 次
    python src/evaluations/repeat_event_test.py

    # 自定义重复次数 / 选择第 index 条事件 / 设置随机种子
    python src/evaluations/repeat_event_test.py --n-repeats 20 --event-index 0 --seed 0
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np
import matplotlib.pyplot as plt

# ---- 为了能从 src/ 下 import sampling / connection，这里手动把 src 加到 sys.path ----
SRC_DIR = Path(__file__).resolve().parent.parent  # .../src
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from sampling import SamplingConfig, build_event_hypergraph, load_events_from_csv  # type: ignore
from connection import HyperMemory  # type: ignore


# ============== 核心测试逻辑 ==============

def run_repeat_event_test(
    n_repeats: int = 10,
    event_index: int = 0,
    seed: int = 0,
):
    """
    重复事件测试核心函数。

    参数：
        n_repeats   : 同一个事件重复喂给 HyperMemory 的次数
        event_index : 选哪一条事件作为 E（从 0 开始）
        seed        : 用于 sampling 的随机种子（构建事件超图时的随机性）
    """
    # 1. 读取一条事件向量 E
    try:
        events = load_events_from_csv()
        if event_index < 0 or event_index >= events.shape[0]:
            raise IndexError(
                f"event_index={event_index} 超出范围，"
                f"当前 events.csv 共有 {events.shape[0]} 条事件。"
            )
        x = events[event_index]
        print(f"[数据] 从 events.csv 中选取事件 index={event_index} 作为 E：")
        print("      x =", x)
    except Exception as e:
        # 如果没有 events.csv，就直接在线生成一条事件
        print("[警告] 读取 events.csv 失败，将在线随机生成一条事件作为 E。")
        print("       错误信息:", repr(e))
        from synthetic_events import EventGeneratorConfig, generate_synthetic_events  # type: ignore
        gen_cfg = EventGeneratorConfig(n_events=1, seed=seed)
        events = generate_synthetic_events(gen_cfg)
        x = events[0]
        print("      生成的 x =", x)

    n_attrs = x.shape[0]

    # 2. 构建 SamplingConfig，并把这条事件变成 EventHypergraph
    cfg = SamplingConfig(
        n_attrs=n_attrs,
        mode="ring",              # 按环形结构采样
        edge_type="random_order", # 随机阶数的超边（论文推荐）
        k_min=2,
        k_max=5,
    )
    rng = np.random.default_rng(seed)

    # 构建一次超图，然后在多次重复中复用这同一结构
    hg = build_event_hypergraph(x, cfg, event_id=0, rng=rng)
    n_event_edges = len(hg.hyperedges)

    # 3. 初始化一个空的 HyperMemory
    mem = HyperMemory(n_attrs=n_attrs)

    print("\n[测试配置]")
    print(f"  n_repeats      = {n_repeats}")
    print(f"  event_index    = {event_index}")
    print(f"  n_attrs        = {n_attrs}")
    print(f"  event_edges    = {n_event_edges}")
    print(f"  seed           = {seed}")
    print("\n[重复事件测试开始]\n")

    header = (
        f"{'step':>4}  "
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
    is_old_list: List[int] = []
    edge_cov_list: List[float] = []
    has_cycle_list: List[int] = []
    n_active_list: List[int] = []
    mem_edges_list: List[int] = []
    mem_links_list: List[int] = []

    for step in range(1, n_repeats + 1):
        result = mem.process_event(hg)

        # 统计“被激活的记忆超边数量”
        activated_edges: Set[int] = set()
        matches_per_edge: Dict[int, List[Dict]] = result["matches_per_edge"]
        for cands in matches_per_edge.values():
            for c in cands:
                activated_edges.add(c["mem_edge_id"])
        n_active = len(activated_edges)

        is_old = result["is_old"]
        edge_cov = result["edge_coverage"]
        has_cycle = result["link_has_cycle"]
        mem_edges = len(mem.edges)
        mem_links = len(mem.link_weights)

        print(
            f"{step:4d}  "
            f"{str(is_old):>6}  "
            f"{edge_cov:8.2f}  "
            f"{str(has_cycle):>9}  "
            f"{n_active:9d}  "
            f"{mem_edges:10d}  "
            f"{mem_links:9d}"
        )

        # 记录到列表
        steps.append(step)
        is_old_list.append(1 if is_old else 0)
        edge_cov_list.append(edge_cov)
        has_cycle_list.append(1 if has_cycle else 0)
        n_active_list.append(n_active)
        mem_edges_list.append(mem_edges)
        mem_links_list.append(mem_links)

    print("\n[测试结束]")
    print(f"  最终记忆中的超边数量: {len(mem.edges)}")
    print(f"  最终记忆中的 links 数量: {len(mem.link_weights)}")

    # 4. 画图可视化
    plot_repeat_event_results(
        steps=steps,
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
    is_old: List[int],
    edge_cov: List[float],
    has_cycle: List[int],
    n_active: List[int],
    mem_edges: List[int],
    mem_links: List[int],
):
    """
    将重复事件测试的结果画成折线图，并保存到 data/evaluations/ 下。
    """
    project_root = Path(__file__).resolve().parent.parent.parent  # 到项目根
    out_dir = project_root / "data" / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "repeat_event_test.png"

    steps_arr = np.array(steps)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # --- 图 1：is_old & has_cycle ---
    ax = axes[0]
    ax.plot(steps_arr, is_old, marker="o", label="is_old (1=True,0=False)")
    ax.plot(steps_arr, has_cycle, marker="s", linestyle="--", label="has_cycle")
    ax.set_ylabel("Binary")
    ax.set_yticks([0, 1])
    ax.set_title("Old/New & Cycle over repeats")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)

    # --- 图 2：edge_coverage & n_active_edges ---
    ax = axes[1]
    ax.plot(steps_arr, edge_cov, marker="o", label="edge_coverage")
    ax.set_ylabel("Edge coverage")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(steps_arr, n_active, marker="^", linestyle="--", label="n_active_edges")
    ax2.set_ylabel("Active memory edges")

    # 合并图例
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    # --- 图 3：memory_edges & memory_links ---
    ax = axes[2]
    ax.plot(steps_arr, mem_edges, marker="o", label="memory_edges")
    ax.plot(steps_arr, mem_links, marker="s", linestyle="--", label="memory_links")
    ax.set_xlabel("Repeat step")
    ax.set_ylabel("Count")
    ax.set_title("Memory size over repeats")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[可视化] 图像已保存到: {out_path}")

    # 同时弹出窗口（如果环境支持）
    try:
        plt.show()
    except Exception as e:
        print("[可视化] plt.show() 失败，可能是无图形界面环境。错误信息:", repr(e))


# ============== 命令行入口 ==============

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="5.1.1 重复事件测试：同一个事件多次输入 HyperMemory，观察 old/new 行为。"
    )
    parser.add_argument(
        "--n-repeats",
        "-r",
        type=int,
        default=10,
        help="重复同一个事件的次数（默认 10）",
    )
    parser.add_argument(
        "--event-index",
        "-i",
        type=int,
        default=0,
        help="从 events.csv 中选哪一条事件作为 E（默认 0）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="用于 sampling 的随机种子（默认 0）",
    )

    args = parser.parse_args()

    run_repeat_event_test(
        n_repeats=args.n_repeats,
        event_index=args.event_index,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
