# 文件：src/visualize.py
"""
1. 从 data/event_hypergraphs/event_hypergraphs.pkl 加载事件级超图，提供三种可视化：
   - 节点共现投影图（co-occurrence）
   - 节点在圆上 + 虚线圈出超边（类似论文 hypergraph 示意图）
   - 超边环 + link（类似论文 edge ring 示意图）

2. 从 data/hypermemory/hypermemory.pkl 加载 HyperMemory，
   提供一个“记忆超图概览”：memory hyperedge 作为节点，link 作为边。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, FrozenSet
from pathlib import Path
from math import cos, sin, pi
import pickle

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle


# ================ 与 sampling/connection 中一致的数据结构 ================

# Node 和 EventHypergraph 用来加载事件级超图（来自 sampling.py 作为 __main__ 保存）
@dataclass
class Node:
    index: int
    value: int


@dataclass
class EventHypergraph:
    event_id: int
    nodes: Dict[int, Node]                  # node_id -> Node（这里 node_id 就是维度 index）
    hyperedges: Dict[int, Tuple[int, ...]]  # edge_id -> tuple(indices)
    links: List[Tuple[int, int]]            # (edge_i, edge_j)


# HyperMemory 的精简版（字段要和 connection.py 里的一致，以便反序列化）
NodeKey = Tuple[int, int]


@dataclass
class HyperMemory:
    n_attrs: int
    C: float = 5.0
    edges: Dict[int, FrozenSet[NodeKey]] = field(default_factory=dict)
    edge_sigs: Dict[FrozenSet[NodeKey], int] = field(default_factory=dict)
    next_edge_id: int = 0
    link_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)
    link_weights: Dict[Tuple[int, int], float] = field(default_factory=dict)


# ================ 加载：事件级超图 & HyperMemory ================

def load_event_hypergraphs(
    filename: str = "event_hypergraphs.pkl",
) -> List[EventHypergraph]:
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / "data" / "event_hypergraphs" / filename

    if not path.exists():
        raise FileNotFoundError(f"找不到事件超图文件: {path}")

    with open(path, "rb") as f:
        hg_list = pickle.load(f)

    print(f"[加载] 从 {path} 读取 {len(hg_list)} 个事件超图。")
    return hg_list


def load_hypermemory(
    filename: str = "hypermemory.pkl",
) -> HyperMemory:
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / "data" / "hypermemory" / filename

    if not path.exists():
        raise FileNotFoundError(f"找不到 HyperMemory 文件: {path}")

    with open(path, "rb") as f:
        mem: HyperMemory = pickle.load(f)

    print(f"[加载] 从 {path} 读取 HyperMemory。")
    return mem


# ================ 1. 节点共现投影图 ================

def visualize_event_hypergraph_projection(hg: EventHypergraph):
    """
    G 的节点 = 维度 index
    G 的边   = 在某个 hyperedge 里共同出现过的节点对
    """
    G = nx.Graph()

    # 节点：0..7，标签写 index:value
    for nid, node in hg.nodes.items():
        label = f"{node.index}:{node.value}"
        G.add_node(nid, label=label)

    # 统计共现次数
    pair_counts: Dict[Tuple[int, int], int] = {}
    for _, idx_tuple in hg.hyperedges.items():
        idx_list = list(idx_tuple)
        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                a, b = idx_list[i], idx_list[j]
                if a > b:
                    a, b = b, a
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    for (a, b), c in pair_counts.items():
        G.add_edge(a, b, weight=c)

    print(f"[投影] 事件 {hg.event_id}：节点={G.number_of_nodes()}, 边={G.number_of_edges()}")

    if G.number_of_nodes() == 0:
        print("  没有任何边可视化。")
        return

    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=0)

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1.0
    widths = [0.5 + 2.5 * (w / max_w) for w in weights]

    nx.draw_networkx_nodes(G, pos, node_size=800, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.6)

    labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    plt.title(f"Event {hg.event_id} (node co-occurrence projection)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ================ 2. 节点+圈圈超边 ================

def visualize_event_hypergraph_sets(
    hg: EventHypergraph,
    max_edges: int = 6,
):
    """
    节点在圆上，虚线椭圆圈出超边，类似论文里的 hypergraph 示意图。
    """

    # 1. 节点位置
    n = len(hg.nodes)
    angles = [2 * pi * i / n for i in range(n)]
    pos = {i: (cos(theta), sin(theta)) for i, theta in enumerate(angles)}

    fig, ax = plt.subplots(figsize=(6, 6))

    # 2. 画节点
    for i, (x, y) in pos.items():
        ax.scatter(x, y, s=200, color="tab:blue", zorder=3)
        label = f"x{i+1}"
        ax.text(x, y, label, fontsize=10, ha="center", va="center", color="white", zorder=4)

    # 3. 画超边（椭圆）
    edge_items = list(hg.hyperedges.items())[:max_edges]

    for eid, idx_tuple in edge_items:
        if not idx_tuple:
            continue
        xs = [pos[i][0] for i in idx_tuple]
        ys = [pos[i][1] for i in idx_tuple]

        x_center = sum(xs) / len(xs)
        y_center = sum(ys) / len(ys)

        margin = 0.3
        width = max(xs) - min(xs) + margin
        height = max(ys) - min(ys) + margin

        ell = Ellipse(
            (x_center, y_center),
            width,
            height,
            angle=0,
            fill=False,
            linestyle="--",
            linewidth=1.2,
            edgecolor="gray",
            alpha=0.8,
            zorder=2,
        )
        ax.add_patch(ell)

        set_label = "{" + ", ".join(f"x{i+1}" for i in idx_tuple) + "}"
        ax.text(
            x_center,
            y_center,
            set_label,
            fontsize=8,
            ha="center",
            va="center",
            color="black",
            zorder=1,
        )

    ax.set_aspect("equal")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.axis("off")
    ax.set_title(f"Event {hg.event_id} Hypergraph (hyperedges as dashed loops)")
    plt.tight_layout()
    plt.show()


# ================ 3. 超边环 + link 图 ================

def visualize_event_hypergraph_edge_ring(
    hg: EventHypergraph,
    max_edges: int = 8,
):
    """
    每条 hyperedge 画成一个由小方块组成的条，沿圆环排布，
    hg.links 中的 (edge_i, edge_j) 用虚线连接，类似论文的 edge ring 图。
    """

    m_total = len(hg.hyperedges)
    if m_total == 0:
        print(f"[edge_ring] 事件 {hg.event_id} 没有任何 hyperedge")
        return

    if max_edges is None or max_edges > m_total:
        max_edges = m_total

    edge_items = sorted(hg.hyperedges.items(), key=lambda kv: kv[0])[:max_edges]
    edge_ids = [eid for eid, _ in edge_items]
    k = len(edge_ids)

    # 圆环上的中心
    R = 2.0
    angles = [2 * pi * i / k for i in range(k)]
    edge_centers = {
        eid: (R * cos(theta), R * sin(theta))
        for eid, theta in zip(edge_ids, angles)
    }

    fig, ax = plt.subplots(figsize=(7, 7))

    box_w = 0.35
    box_h = 0.25

    for eid in edge_ids:
        idx_tuple = hg.hyperedges[eid]
        center_x, center_y = edge_centers[eid]

        num_nodes = len(idx_tuple)
        total_w = num_nodes * box_w
        x0 = center_x - total_w / 2.0
        y0 = center_y - box_h / 2.0

        for j, dim_idx in enumerate(idx_tuple):
            rx = x0 + j * box_w
            ry = y0

            rect = Rectangle(
                (rx, ry),
                box_w,
                box_h,
                linewidth=1.0,
                edgecolor="black",
                facecolor="white",
                zorder=3,
            )
            ax.add_patch(rect)

            label = f"x{dim_idx + 1}"
            ax.text(
                rx + box_w / 2.0,
                ry + box_h / 2.0,
                label,
                fontsize=8,
                ha="center",
                va="center",
                zorder=4,
            )

    # 画 links
    for (e_i, e_j) in hg.links:
        if e_i not in edge_centers or e_j not in edge_centers:
            continue
        x1, y1 = edge_centers[e_i]
        x2, y2 = edge_centers[e_j]
        ax.plot(
            [x1, x2],
            [y1, y2],
            linestyle=":",
            color="gray",
            linewidth=1.0,
            zorder=1,
        )

    ax.set_aspect("equal")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axis("off")
    ax.set_title(f"Event {hg.event_id} Hyperedges and Links (ring view)")
    plt.tight_layout()
    plt.show()


# ================ 4. HyperMemory 概览可视化 ================

def visualize_hypermemory_overview(
    mem: HyperMemory,
    max_nodes: int = 200,
    weight_threshold: float = 0.0,
):
    """
    把 HyperMemory 投影成一张普通图：

      - 每个节点 = 一条 memory hyperedge（一个记忆片段）
      - 每条边   = 这两条 memory hyperedge 之间存在 link（共现）

    这里只看结构，不展开每条 hyperedge 里面具体有哪些 (dim, value)。
    """

    G = nx.Graph()

    # 先把所有 hyperedge 当成节点
    for eid in mem.edges.keys():
        G.add_node(eid)

    # 再根据 link_weights 加边
    for (i, j), w in mem.link_weights.items():
        if w <= weight_threshold:
            continue
        G.add_edge(i, j, weight=w)

    print(f"[HyperMemory 概览] 节点={G.number_of_nodes()}, 边={G.number_of_edges()}")

    if G.number_of_nodes() == 0:
        print("  记忆中还没有任何链接。")
        return

    # 如果节点太多，只取度数最高的 max_nodes 个，方便可视化
    if G.number_of_nodes() > max_nodes:
        degrees = sorted(G.degree, key=lambda kv: kv[1], reverse=True)
        top_nodes = [n for n, _ in degrees[:max_nodes]]
        G = G.subgraph(top_nodes).copy()
        print(f"  子图：节点={G.number_of_nodes()}, 边={G.number_of_edges()}")

    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=0)

    degrees = dict(G.degree)
    node_sizes = [50 + degrees[n] * 10 for n in G.nodes()]
    widths = [0.5 + 2.5 * G[u][v]["weight"] for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.4)

    plt.title("HyperMemory Overview (memory hyperedges as nodes)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ================ main：按需选择要看的图 ================

def main():
    # 1) 尝试画事件级超图（如果你之前跑过 sampling.py 保存了 event_hypergraphs）
    try:
        hg_list = load_event_hypergraphs()
        if hg_list:
            hg = hg_list[0]  # 看第 0 条事件
            visualize_event_hypergraph_sets(hg, max_edges=6)
            visualize_event_hypergraph_edge_ring(hg, max_edges=6)
            # visualize_event_hypergraph_projection(hg)
    except FileNotFoundError as e:
        print(e)
        print("  跳过事件级超图可视化。")

    # 2) 画 HyperMemory 概览（需要先跑 connection.py 生成 hypermemory.pkl）
    try:
        mem = load_hypermemory()
        visualize_hypermemory_overview(mem, max_nodes=200, weight_threshold=0.0)
    except FileNotFoundError as e:
        print(e)
        print("  请先运行 `python src/connection.py` 构建并保存 HyperMemory。")


if __name__ == "__main__":
    main()
