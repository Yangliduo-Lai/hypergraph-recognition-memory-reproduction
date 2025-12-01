# 文件：src/sampling.py
"""
1. 定义 sampling 规则：
   - 给定一条 d 维事件 x（这里 d=8）
   - 按 edge configuration（fixed / random-order / random-combination）
     采样出一组“维度索引子集”，每个子集对应一条 hyperedge

2. 定义 EventHypergraph：
   - 对一条事件构造一个小 hypergraph：
       节点：8 个维度，对应 index 0..7，每个节点带有该 event 上的取值
       超边：维度索引的子集
       links：超边之间的顺序连接关系

3. 从 data/synthetic_events_data/events.csv 读取事件：
   - 为每条事件构建 EventHypergraph
   - 保存到 data/event_hypergraphs/event_hypergraphs.pkl
"""

from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional, Dict
from typing import Literal
from pathlib import Path
import pickle

import numpy as np


# 一条超边：若干“维度索引”（node index）
HyperEdge = Tuple[int, ...]


# ======================== 采样配置 & 函数 ========================

@dataclass
class SamplingConfig:
    n_attrs: int = 8
    mode: Literal["ring", "linear"] = "ring"
    edge_type: Literal["fixed", "random_order", "random_combination"] = "random_order"

    # 固定阶数
    k: int = 3

    # 随机阶数范围（含端点）
    k_min: int = 2
    k_max: int = 5


def _choose_k(cfg: SamplingConfig, rng: np.random.Generator) -> int:
    if cfg.edge_type == "fixed":
        return max(1, min(cfg.k, cfg.n_attrs))
    else:
        low = max(1, cfg.k_min)
        high = max(low, min(cfg.k_max, cfg.n_attrs))
        return int(rng.integers(low, high + 1))


def _contiguous_indices(start: int, k: int, d: int, mode: str) -> List[int]:
    if mode == "ring":
        return [(start + i) % d for i in range(k)]
    else:  # linear
        if start + k > d:
            return []
        return list(range(start, start + k))


def sample_hyperedge_indices(
    x: Sequence[int],
    cfg: SamplingConfig,
    rng: Optional[np.random.Generator] = None,
) -> List[HyperEdge]:
    """
    对一条事件 x 进行 sampling，生成“维度索引子集”的列表。
    """
    if rng is None:
        rng = np.random.default_rng()

    d = cfg.n_attrs
    assert len(x) == d, f"事件长度 {len(x)} 与 cfg.n_attrs={d} 不一致"

    edges: List[HyperEdge] = []

    if cfg.edge_type in ("fixed", "random_order"):
        for start in range(d):
            k = _choose_k(cfg, rng)
            indices = _contiguous_indices(start, k, d, cfg.mode)
            if not indices:
                continue
            edges.append(tuple(indices))

    elif cfg.edge_type == "random_combination":
        num_edges = d
        all_indices = list(range(d))

        for _ in range(num_edges):
            k = _choose_k(cfg, rng)
            if k > d:
                k = d
            chosen_indices = list(rng.choice(all_indices, size=k, replace=False))
            chosen_indices.sort()
            edges.append(tuple(chosen_indices))
    else:
        raise ValueError(f"未知 edge_type: {cfg.edge_type}")

    return edges


# ======================== 事件级 Hypergraph ========================

@dataclass
class Node:
    index: int
    value: int


@dataclass
class EventHypergraph:
    event_id: int
    nodes: Dict[int, Node]                  # node_id -> Node（这里 node_id 就是维度 index）
    hyperedges: Dict[int, HyperEdge]        # edge_id -> tuple(indices)
    links: List[Tuple[int, int]]            # (edge_i, edge_j)


def build_event_hypergraph(
    x: Sequence[int],
    cfg: SamplingConfig,
    event_id: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> EventHypergraph:
    if rng is None:
        rng = np.random.default_rng()

    d = cfg.n_attrs
    assert len(x) == d, f"事件长度 {len(x)} 与 cfg.n_attrs={d} 不一致"

    # 1. 节点
    nodes: Dict[int, Node] = {i: Node(index=i, value=int(x[i])) for i in range(d)}

    # 2. 超边（只是维度索引子集）
    edge_index_list: List[HyperEdge] = sample_hyperedge_indices(x, cfg, rng=rng)

    hyperedges: Dict[int, HyperEdge] = {eid: idx_tuple for eid, idx_tuple in enumerate(edge_index_list)}

    # 3. 在超边序列上建环形 link
    links: List[Tuple[int, int]] = []
    m = len(hyperedges)
    if m > 0:
        for eid in range(m):
            j = (eid + 1) % m
            links.append((eid, j))

    return EventHypergraph(
        event_id=event_id,
        nodes=nodes,
        hyperedges=hyperedges,
        links=links,
    )


# ======================== 读 CSV + 构建 & 保存 ========================

def load_events_from_csv() -> np.ndarray:
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "synthetic_events_data" / "events.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到事件文件: {csv_path}")

    events = np.loadtxt(csv_path, delimiter=",", dtype=int)
    if events.ndim == 1:
        events = events.reshape(1, -1)

    print(f"[加载] 事件矩阵形状: {events.shape}")
    return events


def build_all_event_hypergraphs(
    events: np.ndarray,
    cfg: SamplingConfig,
) -> List[EventHypergraph]:
    n_events, n_attrs = events.shape
    print(f"[构建] 共 {n_events} 条事件，每条 {n_attrs} 维。")

    hg_list: List[EventHypergraph] = []
    rng = np.random.default_rng(0)

    for eid in range(n_events):
        x = events[eid]
        hg = build_event_hypergraph(x, cfg, event_id=eid, rng=rng)
        hg_list.append(hg)
        if (eid + 1) % 1000 == 0:
            print(f"  已构建 {eid+1}/{n_events} 个事件超图...")

    print("[构建] 所有事件的 EventHypergraph 构建完成。")
    return hg_list


def save_event_hypergraphs(
    hg_list: List[EventHypergraph],
    filename: str = "event_hypergraphs.pkl",
) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    save_dir = project_root / "data" / "event_hypergraphs"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / filename
    with open(save_path, "wb") as f:
        pickle.dump(hg_list, f)

    print(f"[保存] 事件级超图列表已保存到: {save_path}")
    return save_path


def main():
    events = load_events_from_csv()

    cfg = SamplingConfig(
        n_attrs=events.shape[1],
        mode="ring",
        edge_type="random_order",
        k_min=2,
        k_max=5,
    )

    hg_list = build_all_event_hypergraphs(events, cfg)
    save_event_hypergraphs(hg_list)


if __name__ == "__main__":
    main()
