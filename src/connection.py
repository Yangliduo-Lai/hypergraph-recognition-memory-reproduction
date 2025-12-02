# 文件：src/connection.py
"""
实现 HyperMemory 的 connection & familiarity judgment 逻辑。

核心概念：

- NodeKey: (维度 index, 该维的取值)，比如 (0, 15) 表示 x1=15
- Memory hyperedge: 若干 NodeKey 的集合，表示跨事件累积的“记忆片段”
- HyperMemory: 记忆系统，里面存了很多 memory hyperedge 以及它们之间的 link

当有一个新事件进来时（已经在 sampling 中变成 EventHypergraph）：

1) familiarity_judgment:
   - 对事件中的每一条“事件超边” e_q：
       * 把它变成一组 NodeKey = {(i, value_i), ...}
       * 跟记忆中的每一条 memory hyperedge 做匹配：
           匹配条件 = “部分匹配 + 无值冲突”
           实现为：
             (a) 先看两条边在维度上的交集 common_dims = dims(query) ∩ dims(memory)
             (b) 如果 common_dims 为空 → 不匹配
             (c) 对每个 dim ∈ common_dims，value 必须相同，否则视为值冲突 → 不匹配
             (d) 否则 overlap = |common_dims|，视为一条匹配

       * 满足上述条件的 memory hyperedge 即为“被激活的超边”

   - 如果有任何一条事件超边完全没有匹配 → 直接判定为 new。

   - 否则，把所有激活的 memory hyperedge 组成集合 activated_edges，
     如果在记忆的 link 图中，activated_edges 诱导的子图存在一个环，则判定为 old，
     否则为 new。

2) integrate_event:
   - 如果事件被判定为 new：
       * 对每一条事件超边：
           - 若有匹配的 memory hyperedge，就选 overlap 最大的那一条复用
           - 若完全没匹配，就新建一条 memory hyperedge
       * 然后根据事件内的环结构，在这些 memory hyperedge 之间累积 link（次数→权重）

demo():
   - 记忆从空开始，按顺序一条一条事件读入，依次做 old/new 判断并更新记忆
   - 结束后将 HyperMemory 保存到 data/hypermemory/hypermemory.pkl
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Set, FrozenSet

from pathlib import Path
import pickle
import numpy as np

from sampling import (
    SamplingConfig,
    EventHypergraph,
    build_event_hypergraph,
    load_events_from_csv,
)

# (维度 index, 该维的取值)
NodeKey = Tuple[int, int]


@dataclass
class HyperMemory:
    """
    全局记忆超图：

      - edges:      edge_id -> frozenset[NodeKey]
      - edge_sigs:  frozenset[NodeKey] -> edge_id
      - link_counts / link_weights: memory hyperedge 之间的连接及其权重
    """

    n_attrs: int
    C: float = 5.0  # 控制 link 权重 sigmoid 的常数

    edges: Dict[int, FrozenSet[NodeKey]] = field(default_factory=dict)
    edge_sigs: Dict[FrozenSet[NodeKey], int] = field(default_factory=dict)
    next_edge_id: int = 0

    link_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)
    link_weights: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # ---------- 基本工具 ----------

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _canonical_pair(i: int, j: int) -> Tuple[int, int]:
        return (i, j) if i <= j else (j, i)

    def _get_or_create_edge(self, nodes: Set[NodeKey]) -> int:
        """
        根据一组 NodeKey（(attr, value)）获取或创建一条 memory hyperedge。
        """
        sig = frozenset(nodes)
        if sig in self.edge_sigs:
            return self.edge_sigs[sig]

        edge_id = self.next_edge_id
        self.next_edge_id += 1
        self.edge_sigs[sig] = edge_id
        self.edges[edge_id] = sig
        return edge_id

    def _update_link(self, edge_i: int, edge_j: int):
        """
        累积 edge_i 与 edge_j 之间的 link 计数，并更新权重。
        """
        if edge_i == edge_j:
            return

        key = self._canonical_pair(edge_i, edge_j)
        count = self.link_counts.get(key, 0) + 1
        self.link_counts[key] = count
        # 简单 sigmoid 把次数映射到 [0,1]
        self.link_weights[key] = self._sigmoid(count / self.C)

    # ---------- 匹配逻辑（部分匹配 + 无值冲突） ----------

    @staticmethod
    def _build_dim_to_val(nodes: Set[NodeKey]) -> Dict[int, int]:
        """
        把 NodeKey 集合 {(dim, val), ...} 转成 dim -> val 的字典。
        假设同一 dim 不会出现多个 val。
        """
        return {dim: val for (dim, val) in nodes}

    @staticmethod
    def _match_query_to_memory(
        query_nodes: Set[NodeKey],
        mem_nodes: Set[NodeKey],
    ) -> Tuple[bool, int]:
        """
        判断一条“查询超边”（query_nodes）是否与一条记忆超边（mem_nodes）匹配。

        匹配条件：

          1. 至少有一个维度 dim，两条边都包含（有交集）：
               common_dims = dims(query) ∩ dims(memory)
             如果 common_dims 为空 → 不匹配

          2. 无值冲突：
               对每个 dim ∈ common_dims，
               记忆中的值必须和新超边中的值相同，
               如果有 dim 的值不同，就判定为冲突（不匹配）。

          3. overlap 返回“共同维度中值相同”的个数（如果都不冲突，就是 |common_dims|）。
        """
        q = HyperMemory._build_dim_to_val(query_nodes)
        m = HyperMemory._build_dim_to_val(mem_nodes)

        dims_q = set(q.keys())
        dims_m = set(m.keys())

        common_dims = dims_q & dims_m
        if not common_dims:
            # 完全没有共同维度 → 谈不上部分匹配
            return False, 0

        overlap = 0
        for dim in common_dims:
            if q[dim] != m[dim]:
                # 在共同维度上出现不同的值 → 冲突 → 不匹配
                return False, 0
            overlap += 1

        # 走到这里说明在 common_dims 上完全一致，且 common_dims 非空
        return True, overlap

    def _event_edge_to_nodekeys(self, hg: EventHypergraph, edge_id: int) -> Set[NodeKey]:
        """
        把事件的小超图中的一条 hyperedge（维度索引子集）
        转成一组 NodeKey = {(dim_index, value_at_this_dim), ...}
        """
        idx_tuple = hg.hyperedges[edge_id]  # 比如 (0,1,2)
        return {(dim_idx, hg.nodes[dim_idx].value) for dim_idx in idx_tuple}

    # ---------- 环检测：激活的记忆超边能否构成一个环 ----------

    def _has_cycle_among(self, active_edges: Set[int]) -> bool:
        """
        在记忆的 link 图中，检查由 active_edges 诱导的子图里是否存在环。

        active_edges: 被激活的记忆超边 id 集合。
        """
        if not active_edges:
            return False

        # 构建邻接表，仅保留 active_edges 之间、权重大于 0 的连接
        adj: Dict[int, List[int]] = {e: [] for e in active_edges}
        for (i, j), w in self.link_weights.items():
            if w <= 0.0:
                continue
            if i in active_edges and j in active_edges:
                adj[i].append(j)
                adj[j].append(i)

        visited: Set[int] = set()

        for start in active_edges:
            if start in visited:
                continue
            if self._dfs_cycle(start, parent=-1, visited=visited, adj=adj):
                return True

        return False
    
    def _sum_link_weights_among(self, active_edges: set[int]) -> float:
        """
        计算在 active_edges 所诱导的子图中，所有 link 权重之和。
        用来做 familiarity 的结构部分（越大说明这些边在记忆里越“常见”）。
        """
        if not active_edges:
            return 0.0

        total = 0.0
        for (i, j), w in self.link_weights.items():
            if i in active_edges and j in active_edges:
                total += w
        return float(total)

    def _dfs_cycle(
        self,
        v: int,
        parent: int,
        visited: Set[int],
        adj: Dict[int, List[int]],
    ) -> bool:
        """
        在无向图里用 DFS 检测环：
          - 如果访问到一个已经访问过、且不是父节点的邻居，就说明存在环。
        """
        visited.add(v)
        for u in adj.get(v, []):
            if u not in visited:
                if self._dfs_cycle(u, parent=v, visited=visited, adj=adj):
                    return True
            elif u != parent:
                # 访问到一个已访问过、且不是父节点的点 → 有环
                return True
        return False

    # ---------- Step 1: familiarity judgment ----------

    def familiarity_judgment(self, hg) -> dict:
        """
        对一条事件级超图 hg 做熟悉度判断。

        额外返回：
          - num_active_edges   : 被激活的记忆超边数量
          - sum_link_weights   : 激活子图中 link 权重之和
          - familiarity_score  : 一个简单的一维熟悉度分数，用于 ROC：
                                 familiarity_score = edge_coverage * num_active_edges
                                                     + sum_link_weights
        """
        # 记忆还是空的情况
        if not self.edges:
            return {
                "is_old": False,
                "matches_per_edge": {eid: [] for eid in hg.hyperedges.keys()},
                "edge_coverage": 0.0,
                "link_has_cycle": False,
                "num_active_edges": 0,
                "sum_link_weights": 0.0,
                "familiarity_score": 0.0,
            }

        matches_per_edge: dict[int, list[dict]] = {}

        # --- 1) 对每条事件超边找匹配的 memory hyperedges ---
        for e_id in hg.hyperedges.keys():
            # 事件里的这一条超边对应的节点 key 集合
            q_nodes = self._event_edge_to_nodekeys(hg, e_id)
            candidates: list[dict] = []

            for mem_eid, mem_nodes in self.edges.items():
                is_match, overlap = self._match_query_to_memory(q_nodes, set(mem_nodes))
                if is_match:
                    candidates.append({"mem_edge_id": mem_eid, "overlap": overlap})

            matches_per_edge[e_id] = candidates

        total_edges = len(hg.hyperedges)
        matched_edges = sum(1 for _, cands in matches_per_edge.items() if cands)
        edge_coverage = matched_edges / total_edges if total_edges > 0 else 0.0

        # --- 2) 统计被激活的记忆超边 & link 权重 ---
        activated_edges: set[int] = set()
        for _, cands in matches_per_edge.items():
            for c in cands:
                activated_edges.add(c["mem_edge_id"])

        num_active_edges = len(activated_edges)
        sum_link_weights = self._sum_link_weights_among(activated_edges)

        # 简单的一维熟悉度分数：局部覆盖 * 激活边数 + 结构权重
        familiarity_score = edge_coverage * float(num_active_edges) + sum_link_weights

        # --- 3) old/new 判定逻辑保持原样 ---
        # 如果有些事件超边根本找不到匹配，直接判 new
        if matched_edges < total_edges:
            return {
                "is_old": False,
                "matches_per_edge": matches_per_edge,
                "edge_coverage": edge_coverage,
                "link_has_cycle": False,
                "num_active_edges": num_active_edges,
                "sum_link_weights": sum_link_weights,
                "familiarity_score": familiarity_score,
            }

        # 所有事件超边都有匹配 → 看激活子图中是否存在环
        has_cycle = self._has_cycle_among(activated_edges)
        is_old = has_cycle

        return {
            "is_old": is_old,
            "matches_per_edge": matches_per_edge,
            "edge_coverage": edge_coverage,
            "link_has_cycle": has_cycle,
            "num_active_edges": num_active_edges,
            "sum_link_weights": sum_link_weights,
            "familiarity_score": familiarity_score,
        }


    # ---------- Step 2: 整合 new 事件到记忆里 ----------

    def _integrate_event(
        self,
        hg: EventHypergraph,
        matches_per_edge: Dict[int, List[Dict]],
    ) -> Dict[int, int]:
        """
        把一条事件的小超图整合进记忆中。

        规则：
          - 对每一条事件超边 e_q：
              * 如果 matches_per_edge[e_q] 非空：
                    找 overlap 最高的 memory hyperedge 复用
              * 否则：
                    用该超边的 NodeKey 集合新建一条 memory hyperedge

          - 然后按照 hg.links（事件内部的环结构），在对应的 memory hyperedge
            之间更新 link_counts / link_weights。

        返回：
          event_edge_to_mem_edge: 事件超边 id -> 记忆超边 id
        """
        event_to_mem: Dict[int, int] = {}

        # 1) 决定每条事件超边映射到哪条 memory hyperedge
        for e_id in hg.hyperedges.keys():
            candidates = matches_per_edge.get(e_id, [])
            q_nodes = self._event_edge_to_nodekeys(hg, e_id)

            if candidates:
                # 选 overlap 最大的那条记忆超边（如果有并列，随便取一个）
                best = max(candidates, key=lambda d: d["overlap"])
                mem_eid = best["mem_edge_id"]
            else:
                # 完全没匹配，新建
                mem_eid = self._get_or_create_edge(q_nodes)

            event_to_mem[e_id] = mem_eid

        # 2) 按事件环结构更新 memory link
        for (e_i, e_j) in hg.links:
            m_i = event_to_mem[e_i]
            m_j = event_to_mem[e_j]
            self._update_link(m_i, m_j)

        return event_to_mem

    def add_event_as_new(self, hg: EventHypergraph):
        """
        用于“强制当作 new 写入记忆”的接口：不做熟悉度判断。
        """
        empty_matches = {e_id: [] for e_id in hg.hyperedges.keys()}
        self._integrate_event(hg, empty_matches)

    def process_event(self, hg: EventHypergraph) -> Dict:
        """
        对一条事件执行完整流程：

          1) familiarity_judgment
          2) 如果 is_old=False，则把它整合进记忆（匹配的超边复用，不匹配的超边新建）

        返回：
          {
            "is_old": bool,
            "edge_coverage": float,
            "link_has_cycle": bool,
            "matches_per_edge": {...},
          }
        """
        fam = self.familiarity_judgment(hg)

        if not fam["is_old"]:
            self._integrate_event(hg, fam["matches_per_edge"])

        return fam


# ===================== HyperMemory 的存取 =====================

def save_hypermemory(mem: HyperMemory, filename: str = "hypermemory.pkl") -> Path:
    """
    将 HyperMemory 序列化保存到 data/hypermemory/ 下。
    """
    project_root = Path(__file__).resolve().parent.parent
    save_dir = project_root / "data" / "hypermemory"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / filename
    with open(save_path, "wb") as f:
        pickle.dump(mem, f)

    print(f"[保存] HyperMemory 已保存到: {save_path}")
    return save_path


def load_hypermemory(filename: str = "hypermemory.pkl") -> HyperMemory:
    """
    从 data/hypermemory/ 读取 HyperMemory（一般给可视化用）。
    """
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / "data" / "hypermemory" / filename

    if not path.exists():
        raise FileNotFoundError(f"找不到 HyperMemory 文件: {path}")

    with open(path, "rb") as f:
        mem: HyperMemory = pickle.load(f)

    print(f"[加载] HyperMemory 来自: {path}")
    return mem


# ===================== 顺序读入事件的 demo =====================

def demo():
    """
    从 events.csv 读事件，记忆从空开始，
    按顺序一条一条事件读入，依次做 old/new 判断并更新记忆。
    最后将 HyperMemory 保存到 data/hypermemory/hypermemory.pkl。
    """
    events = load_events_from_csv()
    cfg = SamplingConfig(
        n_attrs=events.shape[1],
        mode="ring",
        edge_type="random_order",
        k_min=2,
        k_max=5,
    )

    mem = HyperMemory(n_attrs=events.shape[1])
    rng = np.random.default_rng(0)

    print(f"[顺序处理] 共 {events.shape[0]} 条事件，记忆初始为空。")

    for i in range(events.shape[0]):
        hg = build_event_hypergraph(events[i], cfg, event_id=i, rng=rng)
        result = mem.process_event(hg)
        print(
            f"  Event {i}: "
            f"is_old={result['is_old']}, "
            f"edge_cov={result['edge_coverage']:.2f}, "
            f"link_has_cycle={result['link_has_cycle']}, "
            f"memory_edges={len(mem.edges)}, links={len(mem.link_weights)}"
        )

    # 最后把 HyperMemory 存起来
    save_hypermemory(mem)


if __name__ == "__main__":
    demo()
