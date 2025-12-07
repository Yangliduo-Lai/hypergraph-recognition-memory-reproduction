# 文件：src/pattern_completion.py
"""
基于 HyperMemory 的 pattern completion 脚本（环状结构版，贴论文逻辑）。

假设 HyperMemory 的结构为：
    - n_attrs: int
    - edges: Dict[int, frozenset[(attr_idx, value)]]
    - link_counts: Dict[(edge_i, edge_j), int]

核心逻辑（只考虑 ring-type 结构）：
    1. 给定部分事件（partial_event），用“部分匹配 + 无冲突”的 δ 规则
       在 HyperMemory 中激活一批超边（memory hyperedges）。
    2. 在这些激活的超边上，利用 link_counts 构造子图，
       在子图中搜索“闭环”（simple cycles）。
    3. 对每一个候选环：
         - 把环上的所有 (attr_idx, value) 合在一起：
             * 若同一维度出现多个不同值 → 冲突，丢弃；
             * 若所有维度 0..n_attrs-1 都至少出现一次 → 完备事件。
       这样得到的完整事件就是 pattern completion 的结果。
    4. Completeness = 是否存在至少一个这样的环；
       这里我们选一个得分最高的环（按环上 link_counts 之和）作为 best_completed_event。

使用示例（在项目根目录）：

    # 1) 默认：随机从 events.csv 中抽一条事件，随机遮掉 3 维，然后补全
    python src/pattern_completion.py

    # 2) 手动指定部分事件向量（长度必须为 n_attrs，比如 8 维）
    python src/pattern_completion.py --partial 2,22,2,8,-1,-1,0,-1

    # 3) 使用指定 index 的事件，遮掉 4 个维度
    python src/pattern_completion.py --event-index 10 --n-missing 4 --seed 123
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Sequence

import numpy as np

# ---- 把 src 加到 sys.path，方便 import ----
SRC_DIR = Path(__file__).resolve().parent  # .../src
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from connection import HyperMemory  # type: ignore
except Exception:
    HyperMemory = object  # 类型兜底，不影响运行


# ========== 辅助：从一条超边里拿出 (attr_idx, value) ==========

def _edge_attr_pairs(edge, n_attrs: int) -> List[Tuple[int, int]]:
    """
    针对当前 HyperMemory 结构：
        edges: Dict[int, frozenset[(attr_idx, value)]]

    直接把 frozenset 里的元素当成 (attr_idx, value) 对解析，
    同时兼容 list/tuple 的情况。
    """
    pairs: List[Tuple[int, int]] = []

    if isinstance(edge, (set, frozenset, list, tuple)):
        for item in edge:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                try:
                    a = int(item[0])
                    v = int(item[1])
                except Exception:
                    continue
                if 0 <= a < n_attrs:
                    pairs.append((a, v))
        return pairs

    return pairs


def _get_link_weight_from_counts(mem: HyperMemory, e1: int, e2: int) -> float:
    """
    用 link_counts 作为边权重：
        link_counts[(i, j)] = count

    为了稳妥，我们尝试 (e1, e2)、(e2, e1) 两种顺序。
    """
    if not hasattr(mem, "link_counts"):
        return 0.0
    lc = mem.link_counts
    if (e1, e2) in lc:
        return float(lc[(e1, e2)])
    if (e2, e1) in lc:
        return float(lc[(e2, e1)])
    return 0.0


# ========== 核心：基于环的 pattern completion（论文式） ==========

def pattern_completion(
    mem: HyperMemory,
    partial_event: Sequence[int],
    missing_value: int = -1,
    max_cycle_len: int | None = None,
) -> Dict[str, Any]:
    """
    严格按“环状结构 + δ 激活”的方式做模式补全：

        1. δ 规则激活记忆超边：
             - 至少一个 (attr_idx, value) 和已知 cue 一致（overlap）
             - 不得在已知维度上出现冲突值
        2. 在激活边上，用 link_counts 构造邻接图，搜索简单闭环；
        3. 对每个闭环，检查：
             - 同一 attr 不能有多个值（自洽）；
             - 覆盖所有维度 0..n_attrs-1；
             - cue 中的已知维度必须匹配；
        4. 符合条件的环 → 完整事件候选；
           以环上 link_counts 总和为 score，选最高的一个。
    """
    x = np.asarray(partial_event, dtype=int)
    if x.ndim != 1:
        raise ValueError(f"partial_event 必须是一维向量，当前 ndim={x.ndim}")
    if not hasattr(mem, "n_attrs"):
        raise ValueError("HyperMemory 实例缺少 n_attrs 属性。")
    if x.shape[0] != mem.n_attrs:
        raise ValueError(
            f"partial_event 长度为 {x.shape[0]}，但 HyperMemory.n_attrs = {mem.n_attrs}"
        )

    n_attrs = mem.n_attrs

    if not hasattr(mem, "edges"):
        raise ValueError("HyperMemory 实例缺少 edges 属性，无法进行补全。")
    if not hasattr(mem, "link_counts"):
        raise ValueError("HyperMemory 实例缺少 link_counts 属性，无法进行环状补全。")

    # 1. 提取 cue（已知维度）
    known_attrs: Dict[int, int] = {
        idx: int(val)
        for idx, val in enumerate(x)
        if val != missing_value
    }
    if not known_attrs:
        raise ValueError("pattern_completion: partial_event 中没有任何已知维度，无法进行补全。")

    # 2. δ 规则激活记忆超边
    activated_edges: List[int] = []

    for edge_id, edge_sig in mem.edges.items():
        pairs = _edge_attr_pairs(edge_sig, n_attrs)
        if not pairs:
            continue

        edge_attr: Dict[int, int] = {}
        for a, v in pairs:
            edge_attr[a] = v

        has_overlap = False
        conflict = False
        for a_k, v_k in known_attrs.items():
            if a_k in edge_attr:
                if edge_attr[a_k] == v_k:
                    has_overlap = True
                else:
                    conflict = True
                    break

        if conflict or not has_overlap:
            continue

        activated_edges.append(edge_id)

    n_support_edges = len(activated_edges)
    if n_support_edges == 0:
        return {
            "partial_event": x,
            "known_attrs": known_attrs,
            "activated_edges": [],
            "completeness": False,
            "best_completed_event": None,
            "completed_events": [],
            "cycle_infos": [],
            "n_support_edges": 0,
            "used_cycle": None,
        }

    # 3. 在激活子图上构造邻接表（只用 link_counts）
    from collections import defaultdict as _dd

    adj: Dict[int, set[int]] = _dd(set)
    for (e1, e2), cnt in mem.link_counts.items():
        if cnt <= 0:
            continue
        if e1 in activated_edges and e2 in activated_edges:
            adj[e1].add(e2)
            adj[e2].add(e1)

    if not adj:
        return {
            "partial_event": x,
            "known_attrs": known_attrs,
            "activated_edges": activated_edges,
            "completeness": False,
            "best_completed_event": None,
            "completed_events": [],
            "cycle_infos": [],
            "n_support_edges": n_support_edges,
            "used_cycle": None,
        }

    # 4. 搜索简单闭环
    if max_cycle_len is None:
        max_cycle_len = max(3, 2 * n_attrs)

    nodes = list(adj.keys())
    cycles: set[Tuple[int, ...]] = set()

    def dfs(start: int, current: int, path: List[int], visited: set[int]):
        for nb in adj[current]:
            if nb == start and len(path) >= 3:
                cycles.add(tuple(sorted(path)))
            elif nb not in visited and len(path) < max_cycle_len:
                visited.add(nb)
                dfs(start, nb, path + [nb], visited)
                visited.remove(nb)

    for s in nodes:
        dfs(s, s, [s], {s})

    if not cycles:
        return {
            "partial_event": x,
            "known_attrs": known_attrs,
            "activated_edges": activated_edges,
            "completeness": False,
            "best_completed_event": None,
            "completed_events": [],
            "cycle_infos": [],
            "n_support_edges": n_support_edges,
            "used_cycle": None,
        }

    # 5. 对每个环检查：自洽 + 完备 + cue 匹配
    completed_events: List[np.ndarray] = []
    cycle_infos: List[Dict[str, Any]] = []

    for cyc in cycles:
        attr_to_val: Dict[int, int] = {}
        valid = True

        for eid in cyc:
            edge_sig = mem.edges[eid]
            pairs = _edge_attr_pairs(edge_sig, n_attrs)
            for a, v in pairs:
                if a in attr_to_val and attr_to_val[a] != v:
                    valid = False
                    break
                attr_to_val[a] = v
            if not valid:
                break

        if not valid:
            continue

        # cue 必须被覆盖，且值一致
        for a_k, v_k in known_attrs.items():
            if a_k not in attr_to_val or attr_to_val[a_k] != v_k:
                valid = False
                break
        if not valid:
            continue

        # 完备性：覆盖所有维度
        if any(a not in attr_to_val for a in range(n_attrs)):
            continue

        ev = np.array([attr_to_val[a] for a in range(n_attrs)], dtype=int)

        # 环得分：按 link_counts 累加
        cyc_list = list(cyc)
        score = 0.0
        if len(cyc_list) > 1:
            for i in range(len(cyc_list)):
                e1 = cyc_list[i]
                e2 = cyc_list[(i + 1) % len(cyc_list)]
                score += _get_link_weight_from_counts(mem, e1, e2)

        completed_events.append(ev)
        cycle_infos.append({
            "edge_ids": cyc_list,
            "event": ev,
            "score": score,
        })

    completeness = len(completed_events) > 0
    best_event = None
    used_cycle = None

    if completeness:
        best_idx = max(
            range(len(cycle_infos)),
            key=lambda i: (cycle_infos[i]["score"], -len(cycle_infos[i]["edge_ids"])),
        )
        best_event = completed_events[best_idx]
        used_cycle = cycle_infos[best_idx]

    return {
        "partial_event": x,
        "known_attrs": known_attrs,
        "activated_edges": activated_edges,
        "completeness": completeness,
        "best_completed_event": best_event,
        "completed_events": completed_events,
        "cycle_infos": cycle_infos,
        "n_support_edges": n_support_edges,
        "used_cycle": used_cycle,
    }


def pretty_print_completion(result: Dict[str, Any]) -> None:
    """
    打印 pattern_completion 的结果，突出：
        - partial_event
        - 激活超边数量
        - Completeness
        - 最佳补全事件
        - 所用闭环的边 ID 和得分
    """
    partial = result["partial_event"]
    best = result["best_completed_event"]
    completeness = result["completeness"]
    known_attrs = result["known_attrs"]
    n_support = result["n_support_edges"]
    used_cycle = result["used_cycle"]

    print("=== Pattern Completion 结果（环状结构版） ===")
    print("部分输入 (partial_event):")
    print("  ", partial.tolist())
    print()
    print(f"用于支持补全的激活超边数量: {n_support}")
    print(f"是否找到完整环 (Completeness): {completeness}")
    print()

    if not completeness:
        print("未能在激活子图中找到覆盖所有维度且自洽的闭环，无法按论文逻辑完成补全。")
        return

    print("最佳补全事件 (best_completed_event):")
    print("  ", best.tolist())
    print()

    if used_cycle is not None:
        print("用于补全的环信息：")
        print("  边 ID 列表:", used_cycle["edge_ids"])
        print("  环得分 (link_counts 之和):", f"{used_cycle['score']:.2f}")


# ========== 从 pickle 加载 HyperMemory ==========

def load_hypermemory_from_pickle() -> HyperMemory:
    project_root = SRC_DIR.parent
    pkl_path = project_root / "data" / "hypermemory" / "hypermemory.pkl"

    if not pkl_path.exists():
        raise FileNotFoundError(f"找不到 HyperMemory pickle 文件: {pkl_path}")

    with open(pkl_path, "rb") as f:
        mem = pickle.load(f)

    n_attrs = getattr(mem, "n_attrs", None)
    n_edges = len(getattr(mem, "edges", {}))
    print(f"[加载] 已从 {pkl_path} 加载 HyperMemory")
    print(f"       n_attrs={n_attrs}, #edges={n_edges}")
    return mem


def load_events_or_generate(seed: int = 0) -> np.ndarray:
    """
    优先从 data/synthetic_events_data/events.csv 读取事件，
    如果失败，就在线生成一批 synthetic events。
    """
    try:
        from sampling import load_events_from_csv  # type: ignore
        events = load_events_from_csv()
        print(f"[数据] 已从 events.csv 读取 {events.shape[0]} 条事件。")
        return events
    except Exception as e:
        print("[警告] 读取 events.csv 失败，将在线生成 synthetic events。")
        print("       错误信息:", repr(e))
        from synthetic_events import EventGeneratorConfig, generate_synthetic_events  # type: ignore
        cfg = EventGeneratorConfig(n_events=1000, seed=seed)
        events = generate_synthetic_events(cfg)
        print(f"[数据] 生成 synthetic events: {events.shape[0]} 条。")
        return events


# ========== 命令行入口 ==========

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="基于 HyperMemory 的环状 pattern completion Demo（论文复现版）"
    )
    parser.add_argument(
        "--partial",
        type=str,
        default=None,
        help="手动指定部分事件向量，逗号分隔，例如: 2,22,2,8,-1,-1,0,-1 。"
             "如果提供该参数，则不会去读 events.csv。",
    )
    parser.add_argument(
        "--event-index",
        "-i",
        type=int,
        default=-1,
        help="从 events 中选哪一条作为完整事件；"
             "默认 -1 表示随机抽取一条。",
    )
    parser.add_argument(
        "--n-missing",
        "-k",
        type=int,
        default=3,
        help="如果不指定 --partial，则随机遮掉多少个维度（默认 3）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（用于选择事件 & 遮掉哪些维度）。不指定则每次运行都会随机。",
    )

    args = parser.parse_args()

    # 1. 加载 HyperMemory
    mem = load_hypermemory_from_pickle()

    # 2. 构造 partial_event
    if args.partial is not None:
        partial_vec = np.array([int(t) for t in args.partial.split(",")], dtype=int)
        print("[输入] 使用命令行提供的 partial 向量：")
        print("       ", partial_vec.tolist())
    else:
        events = load_events_or_generate(seed=args.seed)

        # 如果用户没有指定 seed，就用系统随机种子（每次运行都不同）
        if args.seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(args.seed)

        # 如果 event_index >= 0，则用指定事件；否则随机抽一条
        if args.event_index >= 0:
            idx = args.event_index
        else:
            idx = int(rng.integers(0, events.shape[0]))


        if idx < 0 or idx >= events.shape[0]:
            raise IndexError(
                f"event_index={idx} 超出范围，当前共有 {events.shape[0]} 条事件。"
            )

        full = events[idx].astype(int)
        n_attrs = full.shape[0]

        k = max(0, min(args.n_missing, n_attrs))
        mask_dims = rng.choice(n_attrs, size=k, replace=False)

        partial_vec = full.copy()
        partial_vec[mask_dims] = -1

        print("[输入] 从 events 中选取完整事件：")
        print("       选中 index =", idx)
        print("       full =", full.tolist())
        print(f"       随机遮掉 {k} 个维度，mask_dims = {sorted(mask_dims.tolist())}")
        print("       partial =", partial_vec.tolist())

    # 3. 调用基于环的 pattern_completion
    result = pattern_completion(
        mem,
        partial_event=partial_vec,
        missing_value=-1,
    )

    # 4. 打印结果
    pretty_print_completion(result)


if __name__ == "__main__":
    main()
