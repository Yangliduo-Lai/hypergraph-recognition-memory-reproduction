"""
生成 8 维离散事件向量的“人造原始数据”，不依赖任何真实日志。

特点：
- 每个事件是一个长度为 8 的整数向量：x = [x0, x1, ..., x7]
- 每一维有各自的取值范围（domain_sizes）
- 通过“原型模式 + 噪声”的方式，让事件流里面有重复结构，
  这样 HyperMemory 这种记忆模型就有东西可以记。

使用示例：

    from synthetic_events import EventGeneratorConfig, generate_synthetic_events

    cfg = EventGeneratorConfig(n_events=5000)
    events = generate_synthetic_events(cfg)

    # events.shape == (5000, 8)
    # 然后对每一行事件 x 调用 hm.encode_event(x)
"""

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from pathlib import Path


@dataclass
class EventGeneratorConfig:
    # 要生成的事件数量
    n_events: int = 100
    # 事件维度，论文用的是 8 维
    n_attrs: int = 8
    # 每一维的取值个数
    # 第 i 维的取值范围就是 0 .. domain_sizes[i]-1
    domain_sizes: Sequence[int] = (24, 30, 4, 20, 40, 2, 3, 10)

    # 潜在“模式”的数量：每个模式对应一组典型的 8 维取值
    n_patterns: int = 5

    # 模式强度：每个属性有多大概率直接采用模式的典型值，
    # 其余情况下会随机改成别的值（加入噪声）
    pattern_strength: float = 0.8  # 0.0 ~ 1.0

    # 控制事件序列中模式的切换速度
    # - 如果 = 0：每个事件独立随机选择模式
    # - 如果接近 1：模式很“粘”，长时间停留在同一个模式
    pattern_persistence: float = 0.9  # 0.0 ~ 1.0

    # 随机种子
    seed: int = 0


def generate_synthetic_events(cfg: EventGeneratorConfig) -> np.ndarray:
    """
    根据给定配置生成一串 8 维离散事件向量。

    返回：
        events: np.ndarray，形状为 (n_events, n_attrs)，元素为 int
    """
    rng = np.random.default_rng(cfg.seed)

    # 处理 domain_sizes：如果只给了一个数字，就复制 n_attrs 份
    if len(cfg.domain_sizes) == 1:
        domain_sizes = list(cfg.domain_sizes) * cfg.n_attrs
    else:
        domain_sizes = list(cfg.domain_sizes)
        assert len(domain_sizes) == cfg.n_attrs, \
            "domain_sizes 的长度必须为 1 或 n_attrs"

    n_events = cfg.n_events
    n_attrs = cfg.n_attrs
    n_patterns = cfg.n_patterns

    # 1. 先为每个“模式”生成一个原型向量（prototype）
    #    prototypes 的形状： (n_patterns, n_attrs)
    prototypes = np.zeros((n_patterns, n_attrs), dtype=int)
    for p in range(n_patterns):
        for a in range(n_attrs):
            prototypes[p, a] = rng.integers(0, domain_sizes[a])

    # 2. 按时间顺序生成事件流
    events = np.zeros((n_events, n_attrs), dtype=int)

    # 当前所处的模式 index
    current_pattern = rng.integers(0, n_patterns)

    for t in range(n_events):
        # 决定是否切换到新的模式
        if t > 0 and rng.random() > cfg.pattern_persistence:
            # 随机换一个不同的模式
            available = [p for p in range(n_patterns) if p != current_pattern]
            current_pattern = rng.choice(available)

        proto = prototypes[current_pattern]

        # 根据模式 + 噪声生成一个事件
        x = np.zeros(n_attrs, dtype=int)
        for a in range(n_attrs):
            if rng.random() < cfg.pattern_strength:
                # 直接使用当前模式的典型值
                x[a] = proto[a]
            else:
                # 加一点噪声：在该维的取值空间里随机选一个（可与 proto 不同）
                x[a] = rng.integers(0, domain_sizes[a])

        events[t] = x

    return events


def save_events_to_data_folder(events: np.ndarray, filename_prefix: str = "events"):
    """
    将生成的事件保存到 data/synthetic_events_data/ 目录中。
    会生成：
      - data/synthetic_events_data/<prefix>.npy
      - data/synthetic_events_data/<prefix>.csv
    """
    project_root = Path(__file__).resolve().parent.parent

    data_dir = project_root / "data" / "synthetic_events_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    npy_path = data_dir / f"{filename_prefix}.npy"
    csv_path = data_dir / f"{filename_prefix}.csv"

    # 保存为 numpy 二进制
    np.save(npy_path, events)

    # 保存为 csv（纯文本，逗号分隔，整型）
    np.savetxt(csv_path, events, fmt="%d", delimiter=",")

    print(f"已保存到: {npy_path}")
    print(f"已保存到: {csv_path}")


if __name__ == "__main__":
    cfg = EventGeneratorConfig(
        # 参数修改在这
    )
    evts = generate_synthetic_events(cfg)
    print("events shape:", evts.shape)
    print("前 5 条事件：")
    print(evts[:5])

    save_events_to_data_folder(evts, filename_prefix="events")
