本仓库是对论文 Hypergraph-Based Recognition Memory Model for Lifelong Experience 的部分功能复现，只考虑了完整的 events。

仓库文件结构：
- `synthetic_events.py`：生成人造 8 维事件。
- `sampling.py`：对生成的事件进行采样，得到小超图。
- `connection.py`：对新输入的事件判断 old/new，并维护 hyperMemory。
- `visualize.py`：可视化

### 1. 数据准备

这里使用代码生成的人造的事件。（Reality Mining dataset 管理员并没有回复我）
主要用于构建一个初始的 HyperMemory 库。

数据 belike：x = [x0, x1, ..., x7]
取值范围：domain_sizes = [24, 30, 4, 20, 40, 2, 3, 10]

```bash
# --n-events 事件数量
# --seed 就是 seed
# --prefix 文件名前缀
python src/synthetic_events.py --n-events 1000 --seed 42 --prefix events_10k_s123
```

### 2. 采样（Sampling）

Sampling 是输入事件（instance）进入记忆模型时的第一步。它的作用是：
把一个完整的事件拆分成多个 hyperedges（超边），以便存入 hypergraph（超图）里。

```bash
python src/sampling.py
```

### 3. 连接（Connection）

这一步会对每一条事件进行熟悉度判断，并进行 hyperMemory 的维护。

```bash
python src/connection.py
```

### 4. 模式补全（Pattern Completion）

```bash
python src/pattern_completion.py
```

### 5. 可视化

```bash
python src/visualize.py
```

# 实验

### 轻微变形的事件

```bash
# 默认：重复同一个事件 10 次
python src/evaluations/repeat_event_test.py
```

