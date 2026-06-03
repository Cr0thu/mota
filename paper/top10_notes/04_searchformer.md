# 04 - Searchformer

## 基本信息

- 论文：Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping
- 作者：Lucas Lehnert et al.
- 年份：2024
- 链接：https://arxiv.org/abs/2402.14083
- 本地 PDF：`paper/top10_pdfs/04_searchformer_2402.14083.pdf`
- 抽取文本：`paper/top10_extracted/04_searchformer_2402.14083.md`

## 一句话结论

训练 Transformer 直接输出答案不如训练它模仿搜索过程；有了搜索 trace，模型不仅更省数据，还能通过 bootstrapping 学出比原始 A* 更短的搜索动态。

## 论文解决的问题

LLM/Transformer 在规划任务上常常不稳定。直接给模型输入问题，让它输出最终计划，容易学到脆弱相关性。Searchformer 的问题是：能不能把符号规划算法的内部搜索过程也变成训练序列，让 Transformer 学“怎么搜”，而不是只学“答案是什么”。

这与魔塔高度相关。我们当前 solver 只输出最终路线，这等价于只给模型答案；但真正有价值的是搜索器在每个状态扩展了什么、剪掉了什么、为什么选择当前路径。

## 核心方法

论文把 A* 的执行过程 token 化：

- prompt：任务描述，例如迷宫或 Sokoban 初始状态。
- trace：A* 搜索过程中节点进入 frontier、进入 closed set、节点 cost 和 heuristic。
- plan：最终最优路径。

训练序列分两类：

1. solution-only：`<prompt><plan>`
2. search-augmented：`<prompt><trace><plan>`

模型是 encoder-decoder Transformer。encoder 读任务 prompt，decoder 自回归生成 trace 和 plan。关键不是让模型调用外部 A*，而是让它通过 next-token prediction 学会 A* 的动态。

论文进一步提出 search dynamics bootstrapping：

1. 先训练模型模仿非确定性 A* 的搜索轨迹。
2. 让模型对同一任务采样多个 trace。
3. 保留那些能得到最优 plan 且 trace 更短的样本。
4. 用更短 trace 微调模型。
5. 重复迭代。

最后的 Searchformer 不再只是 A* 复制品，而是学出更短的搜索方式。

## 实验结论

在 maze navigation 和 Sokoban 中，search-augmented 模型明显优于 solution-only，尤其在训练数据较少和任务更难时。Sokoban 测试中，Searchformer 在 45M 参数规模下达到约 93.7% optimal，并且第三轮 bootstrapping 后平均比训练用的 A* trace 少约 26.8% 搜索步骤。论文还报告了 solution-only 即使用更大模型，也不一定比小一些的 search-augmented 模型好。

这说明“中间搜索过程”本身是强监督信号。

## 对魔塔的启发

魔塔路线不是只需要一条专家路径，而是需要完整的搜索数据。当前项目应该把 solver 变成数据生产器，保存：

- 当前 state 的资源、地图消耗集合、flags。
- 候选宏动作列表。
- 每个动作的合法性与 mask 原因。
- 每个动作后的伤害、钥匙变化、拾取物、可达区域变化。
- heuristic 分解。
- dominance pruning 是否发生，以及被哪个状态支配。
- 当前 open queue 中该节点的 priority。
- 最终 route outcome。

这些数据可以训练三个东西：

1. 动作排序器：给搜索器排候选动作。
2. 剪枝判别器：预测当前状态是否值得继续。
3. 路线生成器：模仿 search trace 输出高层宏动作序列。

## 工程落地

建议新增 `artifacts/search_traces/*.jsonl`，每次 expansion 记录一行：

```json
{
  "node_id": 12345,
  "parent_id": 12301,
  "state": {...},
  "expanded_action": {...},
  "candidate_actions": [
    {"action": "...", "legal": true, "score": 12.3, "features": {...}},
    {"action": "...", "legal": false, "mask_reason": "need_blue_key"}
  ],
  "priority": -903.1,
  "dominance": {"pruned": false},
  "outcome_so_far": {"stage": "pre_shield", "boss_damage": 616}
}
```

短期可以先不用训练 Transformer，只用这些 trace 做因子回测和监督学习排序器。

## 局限

Searchformer 训练成本高，Sokoban trace 甚至可能接近十万 token。魔塔如果把 primitive move 全部 token 化会更长。因此必须用宏动作 trace，而不是方向键级别 trace。

## 给本项目的下一步

1. 改 solver：保存 expansion trace。
2. 把 final route、partial route、deadend route 都纳入数据。
3. 训练一个小模型预测 action ranking。
4. 用 learned ranking 替换或融合当前 hand-crafted heuristic。
