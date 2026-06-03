# 08 - Policy Improvement by Planning with Gumbel

## 基本信息

- 论文：Policy Improvement by Planning with Gumbel
- 作者：Ivo Danihelka, Arthur Guez, Julian Schrittwieser, David Silver
- 年份：2021
- 链接：https://openreview.net/forum?id=bERaNdoegnO
- 本地 PDF：`paper/top10_pdfs/08_gumbel_muzero_bERaNdoegnO.pdf`
- 抽取文本：`paper/top10_extracted/08_gumbel_muzero_bERaNdoegnO.md`

## 一句话结论

Gumbel MuZero 把 AlphaZero/MuZero 中不少启发式动作选择替换成更有 policy improvement 保证的机制，在少量 simulations 时尤其有效。

## 论文解决的问题

AlphaZero/MuZero 在 root 节点用 PUCT、Dirichlet noise、visit count policy 等启发式机制。大搜索预算下这些机制工作得很好，但当 simulations 少于动作数时，可能根本没访问到关键动作，policy target 也未必是改进策略。

论文指出一个简单反例：如果 top probability 的动作并不是高价值动作，只访问高概率动作会让搜索结果比原 policy 还差。

## 核心方法

论文用 Gumbel-Top-k 做无放回采样：

```text
sample gumbel noise g
select top-k actions by g(a) + logits(a)
evaluate sampled actions
choose action by g(a) + logits(a) + sigma(q(a))
```

关键点是：采样和最终选择使用同一组 Gumbel 噪声，避免双重计数 bias。论文证明，在 Q 值准确时，这个过程能产生 policy improvement。

对于 stochastic bandit，论文把 Gumbel-Top-k 和 Sequential Halving 结合，把预算集中到采样出的候选动作上。

## 政策更新

论文不只用最终选中的动作训练 policy，还提出 completed Q-values：

- 已访问动作使用搜索得到的 Q。
- 未访问动作用 `v_pi` 近似填充。
- 构造 improved policy：`softmax(logits + sigma(completedQ))`。
- 用 KL 把网络 policy 蒸馏到 improved policy。

这比只模仿一个 best action 提供更多训练信号。

## 实验结论

在 9x9 Go 中，普通 MuZero 在 16 次或更少 simulations 下学习困难，而 Gumbel MuZero 即使 2 次 simulations 也能可靠学习。大规模 19x19 Go 和 chess 中，Gumbel 版本不弱于原方法。在 Atari 的 Ms. Pacman 中，Gumbel MuZero 在极低 simulation 数下也明显更稳。

论文强调它最适合搜索预算小于动作数或刚好接近动作数的场景。

## 对魔塔的启发

魔塔宏动作虽然有 action mask，但关键节点仍然可能有十几个候选目标：

- 开哪扇门。
- 杀哪个怪。
- 先拿哪组宝石。
- 是否回低层拿资源。
- 是否先上楼。

当前 best-first/beam 搜索如果 heuristic 排序错，关键动作可能很晚才被扩展。Gumbel MuZero 的启发是：候选动作选择不应只取 top heuristic，也应保留一定无放回探索，同时最终选择仍结合价值评估。

短期可以做一个非神经版本：

1. 对所有 legal macro actions 算 prior score。
2. 用 Gumbel-Top-k 采样候选动作。
3. 对候选动作做小深度 rollout 或静态 evaluator。
4. 用 `prior + value_delta` 选择扩展顺序。

这会比固定 top-k 更不容易漏掉低 prior 但高价值的路线。

## 工程落地

建议新增一个 solver heuristic mode：

```text
--action-selection gumbel_topk
--gumbel-samples 8
--rollout-depth 2
```

每次 expansion 不是按 action_bias 固定排序，而是：

```text
score = log_prior(action) + gumbel_noise + scaled_static_value(next_state)
```

评估时固定 seed，保证可复现。

## 局限

Gumbel MuZero 仍然依赖 policy/value 质量。魔塔当前没有训练好的 policy/value，所以第一版只能用手写 heuristic 近似 logits 和 Q。它不能替代 staged solver，只能改善候选动作探索。

## 给本项目的下一步

1. 在 macro action ranking 中加入 Gumbel-Top-k 可选模式。
2. 记录每个动作的 prior、gumbel、value、最终是否进入 best route。
3. 后续用 search trace 训练 prior policy。
