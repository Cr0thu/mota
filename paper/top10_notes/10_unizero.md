# 10 - UniZero

## 基本信息

- 论文：UniZero: Generalized and Efficient Planning with Scalable Latent World Models
- 作者：Yuan Pu et al.
- 年份：2024
- 链接：https://arxiv.org/abs/2406.10667
- 本地 PDF：`paper/top10_pdfs/10_unizero_2406.10667.pdf`
- 抽取文本：`paper/top10_extracted/10_unizero_2406.10667.md`

## 一句话结论

UniZero 用 Transformer latent world model 替代 MuZero 式递归 latent dynamics，显式利用历史上下文，解决长记忆和多任务场景中 latent state 与历史信息纠缠的问题。

## 论文解决的问题

MuZero-style 架构有两个限制：

1. 训练时只从序列初始 observation 开始 unroll，轨迹数据利用不足。
2. 递归 latent state 同时承担“当前状态”和“历史记忆”，导致 latent representation 与历史纠缠。

在需要长历史的 POMDP 中，这会很严重。魔塔 50 层也有类似问题：低层未开的门、高层拿到钥匙后回访、早期攻防选择影响后期 Boss，这些都需要长期记忆。

## 核心方法

UniZero 采用模块化 latent world model：

1. domain-specific encoder：把不同模态 observation/action 编码到统一 latent。
2. Transformer backbone：处理一段 latent state/action history。
3. dynamics head：预测下一 latent state 和 reward。
4. decision head：预测 policy 和 value。

关键思想是把 latent state 和 implicit latent history 解耦。Transformer 负责历史上下文，当前 latent state 不必独自承载所有历史。

训练时，UniZero 使用整段序列，而不是只从第一个 observation 开始递归展开。推理时，MCTS 在 learned latent space 里做，root node 使用当前 observation 编码和 KV cache 中的历史信息。

## MCTS 方式

UniZero 的 MCTS 仍包括 selection、expansion、backup：

- selection 用 PUCT。
- expansion 时，dynamics/decision head 根据当前 latent 和 KV memory 预测 next latent、reward、policy、value。
- backup 把预测 reward 和 target value 回传。
- root visit counts 归一化成 improved policy。

它保留 MuZero 的 MCTS policy improvement，但把世界模型换成可扩展的 Transformer latent model。

## 实验结论

论文测试了：

1. VisualMatch：需要记住早期看到的颜色，经过长 distraction 后选择匹配目标。MuZero 因历史不足表现差，SAC-GPT 随 memory length 增加而下降，UniZero 维持高成功率。
2. Atari multitask：一个模型训练多个 Atari，UniZero MT 优于 MuZero MT，并且显示模型规模增加能提高样本效率。
3. Atari 100K single-task：UniZero 高于复现 MuZero。
4. DMControl：UniZero 在多项连续控制任务上与 DreamerV3 竞争或超过。

Ablation 显示 SimNorm 对训练稳定性很重要；单纯 decode reconstruction regularization 影响较小，说明决策相关 latent 比像素重建更重要。

## 对魔塔的启发

如果只做固定前十层，UniZero 不是第一优先级。但如果目标扩展到：

- 50 层全流程。
- 多个魔塔版本。
- HTML5/SWF 事件差异。
- 从游戏文件自动解析新塔。
- 跨楼层回访和长期资源规划。

UniZero 比标准 MuZero 更合适。

魔塔中的历史依赖包括：

- 某层某门是否开过。
- 某怪是否杀过。
- 某个 NPC/事件 flag 是否触发。
- 已消耗钥匙和剩余钥匙期权。
- 低层可回收资源是否仍存在。

这些很难压进一个短 latent state；Transformer history 更自然。

## 工程落地

当前项目可以提前按 UniZero 需要设计数据格式：

```text
sequence = [
  (obs_t, macro_action_t, reward_t, flags_t, phi_components_t),
  ...
]
```

观察建议包括：

- floor id。
- entity grid。
- hero scalar。
- keys/items/flags。
- action mask。
- search improved policy。
- value target：最终成功、Boss margin、stage progress。

后续可用 LightZero 接 UniZero/MuZero 系列，而不是从零实现。

## 局限

UniZero 仍然是高成本研究路线。它需要大量轨迹、GPU、稳定实现和 MCTS 训练闭环。魔塔当前最缺的是成功专家路线和高质量 simulator trace，不是模型规模。

## 给本项目的下一步

1. 先把前十层 solver 做成稳定专家。
2. 保存序列数据时按 UniZero/MuZero 兼容格式组织。
3. 当路线数据足够后，用 LightZero 做小规模实验。
4. 50 层阶段再考虑 UniZero 作为主世界模型。
