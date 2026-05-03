# 《魔塔》前十层 RL+规划求解阅读报告

## 结论先行

《魔塔》前十层不适合作为第一阶段的纯 model-free RL 任务。可实施路线是：确定性模拟器作为真值，搜索专家产出轨迹，行为克隆初始化 masked macro policy，再用 Maskable PPO/DQN+PER 微调。MuZero/UniZero/Thinker 是后续研究路线，不应阻塞前十层验收。

## 逐篇摘记

1. AlphaZero：问题是无外部数据的完美信息博弈；方法是神经网络价值/策略加 MCTS；对魔塔启示是必须把深搜预算集中在门、钥匙、商店、Boss 等分叉点；工程取舍是先保留显式模拟器，不直接学习规则。
2. AlphaGo Zero：问题是从自我对弈中发现长程策略；方法是残差网络加搜索改进；启示是专家数据不是必要条件但能显著节省冷启动；本项目用搜索轨迹替代人类谱。
3. DQN：问题是从像素学习 Atari 控制；方法是 Q-learning、目标网络、回放；启示是可做 baseline，但魔塔稀疏奖励会失败；只保留 DQN+mask/PER 作为对照。
4. Nature DQN：问题是通用视觉控制；方法是端到端卷积 Q 网络；启示是通用性不等于适配长视野规划；前十层不走像素输入。
5. Double DQN：问题是 Q 值过估计；方法是动作选择与评估解耦；启示是离散宏动作 RL 需要控制过估计；DQN baseline 应默认启用。
6. Dueling DQN：问题是许多动作价值差异小；方法是 V/A 分解；启示是移动类宏动作很多，可分离状态价值与动作优势；适合作为轻量 baseline。
7. PER：问题是关键经验被普通回放稀释；方法是按 TD error 采样；启示是 Boss、商店、装备、钥匙门转换必须高频回放；本项目路线 JSONL 可直接转 PER 种子。
8. NoisyNet：问题是探索不足；方法是参数噪声；启示是比 epsilon 更适合 mask 后动作；但前十层仍以专家轨迹起步。
9. C51：问题是价值分布而非期望；方法是 categorical distributional RL；启示是死局风险可通过分布尾部表达；工程上暂不优先。
10. Rainbow：问题是单项 DQN 改进不足；方法是组合 Double、Dueling、PER、Noisy、C51、多步；启示是本项目 DQN baseline 应组合化；优先级低于搜索专家。
11. BTR：问题是桌面算力下高性能 Atari；方法是 Rainbow 后续技巧工程化；启示是本地/Pod 都可先跑轻量 baseline；用于资源受限训练。
12. MuZero：问题是无规则模型也能规划；方法是 representation/dynamics/prediction 三网络加 MCTS；启示是魔塔后续可学习隐式伤害与资源模型；前十层已有模拟器，不先用 MuZero。
13. Sampled MuZero：问题是大动作空间；方法是采样动作子集规划；启示是魔塔宏动作可只采样可达交互目标；当前环境已实现 action mask。
14. Gumbel MuZero：问题是搜索改进效率；方法是 Gumbel top-k 与策略改进；启示是单人长序列任务可用更少模拟；后续可接 LightZero。
15. EfficientZero：问题是样本效率；方法是自监督一致性、价值前缀和探索；启示是数据少时适合；但实现成本高于当前阶段收益。
16. UniZero：问题是长历史与潜状态纠缠；方法是 Transformer 隐世界模型解耦历史；启示是跨楼层记忆适合魔塔；作为第二阶段研究主线。
17. Thinker：问题是何时规划；方法是让 agent 学会调用内部世界模型；启示是走廊可少算、资源分叉多算；工程上可映射为动态 beam/MCTS budget。
18. MCTSnet：问题是把树搜索结构神经化；方法是可微搜索模块；启示是可学习启发式排序；当前用手写启发式，后续用轨迹训练排序器。
19. World Models：问题是学习环境压缩模型；方法是 VAE+RNN+controller；启示是可把地图和资源压到 latent；但魔塔确定性强，显式模型更稳。
20. PlaNet：问题是像素规划；方法是 latent dynamics 和 CEM planning；启示是可做 primitive action wrapper；当前不走像素。
21. Dreamer：问题是 latent imagination 训练 actor；方法是在模型想象轨迹上学习；启示是适合远期扩展到随机塔；前十层先不需要。
22. DreamerV2：问题是离散潜变量提升 Atari；方法是 discrete latent world model；启示是魔塔实体本就是离散 token，可借鉴离散 latent。
23. DreamerV3：问题是跨域稳健世界模型；方法是归一化与统一超参；启示是若做多塔泛化可参考；单塔目标暂缓。
24. Plan2Explore：问题是无奖励探索；方法是 disagreement intrinsic reward；启示是可奖励开新区域、触发新事件；当前 reward shaping 已保留接口。
25. RND：问题是新颖性探索；方法是随机网络预测误差；启示是可推动 agent 找未知楼层；但会误奖无意义绕路，需 action mask。
26. ICM：问题是内在好奇；方法是预测动作导致的特征变化；启示是适合开门/拾取/战斗状态变化；不适合作为主线。
27. Go-Explore：问题是 hard exploration；方法是回到有前途状态再探索；启示是魔塔可缓存“高资源里程碑状态”；专家搜索 dominance pruning 属于同类思想。
28. NGU：问题是长期探索；方法是 episodic novelty 与 lifelong novelty；启示是多塔训练有价值；单塔可简化。
29. Options：问题是时间抽象；方法是 option policy 与 termination；启示是“到某楼梯/取某装备/买属性”天然是 option；当前宏动作即第一版 options。
30. MAXQ：问题是层次价值分解；方法是任务树 decomposition；启示是最终可拆成楼层推进、资源获取、Boss 击杀；需要避免人工规则过强。
31. FeUdal Networks：问题是高层目标引导低层动作；方法是 manager-worker；启示是高层输出资源/坐标目标，低层执行寻路。
32. Option-Critic：问题是自动学习 options；方法是端到端 option policy；启示是可在专家轨迹上发现常用技能；当前数据量不足。
33. HIRO：问题是高层低频连续目标；方法是 off-policy correction；启示是对魔塔可输出目标状态向量；适合后续跨塔。
34. HAC：问题是稀疏奖励层次学习；方法是 hindsight action transitions；启示是“未达 Boss 但达成装备状态”也能学习；可配合 HER。
35. HER：问题是目标未达成轨迹浪费；方法是把实际终点重标为目标；启示是失败路线可重标为“到达 MT8/拿铁盾”；适合利用当前 partial route。
36. HalfWeg：问题是 Sokoban 长规划；方法是递归中间地标子目标；启示是魔塔可自动发现“铁剑+铁盾+关键钥匙”中继状态；工程上建议第二阶段实现。
37. SEADS：问题是技能与符号效果连接；方法是学习多样技能和符号 forward model；启示是魔塔宏动作效果可符号化；适合把 RL 与规划器融合。
38. Sokoban complexity：问题是运动规划复杂性；结论是 Sokoban 类问题 PSPACE/NP-hard；启示是魔塔路线死局不是实现问题，而是问题本性；必须剪枝和抽象。
39. NLE：问题是复杂 roguelike 环境；方法是高性能 Gym 封装和符号观测；启示是魔塔也应提供符号网格、实体、标量资源，而非纯图像。
40. MiniHack：问题是可控 NetHack 子任务；方法是可生成小任务基准；启示是魔塔训练应先切成 1-3F、1-5F、1-10F curriculum。
41. NLD：问题是大规模 NetHack 轨迹数据；方法是压缩人类/bot trajectories；启示是专家路线 JSONL 是必要资产；后续可积累多条路线。
42. SC2LE：问题是复杂部分可观测策略游戏；方法是标准环境和多粒度动作；启示是动作抽象与 observation schema 比算法更早决定成败。
43. Boxoban：问题是可程序生成 Sokoban；方法是大量关卡集；启示是魔塔若要泛化，需要塔生成器；当前只做固定 50 层前十。
44. Model-Free Planning in Boxoban：问题是无模型网络是否规划；方法是 DRC 反复计算；启示是可在策略网络中加入 internal ticks；当前先用外部搜索。
45. Searchformer：问题是 Transformer 学搜索；方法是学习 A* search dynamics 而非只学答案；启示是可以把专家搜索 trace 训练成路线生成器。
46. Path Channels：问题是 RNN 内部规划机制；方法是逆向 DRC hidden channels；启示是魔塔网络应保留循环计算深度和可解释通道。
47. Decision Transformer：问题是把 RL 视为序列建模；方法是 return-conditioned Transformer；启示是可用专家/失败轨迹做离线策略；对在线探索帮助有限。
48. Trajectory Transformer：问题是轨迹建模与规划；方法是 beam search over model tokens；启示是魔塔宏动作序列可 token 化；需要高质量轨迹。
49. Language Models as Zero-Shot Planners：问题是 LLM 生成计划；方法是提示语言模型输出动作；启示是可让 LLM 产高层路线，但必须由模拟器验证。
50. RAP：问题是语言模型推理中的搜索；方法是把 reasoning/action planning 结合；启示是 LLM 可做高层候选，搜索器做验算。
51. LLM-Planner：问题是少样本 grounded planning；方法是 LLM 生成可执行计划再映射环境动作；启示是适合生成魔塔里程碑，不适合替代数值模拟。
52. OpenGame：问题是 LLM 生成游戏；方法是执行反馈调试；启示是解析复杂 JS 游戏工程时可用执行闭环；本项目 Node extractor 已采用执行式数据抽取。
53. LightZero：问题是统一 MCTS/RL 框架；方法是工程化 MuZero/Gumbel/EfficientZero；启示是后续接入比从零实现更现实。
54. mota-js：问题是 HTML5 魔塔工程；方法是标准化地图/事件/实体 JS 数据；启示是本项目直接执行 JS 导出 JSON，避免手写解析。
55. magic-tower-ai：问题是魔塔路线 AI；方法是启发式/图搜索工程；启示是可作为传统 solver 对照；当前还未完整移植。
56. dasstudio2016：问题是复杂魔塔测试与 AI；方法是收集高难测试例；启示是第一阶段不要直接追求泛化，先稳定前十层机制。

## 工程建议

环境层应继续以 HTML5 工程导出的 JSON 为真值，补齐事件解释器和 replay 校验；状态空间用楼层、13x13 实体网格、主角属性、钥匙/flag 组成。算法层应优先完善搜索专家，确保 `route_first10.jsonl` 真正击败 10F 骷髅队长；随后用行为克隆初始化 masked macro policy，再接 Maskable PPO 或 DQN+PER。远期再接 LightZero/UniZero/Thinker。

## 当前风险

最大风险不是 RL 依赖，而是 50 层原版事件语义与 HTML5 复刻工程之间的偏差。第二个风险是专家搜索仍会在 7-9F 钥匙与商店分支上耗散，需要把“8F 蓝钥匙、9F 铁盾、10F 陷阱后小怪清理”提升为显式里程碑。第三个风险是如果只用 partial route 行为克隆，policy 会学到上楼但学不到 Boss 完成条件。

## 里程碑

M1：模拟器单测稳定，能 replay partial/expert route。M2：搜索专家固定输出击败骷髅队长路线。M3：行为克隆在 route replay 上 100% 复现。M4：Maskable PPO/DQN+PER 在固定种子 100 局达到 95% 以上。M5：用 Ruffle/HTML5 UI 回放关键段确认事件一致。
