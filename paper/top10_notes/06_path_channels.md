# 06 - Path Channels and Plan Extension Kernels

## 基本信息

- 论文：Path Channels and Plan Extension Kernels: A Mechanistic Description of Planning in a Sokoban RNN
- 作者：Mohammad Taufeeque, Aaron David Tucker, Adam Gleave, Adrià Garriga-Alonso
- 年份：2026
- 链接：https://openreview.net/forum?id=aAshH4kQ1v
- 本地 PDF：`paper/top10_pdfs/06_path_channels_aAshH4kQ1v.pdf`
- 抽取文本：`paper/top10_extracted/06_path_channels_aAshH4kQ1v.md`

## 一句话结论

一个 model-free Sokoban RNN 真的学出了类似搜索的内部机制：隐藏通道表示未来路径，卷积核扩展路径，负激活负责回溯和剪枝。

## 论文解决的问题

很多人怀疑 model-free 网络只是模式匹配，不是真的规划。该论文对 Sokoban 上的 DRC(3,3) 网络做 mechanistic interpretability，试图回答：网络内部是否存在可解释的计划表示和搜索算法。

这对魔塔有两个意义：

1. 纯网络并非一定不能学规划，但结构要允许多步内部计算。
2. 如果我们以后训练策略网络，应该设计可解释、可迭代的内部状态，而不是普通 MLP。

## 网络对象

论文分析的是 Deep Repeating ConvLSTM，简称 DRC(3,3)：

- 输入是 Sokoban 图像。
- 经过卷积 encoder。
- 3 层 ConvLSTM。
- 每个环境 step 内部重复 3 个 tick。
- 最后 MLP 输出 policy 和 value。

这个结构在每次真实动作前有多次内部计算，因此能在隐藏状态里延展计划。

## 核心发现

论文把许多隐藏通道解释为 path channels：

- box-movement channels：表示箱子未来会从某格往哪个方向移动。
- agent-movement channels：表示 agent 未来会从某格往哪个方向移动。
- combined path channels：组合多方向路径。
- GNA/PNA channels：把完整路径转换为下一步动作。

重要实验：

- 只干预 path channels，解题率大幅下降。
- 干预 non-path channels，下降小得多。
- 用通道激活预测未来动作，有较高 AUC。
- 直接修改通道激活，可以因果性地改变下一步动作。

## 内部规划机制

论文进一步分析卷积核，发现三类机制：

1. 初始化路径：encoder 在箱子、目标、agent 附近激活路径通道。
2. 扩展路径：plan extension kernels 沿方向传播激活，既能从箱子向前扩展，也能从目标反向扩展。
3. 剪枝与回溯：障碍物或不可行位置产生负激活，负激活沿路径反向传播，把坏路径剪掉。

此外，网络有 winner-takes-all 机制，在多个候选路径冲突时保留较强路径。论文还发现长短期路径通道分工：短期通道负责接下来约 0-10 步，长期通道负责更远计划。

## 对魔塔的启发

魔塔中“障碍物”不是固定墙，而是动态障碍：

- 黄门：有黄钥匙才可通过。
- 蓝门：有蓝钥匙才可通过。
- 怪物：能破防且 HP 足够才可通过。
- Boss：攻防阈值和 HP margin 足够才可通过。

因此魔塔的网络如果要学内部规划，路径通道必须把资源条件编码进去。一个格子是否可通行不是只看地图，而是看当前 `ATK/DEF/HP/keys/items/flags`。

这说明普通 CNN 很可能不够。更合理的结构是：

- 网格 entity embedding。
- 全局资源向量通过 FiLM/attention 注入每个格子。
- ConvLSTM/Transformer 多 tick 反复更新可达性。
- 障碍通道显式表示“钥匙不足”“打不过”“战损过高”。

## 工程落地

短期搜索器可以借鉴这套机制，显式计算类似通道：

- `reachable_now`
- `blocked_by_yellow_key`
- `blocked_by_blue_key`
- `blocked_by_damage`
- `valuable_after_unlock`
- `boss_damage_gradient`

这些通道可以成为 `Phi(s)` 和 action ranking 的输入。后续训练网络时，也能用它们做辅助监督，让网络更快学到规划结构。

## 局限

论文只分析 Sokoban DRC，不保证 Transformer 或魔塔网络会学出同样结构。它解释的是固定棋盘和固定规则下的路径规划；魔塔多了数值资源和事件脚本，所以必须加入符号特征。

## 给本项目的下一步

1. 在模拟器中导出可达性/阻塞原因 grid。
2. 记录每个 candidate action 的阻塞类型。
3. 后续策略网络采用 recurrent ticks，不用单步 MLP。
4. 训练后做简单 probe：网络 hidden state 是否编码 boss damage、key pressure、reachable resource。
