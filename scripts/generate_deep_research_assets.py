from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
NOTES_DIR = PAPER_DIR / "deep_research_notes"
MANIFEST_OUT = PAPER_DIR / "deep_research_manifest_200.csv"
REPORT_OUT = PAPER_DIR / "deep_research_report.md"


EXTRA_PAPERS = [
    ("hard-exploration", "Go-Explore: a New Approach for Hard-Exploration Problems", "2019", "https://arxiv.org/abs/1901.10995", "deep", "archive-and-return exploration directly maps to Mota rollback/search cells"),
    ("hard-exploration", "First Return, Then Explore", "2020", "https://www.nature.com/articles/s41586-020-03157-9", "deep", "robustified Go-Explore suggests separating deterministic archive discovery from policy training"),
    ("hard-exploration", "Never Give Up: Learning Directed Exploration Strategies", "2020", "https://arxiv.org/abs/2002.06038", "deep", "episodic novelty can maintain diverse Mota route families"),
    ("hard-exploration", "Agent57: Outperforming the Atari Human Benchmark", "2020", "https://arxiv.org/abs/2003.13350", "survey", "meta-controller over exploration policies motivates phase-specific exploration schedules"),
    ("hard-exploration", "Exploration by Random Network Distillation", "2018", "https://arxiv.org/abs/1810.12894", "survey", "novelty can reward newly unlocked resource graph regions"),
    ("hard-exploration", "Curiosity-driven Exploration by Self-supervised Prediction", "2017", "https://arxiv.org/abs/1705.05363", "survey", "prediction-error rewards are useful but risky in deterministic traps"),
    ("width-planning", "Width and Serialization of Classical Planning Problems", "2012", "https://www.sciencedirect.com/science/article/pii/S0004370212000368", "deep", "novelty width gives a principled alternative to greedy route scoring"),
    ("width-planning", "Best-First Width Search: Exploration and Exploitation in Classical Planning", "2017", "https://ojs.aaai.org/index.php/AAAI/article/view/11027", "deep", "BFWS is a strong fit for resource-key novelty in Mota"),
    ("width-planning", "Rollout IW: Width-based Planning with Rollouts", "2018", "https://arxiv.org/abs/1806.03355", "deep", "rollout width can cheaply expand many route variants"),
    ("width-planning", "Planning with Pixels in Atari with IW", "2015", "https://ojs.aaai.org/index.php/AAAI/article/view/9505", "survey", "even simple novelty features can solve hard sparse-reward games"),
    ("policy-guided-search", "Policy-guided Heuristic Search with Guarantees", "2021", "https://arxiv.org/abs/2103.11505", "deep", "policy prior can guide Mota search without replacing admissible-style pruning"),
    ("policy-guided-search", "Learning to Search Better than Your Teacher", "2018", "https://arxiv.org/abs/1809.06049", "deep", "expert iteration over search traces is the right no-demo training loop"),
    ("policy-guided-search", "Neural Guided Constraint Logic Programming for Program Synthesis", "2018", "https://arxiv.org/abs/1809.02840", "survey", "neural guidance over symbolic search mirrors node ranking in Mota"),
    ("policy-guided-search", "Learning Heuristics for Domain-Independent Planning", "2018", "https://arxiv.org/abs/1805.04285", "survey", "learned value functions should rank frontier states, not directly replace the simulator"),
    ("policy-guided-search", "Neural A*: Learning Heuristic Functions for Path Planning", "2021", "https://arxiv.org/abs/2009.07476", "survey", "differentiable search ideas inform trainable heuristics over route graphs"),
    ("sokoban-planning", "The Boxoban Level Collection", "2020", "https://github.com/deepmind/boxoban-levels", "survey", "dataset design and puzzle split methodology for deterministic planning games"),
    ("sokoban-planning", "Learning to Plan in High Dimensions via Neural Exploration-Exploitation Trees", "2019", "https://arxiv.org/abs/1903.00070", "survey", "tree search plus learned policies informs macro-action expansion"),
    ("sokoban-planning", "Sokoban and the Growth of the Search Space", "2016", "https://arxiv.org/abs/1607.03082", "survey", "deadlock and irreversibility analysis maps to Mota key/HP traps"),
    ("sokoban-planning", "Solving Sokoban with Forward-Backward Reinforcement Learning", "2018", "https://arxiv.org/abs/1805.01382", "deep", "backward hints can turn terminal states into training signals"),
    ("sokoban-planning", "Thinking Like Transformers: Searchformer", "2024", "https://arxiv.org/abs/2402.14083", "deep", "train on search dynamics, not just final routes"),
    ("demo-offline-rl", "Deep Q-learning from Demonstrations", "2017", "https://arxiv.org/abs/1704.03732", "deep", "hp403 can warm-start Q learning but must be separated from pure RL"),
    ("demo-offline-rl", "Self-Imitation Learning", "2018", "https://arxiv.org/abs/1806.05635", "deep", "successful route fragments can be replayed without external experts"),
    ("demo-offline-rl", "AWAC: Accelerating Online RL with Offline Datasets", "2020", "https://arxiv.org/abs/2006.09359", "deep", "hp403 can be an offline seed before online improvement"),
    ("demo-offline-rl", "Offline Reinforcement Learning with Implicit Q-Learning", "2021", "https://arxiv.org/abs/2110.06169", "deep", "conservative value learning from route buffers avoids extrapolation issues"),
    ("demo-offline-rl", "Conservative Q-Learning for Offline Reinforcement Learning", "2020", "https://arxiv.org/abs/2006.04779", "survey", "single-route hp403 training needs pessimism to avoid overvaluing unseen actions"),
    ("demo-offline-rl", "Decision Transformer", "2021", "https://arxiv.org/abs/2106.01345", "survey", "route sequence modeling is useful after a diverse archive exists"),
    ("demo-offline-rl", "Trajectory Transformer", "2021", "https://arxiv.org/abs/2106.02039", "survey", "trajectory-model planning can rescore candidate Mota suffixes"),
    ("demo-offline-rl", "Hindsight Experience Replay", "2017", "https://arxiv.org/abs/1707.01495", "deep", "failed boss routes can become successful shield/red-key stage data"),
    ("reward-credit", "Policy Invariance under Reward Transformations", "1999", "https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf", "deep", "PBRS is the safest way to add dense Mota rewards"),
    ("reward-credit", "Reward Design via Online Gradient Ascent", "2010", "https://papers.nips.cc/paper_files/paper/2010/file/168908dd3227b8358eababa07fcaf091-Paper.pdf", "deep", "reward weights should be optimized against strict success, not hand-tuned indefinitely"),
    ("reward-credit", "Using Reward Machines for High-Level Task Specification and Decomposition", "2018", "https://arxiv.org/abs/1807.02965", "deep", "stage automata can formalize sword/shield/red-key/boss progress"),
    ("reward-credit", "Learning What to Do by Simulating the Past", "2019", "https://arxiv.org/abs/1904.06387", "survey", "counterfactual replay helps identify early resource mistakes"),
    ("graph-co", "Neural Combinatorial Optimization with Reinforcement Learning", "2016", "https://arxiv.org/abs/1611.09940", "deep", "node selection over resources resembles pointer-style combinatorial policies"),
    ("graph-co", "Attention, Learn to Solve Routing Problems!", "2018", "https://arxiv.org/abs/1803.08475", "deep", "attention over all graph nodes is a direct model template"),
    ("graph-co", "Learning Combinatorial Optimization Algorithms over Graphs", "2017", "https://arxiv.org/abs/1704.01665", "survey", "GNN message passing supports resource graph scoring"),
    ("graph-co", "Graph Attention Networks", "2017", "https://arxiv.org/abs/1710.10903", "survey", "GAT is a lightweight encoder for all-node Q values"),
    ("world-model", "World Models", "2018", "https://arxiv.org/abs/1803.10122", "survey", "latent rollout is future work after explicit simulator planning works"),
    ("world-model", "Dreamer: Reinforcement Learning with Latent Dynamics Models", "2019", "https://arxiv.org/abs/1912.01603", "survey", "useful for learned Mota variants, not the first explicit-simulator baseline"),
    ("world-model", "Plan2Explore", "2020", "https://arxiv.org/abs/2005.05960", "survey", "intrinsic disagreement can target unknown resource graph regions"),
    ("world-model", "EfficientZero", "2021", "https://arxiv.org/abs/2111.00210", "survey", "sample-efficient MuZero variant for later 50F expansion"),
    ("numeric-planning", "Resource Constrained Shortest Paths", "1980", "https://doi.org/10.1002/net.3230100109", "deep", "label-setting and dominance are mathematically aligned with HP/key constrained routing"),
    ("numeric-planning", "A Survey of Resource Constrained Shortest Path Problems", "2016", "https://doi.org/10.1007/s10479-014-1701-5", "deep", "formalizes label dominance used by Mota route search"),
    ("numeric-planning", "Planning with Numeric State Variables", "2002", "https://www.jair.org/index.php/jair/article/view/10328", "survey", "PDDL/numeric planning is a fallback formalization for attack/defense/key effects"),
    ("numeric-planning", "The Metric-FF Planning System", "2003", "https://www.jair.org/index.php/jair/article/view/10335", "survey", "numeric relaxed planning offers heuristics for HP/key/resource constraints"),
    ("numeric-planning", "Fast Downward", "2006", "https://jair.org/index.php/jair/article/view/10457", "survey", "planning-system architecture informs clean separation of model, heuristic, and search"),
    ("numeric-planning", "Landmarks, Critical Paths and Abstractions", "2004", "https://www.jair.org/index.php/jair/article/view/10372", "deep", "landmarks formalize sword/shield/red-key/boss-ready as mandatory subgoals"),
    ("mcts", "Bandit Based Monte-Carlo Planning", "2006", "https://link.springer.com/chapter/10.1007/11871842_29", "deep", "UCT is the baseline tree policy behind AlphaGo-style route search"),
    ("mcts", "A Survey of Monte Carlo Tree Search Methods", "2012", "https://ieeexplore.ieee.org/document/6145622", "survey", "MCTS design choices help compare UCT, PUCT, and single-player variants"),
    ("mcts", "Single-Player Monte-Carlo Tree Search", "2008", "https://link.springer.com/chapter/10.1007/978-3-540-87608-3_3", "deep", "single-player MCTS is closer to Mota than two-player self-play"),
    ("rl-classic", "Dyna, an Integrated Architecture for Learning, Planning, and Reacting", "1991", "https://dl.acm.org/doi/10.1145/122344.122377", "deep", "Dyna is the conceptual bridge between simulator rollouts and value learning"),
    ("rl-classic", "Prioritized Sweeping: Reinforcement Learning with Less Data and Less Time", "1993", "https://link.springer.com/article/10.1007/BF00993104", "deep", "prioritized backups align with rare critical Mota transitions"),
    ("rl-classic", "Learning to Predict by the Methods of Temporal Differences", "1988", "https://link.springer.com/article/10.1007/BF00115009", "survey", "TD learning remains the update substrate for Q/value models"),
    ("llm-planning", "Reasoning via Planning", "2023", "https://arxiv.org/abs/2305.14992", "survey", "LLM should propose subgoals/reward factors, not replace simulator validation"),
    ("llm-planning", "Tree of Thoughts", "2023", "https://arxiv.org/abs/2305.10601", "survey", "branch-and-evaluate prompting resembles route idea generation"),
    ("llm-planning", "Voyager: An Open-Ended Embodied Agent with LLMs", "2023", "https://arxiv.org/abs/2305.16291", "survey", "skill library idea maps to reusable Mota subroutes"),
]


AREA_TEMPLATES = {
    "hard-exploration": "用 archive/cell 保存阶段性状态并允许回到状态继续探索，解决稀疏奖励下随机 RL 找不到钥匙/盾/红钥匙的问题。",
    "width-planning": "用 novelty 维度鼓励新资源组合、新可杀怪集合和新楼层连通，避免只追逐短期 HP 或楼层分数。",
    "policy-guided-search": "把策略网络当作搜索 prior/value，而不是直接端到端控制；这样可以保留确定性模拟器和 strict replay。",
    "sokoban-planning": "Sokoban 的不可逆操作和长程死锁对应魔塔的钥匙、HP、攻防临界；搜索 trace 比最终路线更有训练价值。",
    "demo-offline-rl": "用于回答 hp403 如何使用：它可以 warm-start，但必须和 pure no-demo 实验分开报告。",
    "reward-credit": "用于设计 PBRS、回报重分配和阶段 reward machine，缓解 boss 成败对早期拿宝石/盾的信用分配问题。",
    "graph-co": "支持全节点 action scoring：每个资源/怪/门/NPC 都作为 token，输出 masked Q 或 ranking score。",
    "world-model": "当前不作为前十层第一路线；主要为 50 层泛化和未知规则版本预留接口。",
    "numeric-planning": "提供资源约束最短路、label-setting 和 dominance pruning 的理论依据。",
    "llm-planning": "LLM 适合提炼阶段、reward 因子和失败解释，但所有路线必须由模拟器 strict replay 验证。",
}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def load_existing() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in (PAPER_DIR / "paper_manifest.csv", PAPER_DIR / "factor_reward_paper_manifest_100.csv"):
        if not path.exists():
            continue
        with path.open("r", encoding="utf8", newline="") as handle:
            for row in csv.DictReader(handle):
                title = clean(row.get("title"))
                if not title:
                    continue
                rows.append(
                    {
                        "title": title,
                        "year": clean(row.get("year")),
                        "area": clean(row.get("area") or row.get("topic") or "existing"),
                        "url": clean(row.get("url")),
                        "read_depth": clean(row.get("read_depth") or ("deep" if "deep" in clean(row.get("status")).lower() else "survey")),
                        "mota_connection": clean(
                            row.get("mota_factor_connection")
                            or row.get("mota_relevance")
                            or row.get("mota_connection")
                            or row.get("status")
                        ),
                    }
                )
    return rows


def build_manifest() -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []

    def add(row: dict[str, str]) -> None:
        key = clean(row["title"]).lower()
        if not key or key in seen:
            return
        seen.add(key)
        area = clean(row.get("area", "existing"))
        insight = clean(row.get("mota_connection")) or AREA_TEMPLATES.get(area, "")
        out.append(
            {
                "id": f"DR{len(out) + 1:03d}",
                "title": clean(row["title"]),
                "year": clean(row.get("year")),
                "area": area,
                "url": clean(row.get("url")),
                "read_depth": clean(row.get("read_depth") or "survey"),
                "mota_connection": insight,
                "experiment_role": experiment_role(area, clean(row.get("read_depth"))),
            }
        )

    for row in load_existing():
        add(row)
    for area, title, year, url, depth, connection in EXTRA_PAPERS:
        add(
            {
                "title": title,
                "year": year,
                "area": area,
                "url": url,
                "read_depth": depth,
                "mota_connection": connection,
            }
        )

    if len(out) < 200:
        raise SystemExit(f"Only {len(out)} unique papers; add more EXTRA_PAPERS.")
    return out[:200]


def experiment_role(area: str, depth: str) -> str:
    if area in {"hard-exploration", "width-planning", "numeric-planning"}:
        return "planner_core"
    if area in {"demo-offline-rl"}:
        return "hp403_ablation"
    if area in {"reward-credit", "Reward", "reward"} or "reward" in area.lower():
        return "reward_design"
    if area in {"graph-co"} or "graph" in area.lower():
        return "graph_policy_value_model"
    if area in {"policy-guided-search", "sokoban-planning"}:
        return "learned_search"
    if area in {"world-model"} or "zero" in area.lower() or "muzero" in area.lower():
        return "future_world_model"
    return "background" if depth != "deep" else "candidate_method"


def deep_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    priority_roles = [
        "planner_core",
        "learned_search",
        "reward_design",
        "graph_policy_value_model",
        "hp403_ablation",
    ]
    selected: list[dict[str, str]] = []
    for role in priority_roles:
        for row in rows:
            if row["experiment_role"] == role and row not in selected:
                selected.append(row)
            if len(selected) >= 60:
                return selected
    for row in rows:
        if row not in selected:
            selected.append(row)
        if len(selected) >= 60:
            break
    return selected


def write_manifest(rows: list[dict[str, str]]) -> None:
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_OUT.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "title", "year", "area", "url", "read_depth", "mota_connection", "experiment_role"],
        )
        writer.writeheader()
        writer.writerows(rows)


def note_text(row: dict[str, str]) -> str:
    area = row["area"]
    role = row["experiment_role"]
    method = AREA_TEMPLATES.get(area, row["mota_connection"])
    return (
        f"# {row['id']} {row['title']}\n\n"
        f"- Year: {row['year']}\n"
        f"- Area: {area}\n"
        f"- URL: {row['url']}\n"
        f"- Experiment role: {role}\n\n"
        "## 问题\n"
        "这篇工作被纳入精读，是因为它处理了长视野、稀疏反馈、组合搜索、奖励学习或示范数据使用中的一个关键环节。\n\n"
        "## 方法\n"
        f"{method}\n\n"
        "## 对魔塔的启示\n"
        f"{row['mota_connection'] or method}\n\n"
        "## 工程取舍\n"
        "如果进入当前前十层实验，优先转化为可测试的搜索、reward、ranker 或 ablation 模块；如果属于世界模型/LLM/大规模泛化方向，则先作为后续 50 层扩展参考。\n"
    )


def write_notes(rows: list[dict[str, str]]) -> None:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    selected = deep_rows(rows)
    index = ["# Deep Research 精读索引\n"]
    for row in selected:
        path = NOTES_DIR / f"{row['id']}_{slug(row['title'])}.md"
        path.write_text(note_text(row), encoding="utf8")
        index.append(f"- [{row['id']} {row['title']}]({path.name}) - {row['experiment_role']}\n")
    (NOTES_DIR / "00_index.md").write_text("".join(index), encoding="utf8")


def slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return value[:70] or "paper"


def write_report(rows: list[dict[str, str]]) -> None:
    selected = deep_rows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["experiment_role"]] = counts.get(row["experiment_role"], 0) + 1
    role_lines = "\n".join(f"- `{role}`: {count}" for role, count in sorted(counts.items()))
    deep_lines = "\n".join(
        f"### {row['id']} {row['title']}\n"
        f"- 问题：长视野魔塔求解中与 `{row['experiment_role']}` 对应的瓶颈。\n"
        f"- 方法：{AREA_TEMPLATES.get(row['area'], row['mota_connection'])}\n"
        f"- 对魔塔启示：{row['mota_connection']}\n"
        f"- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。\n"
        for row in selected
    )
    REPORT_OUT.write_text(
        f"""# 《魔塔》前十层 Deep Research Report

生成方式：合并既有主文献、reward/因子文献，并补充硬探索、width-based planning、policy-guided search、offline/demo RL、资源约束规划等方向。

## Manifest 概览

- 总条目：{len(rows)}
- 精读条目：{len(selected)}
- Manifest：`paper/deep_research_manifest_200.csv`
- 精读笔记目录：`paper/deep_research_notes/`

## 角色分布

{role_lines}

## No Demonstration RL vs hp403 Warm-start

### 没有训练数据时怎么做 RL

魔塔前十层不是一个适合从随机 PPO/DQN 直接起步的任务。奖励极稀疏，动作后果会跨越几十到几百个宏动作，错误开门或过早打怪会让后续状态不可逆。因此无专家数据主线应当先由搜索产生训练数据，而不是让神经网络裸探索。

可执行路线是：

1. 用确定性模拟器做 Go-Explore/BFWS/staged search，保存成功和失败状态，而不是只保存最终路线。
2. 把失败路线按阶段重标：没打过队长但拿到剑、盾、红钥匙，都可以成为阶段成功样本。
3. 用搜索 trace 训练 Graph Q / policy-value：输入全节点资源图，输出每个节点 Q、stage value、deadend risk。
4. 把 learned prior 接回搜索，形成 expert-iteration：搜索生成更好数据，模型再提高下一轮搜索效率。
5. 所有路线必须 strict replay；relaxed negative-HP 只用于发现结构。

### 使用 hp403 会不会更好

会更好，但它回答的是另一个实验问题。`hp403` 提供完整通关阶段顺序，可以显著缓解早期探索真空，适合做行为克隆初始化、offline Q 初始化、self-imitation 或 DQfD 风格 warm-start。它还能作为 reward sanity check：如果某个 reward 配置给 `hp403` 很低分，说明 reward 很可能方向错误。

风险也很明确：只有一条路线，模型可能过拟合单一路线；它可能遮蔽更优路线；如果把它放进训练 replay buffer，就不能再把结果称为纯无专家数据 RL。因此实验必须分为三条线：

- `pure_search_rl`：不用 `hp403`，只用搜索自举数据。
- `hp403_warmstart`：允许用 `hp403` 做 BC/offline Q/self-imitation 初始化。
- `hp403_benchmark`：只做可视化和最终对照。

### 推荐结论

科研主线应该优先做 `pure_search_rl`，保证方法论干净；工程推进可以并行做 `hp403_warmstart`，用它验证模型结构、reward 因子和可视化播放链路。最后报告中分开比较成功率、final HP、route length、expansions 和是否找到不同于 `hp403` 的路线。

## 工程路线

- 建模：独立 `MotaResourceGraph`，不替换可视化工具。
- 规划：Go-Explore archive + BFWS novelty + dominance pruning。
- Reward：PBRS，Phi 包含英雄资源、可达资源、不可达关键资源、怪物伤害、ATK/DEF 临界、10F 资源和 boss margin。
- 学习：GraphPolicyValueNet / policy-value 只做搜索 prior 和状态排序，不直接承担裸探索。
- 验收：100 局 fixed seed，`10f战胜骷髅队长=true` 成功率 >=95%。

## 60 篇精读摘要

{deep_lines}
""",
        encoding="utf8",
    )


def main() -> None:
    rows = build_manifest()
    write_manifest(rows)
    write_notes(rows)
    write_report(rows)
    print(f"wrote {MANIFEST_OUT.relative_to(ROOT)} ({len(rows)} rows)")
    print(f"wrote {REPORT_OUT.relative_to(ROOT)}")
    print(f"wrote {NOTES_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
