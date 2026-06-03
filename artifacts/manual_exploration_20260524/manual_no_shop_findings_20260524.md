# Manual no-shop exploration findings - 2026-05-24

> Superseded: this interim note was too pessimistic. It has been superseded by
> `manual_success_no_shop_true_10f_trap.jsonl` and
> `manual_success_report_20260524.md`, which document a no-4F-shop, no-fly
> route that defeats the 10F skeleton captain through the trap.

Scope:
- This note records a separate manual exploration pass.
- I did not run RL, Q-learning, PPO, graph Q, staged search, beam search, or reward tuning for this pass.
- The simulator was only used to replay hand-picked macro actions and compute deterministic HP/stat outcomes.
- Files from previous experiments were not overwritten.

Current environment assumptions:
- Scenario starts after the MT2 thief / MT3 demon story reset.
- 4F shop is disabled.
- Fly item is disabled.
- MT6 blue-key merchant and MT7 yellow-key merchant are available once each.
- Gems are +1 ATK / +1 DEF.
- Goal remains `flag:10f战胜骷髅队长=true`.

Hard 10F threshold:
- With all accessible attack and defense resources but no 4F shop, the hero reaches ATK 27 / DEF 27.
- At ATK 27 / DEF 27:
  - skeleton damage = 30
  - skeletonSoldier damage = 75
  - skeletonCaptain damage = 304
- The 10F trap route requires clearing 6 skeletons, 2 skeleton soldiers, then the captain.
- Required HP before the red-door boss sequence is therefore `6*30 + 2*75 + 304 = 634`, and the simulator needs strictly more HP than each fight damage.

Best validated no-shop checkpoint:
- The best existing no-shop route reaches MT10 before opening the red door with:
  - HP 397
  - ATK 27
  - DEF 27
  - redKey 1
  - yellowKey 0
  - blueKey 0
  - money 180
- This is short by about 238 HP for the trap plus captain sequence.

Manual hypotheses tested:
- Move 1F resources earlier after shield:
  - This improves some intermediate checkpoints, but after replaying the remaining route the red-door checkpoint is still only about HP 388-397.
- Insert full 1F resource collection before shield:
  - After inserting it after the early MT3/MT4 gems, shield checkpoint becomes HP 196, ATK 25, DEF 23.
  - This is not better than the old shield checkpoint HP 207, ATK 24, DEF 22 once later duplicated resources are removed.
- Check remaining HP resources at the red-door checkpoint:
  - MT2 blue potions remain on the map but are not reachable from the current topology.
  - MT8 lower-right resources at 7,10 / 8,10 / 7,11 remain on the map but are not reachable through legal macro actions.
  - MT9 yellow keys at 1,7 and 5,7 are net-zero or HP-negative key conversions and do not solve the HP deficit.
- Check 10F original skeleton captain at 6,4:
  - The macro action label can say `fight skeletonCaptain MT10:6,4`, but executing the path steps on MT10:6,5 first, triggers the trap, and the actual fight becomes a skeleton soldier at 6,4.
  - This does not bypass the trap or reduce the required boss-route damage.

Conclusion:
- Under the current simplified no-shop simulator, I do not have a successful legal manual route.
- The blocker is mechanical, not reward-related: after all reachable stat resources, the hero has 27/27, and the best red-door HP checkpoint is roughly 397 against a 634 HP requirement.
- A successful first-10-floor route likely requires one of:
  - re-enabling the original 4F shop mechanics;
  - finding a missing original event/resource that the current simulator does not implement;
  - changing the simplified target to stop before the full 10F trap captain fight.

Useful sanity check:
- If 4F shop were available, three defense purchases would raise DEF from 27 to 39, reducing the 10F trap plus captain damage from 634 to 322. That would make the existing HP 397 checkpoint feasible.
