"""Invariant sweep: run many games and assert everything that must always hold.

Not part of the pytest suite -- it plays 180 games and takes a few minutes --
but it is where most of the interesting bugs came from, so it lives in the repo
rather than in a scratch directory:

    python -m werewolf_harness.tests.invariant_sweep

Unit tests check the cases someone thought of. This checks that no game, under
any guard configuration, ever violates a rule of the game or an accounting
identity of the log.
"""
import sys
from collections import Counter
from werewolf_harness.evalkit.runner import RunConfig, run_game

PROBLEMS = []

def check(cond, msg, ctx):
    if not cond:
        PROBLEMS.append(f"[{ctx}] {msg}")

def audit(log):
    gid = f"{log['game_id']} seed={log['seed']} guard={'+'.join(log['config']['guard_layers']) or 'none'}"
    check(not log["outcome"]["crashed"], f"crashed: {log['outcome'].get('crash_reason')}", gid)
    roles = log["ground_truth"]["roles"]
    wolves = {int(p) for p, r in roles.items() if r == "werewolf"}
    alive = set(range(1, 9))
    potions = Counter()
    checked = set()

    for rnd in log["rounds"]:
        r = rnd["round"]
        ctx = f"{gid} r{r}"
        # rnd["alive"] is recorded AFTER the night resolves, so night deaths
        # come off first and then it must match exactly.
        deaths_pre = set(rnd["night_deaths"])
        check(deaths_pre <= alive, f"night killed someone already dead: {deaths_pre - alive}", ctx)
        alive -= deaths_pre
        check(set(rnd["alive"]) == alive,
              f"alive mismatch: log={rnd['alive']} expected={sorted(alive)}", ctx)

        # --- night ---
        night_actors = [t["player_id"] for t in rnd.get("night_turns", [])]
        check(len(night_actors) == len(set(night_actors)), "a seat acted twice in one night", ctx)
        for t in rnd.get("night_turns", []):
            pid, na = t["player_id"], t.get("night_action") or {}
            check(pid in alive | deaths_pre, f"dead seat {pid} acted at night", ctx)
            role = roles[str(pid)]
            check(role != "villager", f"villager {pid} got a night turn", ctx)
            act, tgt = na.get("action"), na.get("target")
            if act == "night_check":
                check(role == "seer", f"{role} used night_check", ctx)
                check(tgt not in checked, f"seer re-checked {tgt}", ctx)
                check(tgt != pid, "seer checked itself", ctx)
                checked.add(tgt)
            elif act == "night_kill":
                check(role == "werewolf", f"{role} used night_kill", ctx)
                check(tgt not in wolves, f"wolf named packmate {tgt}", ctx)
                check(tgt in alive | deaths_pre, f"wolf named dead {tgt}", ctx)
            elif act in ("night_save", "night_poison"):
                check(role == "witch", f"{role} used {act}", ctx)
                potions[act] += 1
                check(potions[act] <= 1, f"{act} used {potions[act]} times", ctx)
            check(t["task"] == "night", "night turn not tagged as night", ctx)
            check(t.get("model"), "turn has no model tag", ctx)

        # --- day ---
        speakers = [a["player_id"] for a in rnd["agents"] if a["task"] == "speak"]
        voters = [a["player_id"] for a in rnd["agents"] if a["task"] == "vote"]
        if rnd["agents"]:
            check(sorted(speakers) == sorted(alive), f"speakers {sorted(speakers)} != alive {sorted(alive)}", ctx)
            check(sorted(voters) == sorted(alive), f"voters {sorted(voters)} != alive {sorted(alive)}", ctx)
            check(speakers == sorted(speakers), f"speech order not 1..8: {speakers}", ctx)
            orders = [a["speech_order"] for a in rnd["agents"] if a["task"] == "speak"]
            check(orders == list(range(len(orders))), f"speech_order gaps: {orders}", ctx)

        tally = Counter()
        for a in rnd["agents"]:
            if a["task"] != "vote":
                continue
            v = a["vote"]
            check(v is None or v in alive, f"seat {a['player_id']} voted dead/absent {v}", ctx)
            check(v != a["player_id"], f"seat {a['player_id']} voted for itself", ctx)
            if v is not None:
                tally[v] += 1
        logged = {int(k): v for k, v in (rnd.get("vote_counts") or {}).items()}
        check(dict(tally) == logged, f"tally {dict(tally)} != logged {logged}", ctx)

        exiled = rnd.get("exiled")
        if tally:
            top = max(tally.values())
            leaders = [p for p, c in tally.items() if c == top]
            expected = leaders[0] if len(leaders) == 1 else None
            check(exiled == expected, f"exiled {exiled} but tally says {expected} ({dict(tally)})", ctx)
        else:
            check(exiled is None, f"exiled {exiled} with no votes", ctx)
        if exiled is not None:
            check(exiled in alive, f"exiled a dead seat {exiled}", ctx)
            alive.discard(exiled)

        # --- traces ---
        for a in rnd["agents"] + rnd.get("night_turns", []):
            steps = a["react_trace"]
            check(len(steps) <= log["config"]["max_react_steps"] + 1,
                  f"seat {a['player_id']} used {len(steps)} steps", ctx)
            check([s["step"] for s in steps] == sorted(s["step"] for s in steps),
                  f"seat {a['player_id']} step numbers out of order", ctx)

    # --- outcome ---
    live_wolves = alive & wolves
    live_village = alive - wolves
    winner = log["outcome"]["winner"]
    if winner == "village":
        check(not live_wolves, f"village won with wolves alive: {live_wolves}", gid)
    elif winner == "werewolf":
        check(len(live_wolves) >= len(live_village) or len(log["rounds"]) >= 6,
              f"wolves won at {sorted(live_wolves)} vs {sorted(live_village)}", gid)
    else:
        check(False, "no winner recorded", gid)

    # --- payload bookkeeping ---
    for p in log.get("planted_payloads", []):
        check(p["attacker"] in wolves, f"non-wolf {p['attacker']} planted a payload", gid)
        check(p["target"] not in wolves, f"payload aimed at wolf {p['target']}", gid)

CONFIGS = []
for seed in range(20):
    for layers, ev in [((), False), (("L1", "L2"), False), (("L1", "L2", "L3"), True)]:
        for atk, ben in [(True, False), (False, True), (False, False)]:
            CONFIGS.append(RunConfig(seed=seed, guard_layers=layers, evidence_forced=ev,
                                     attack_enabled=atk, benign_persuasion=ben))

print(f"auditing {len(CONFIGS)} games...", file=sys.stderr)
for i, cfg in enumerate(CONFIGS):
    audit(run_game(cfg))
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{len(CONFIGS)}  problems so far: {len(PROBLEMS)}", file=sys.stderr)

print(f"\n=== {len(PROBLEMS)} problems ===")
seen = Counter()
for p in PROBLEMS:
    key = p.split("] ", 1)[1].split(":")[0][:70]
    seen[key] += 1
for key, n in seen.most_common():
    print(f"{n:4d}x  {key}")
print("\n--- first 8 raw ---")
for p in PROBLEMS[:8]:
    print(" ", p)
