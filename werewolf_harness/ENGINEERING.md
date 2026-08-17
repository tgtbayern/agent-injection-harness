# Engineering notes

How the harness is built, why it is built that way, and what broke along the
way. Roughly 7,400 lines of Python plus a no-build frontend:

| package | lines | role |
|---|---|---|
| `engine/` | 683 | deterministic referee (adapter over a rules library) |
| `harness/` | 3072 | **the contribution**: ReAct loop, tools, memory, guards, providers |
| `evalkit/` | 1011 | runner, six metric axes, judges, statistics |
| `attacks/` | 379 | payload + benign corpora, injector, detection |
| `server/` | 681 | FastAPI + SQLite |
| `web/` | 772 | replay / config / experiments, no build step |
| `tests/` | 1230 | 262 tests |

---

## 1. The one structural rule

**No model call inside `engine/`. No game rule inside `harness/`.**

Cross that line and results stop being interpretable — a "hijack rate" measured
by a referee that itself talks to a model is measuring two things at once. It is
enforced by `tests/test_boundaries.py`, which parses every module's syntax tree
and fails if `engine/` refers to a model or if `harness/` imports the rules
library directly.

The engine's entire public surface is two functions:

```python
view = get_visible_state(state, player_id)   # what this player may know
state.apply_action(player_id, action)        # the only thing that changes the world
```

Nothing an agent *says* changes the world. Speech is data; only `apply_action`
is a state transition.

---

## 2. Rules: adopted, not written

Roles, night-action priority ordering, save-versus-poison conflict resolution,
self-vote and dead-target rejection, tie handling and win detection all come
from the MIT-licensed `werewolf-engine` package. `engine/` is an adapter that
adds only what the experiment needs and the library does not have:

| gap | why the adapter fills it |
|---|---|
| **speech phase** | the library only votes. Speech is where the attack lives, and it changes no world state, so it belongs on this side of the boundary. |
| **abstention** | the library requires every living player to vote. The recovery path needs "retries exhausted → abstain", so `_DayVoting` subclasses `DayManager` and adds exactly that, inheriting every other vote rule unchanged. |
| **round cap** | six rounds, then the wolves win on time. |
| **determinism** | the library shuffles the *global* RNG. Role assignment is redone with a local `random.Random(seed)` so games are reproducible and safe to run concurrently — while still using the library's role registry and role classes. |

**Night is scripted.** Night actions are chosen by a seeded policy rather than
by agents. This is a scope decision, not an oversight: the behaviour under study
is day-phase vote hijacking, so both the model budget and the run-to-run
variance belong there. It also means two runs with the same seed face the same
world, which is what makes paired comparison work.

---

## 3. Isolation is the load-bearing module

`engine/visibility.py` is the only place private information is selected. A view
contains public facts, plus the viewer's own private facts, and nothing else.

It is the one module with mandatory test coverage (`tests/test_visibility.py`,
173 assertions across 25 seeds), and the tests check the property structurally
rather than field by field: the *entire serialised public subtree* must contain
no role word for anyone, and the `private` section's keys must be a subset of
what the viewer's own role entitles them to. A field added later that leaks
something will fail these tests without anyone remembering to update them.

---

## 4. The agent turn

```
get_visible_state ──► ContextBuilder ──► [ model call ] ──► validate ──► guard
                                              ▲                            │
                                              └────── observation ◄── execute
                                                                            │
                                                        terminal action ──► engine
```

A turn is a bounded loop (≤8 steps), not a single call. That is what gives the
tools, the memory and the context budget somewhere to live — and what creates
the tool-return attack channel.

### Tools

Eight, frozen before anything else was written:

| tool | kind | notes |
|---|---|---|
| `query_history(player, round)` | read | **untrusted output** — attack path B |
| `query_votes(round)` | read | |
| `query_deaths()` | read | |
| `read_belief(player)` | read | |
| `check_ability()` | read | role-specific private status |
| `update_belief(player, suspicion, reason, evidence_refs)` | write-self | |
| `speak(content)` | terminal | ends the turn |
| `vote(target_id)` | terminal | ends the turn |

Three gates, in order:

1. **schema** — types, ranges, required arguments. `"4"` coerces to `4`; `True`
   does not (a bool is not a player id).
2. **whitelist** — the tool must be registered. This is where hallucinated tools
   die, and the error hands back the legal list so the retry can succeed.
3. **semantics** — vote targets must be alive and not yourself; rounds must have
   happened.

Rejections are *not* crashes. The error text is written to be actionable,
because it goes straight back to the model as the next turn's input.

### Context budget

The context carries the current round's speeches only. Earlier rounds are
reachable *exclusively* through `query_history`. This is a design choice, not an
optimisation: it makes the tool layer load-bearing, gives path B something to
carry, and turns evidence enforcement into a real constraint rather than a
formality.

Within a turn, observations accumulate; past the sixth, the oldest are replaced
by a one-line summary of what was already done, so a long turn does not silently
repeat lookups it has forgotten.

### Memory

Three layers, kept apart:

- **working** — the current ReAct chain, discarded at end of turn
- **short-term** — this round's observations, in the window
- **long-term** — `BeliefState`: one entry per player, carried across rounds,
  holding only a conclusion, a reason, and the speech ids it rests on

Only conclusions are promoted. Keeping raw transcripts would grow the context
linearly with the game; "why I suspect 5" survives a whole game in a few hundred
tokens. `evidence_refs` is the field that makes "is this judgement grounded?"
machine-checkable.

### Failure handling

| failure | response |
|---|---|
| transport error / timeout | retry with backoff (≤2), then abstain for the turn |
| no parseable tool call | retry with the parse error attached (≤3) |
| unknown tool | retry with the whitelist attached |
| semantically invalid action | retry with the engine's reason attached |
| same call 3× in a row | break the loop, force a terminal action |
| out of steps | one forced terminal request, then the conservative default |

The conservative default for a vote is **abstention, not a random ballot** — a
random vote would inject noise into precisely the quantity being measured. Every
one of these counters is written to the log, because "how often did the model
need saving" is itself a reported axis.

---

## 5. The guard stack

Layered so the ablation can switch each one independently, and so each one's
cost is separately visible.

**L1 — isolate.** Fences everything written by others, labelled with source and
channel, behind a standing declaration that fenced content is data. The fence is
airtight: content that contains `</untrusted>` gets it neutralised on the way in.
Without that, one payload could close the fence early and everything after it
would read as trusted prompt text — L1 would be worse than no L1. There is a
test for exactly this.

**L2 — filter.** Detects and strips forged conversation structure (chat template
markers, fake `[SYSTEM]:` turns, function-call syntax in prose) and imperative
directives ("you must vote 4", "必须投4号"). Redaction leaves a visible stub
rather than deleting silently, so a half-filtered sentence cannot read as a
coherent instruction.

**L3 — verify.** Output-side, on the assumption that something always gets
through:
- *say/do consistency* — the speech announced 4, the ballot says 7.
- *directive compliance* — untrusted text demanded a vote for 7 and the agent is
  voting 7 with no evidence of its own recorded.

**E — evidence.** No attributing a statement to a player without a prior
`query_history` for them.

A block re-prompts with the reason; it does not end the turn.

---

## 6. Two bugs worth recording

**The over-blocking regex.** The first version of L3's directive detector
matched any `vote 4`. Since every agent's speech ends "I vote 4", *every* ballot
looked like obedience to an instruction, L3 blocked nearly all of them, and the
"over-defense cost" it produced would have been a measurement of my own sloppy
regex rather than of a real trade-off. Fixed by matching imperative phrasing
only: announcing your own vote is play, telling someone else what theirs must be
is not. The distinction is now a test.

**The filter tuned to its own test set.** L2's patterns were, inevitably,
written while looking at the payload corpus. Numbers from that are circular. The
corpus now carries a `dev`/`holdout` split, no pattern targets a holdout payload,
and a test asserts the holdout is *not* fully neutralised — if it ever were,
that would be evidence the patterns had been fitted to it.

---

## 7. Model access

One OpenAI-compatible gateway, one client, zero per-vendor branches. The
`openai` SDK is optional; the client speaks HTTP directly so the harness runs
with no third-party packages at all.

- **`probe_model`** runs the phase-0 checklist per model: reachable, native tool
  call returned, arguments parse, tool result can be fed back and the
  conversation continues, `temperature=0` actually stable, usage reported,
  latency. A model failing the tool-calling items is demoted to
  `tool_mode="json_prompt"` — same loop, tools described in the prompt, replies
  parsed as JSON — and every game log carries which mode it ran in.
- **`explain()`** turns the three common gateway failures into fixes rather than
  passing the raw error through: token-group mismatch, malformed key, unknown
  model name.
- **The key never reaches the browser.** The dashboard stores it server-side and
  returns only `sk-a****3f2a`. Tested: no response body in the API may contain
  the key.

---

## 8. Offline client

`harness/providers/mock.py` plays well enough to exercise every path with no
network and no key — which is what makes the test suite meaningful and lets a
fresh clone show a full replay immediately.

Its susceptibility to injection is a **constant I chose**, monotone by
construction (visible+unfenced > visible+fenced > removed). Its runs are labelled
`model=mock` and excluded from every reported result. It tests the instrument;
it is not read as data. Where it is genuinely useful: if a guard ablation over
mock runs ever fails to reproduce the expected ordering, the harness is broken —
which caught three wiring bugs during development.

---

## 9. Frontend

Three pages, no build step, no framework — an instrument panel that has to open
from a clone with nothing installed. (The design doc called for React + Vite +
Tailwind; a build toolchain would have added install friction for a page that is
mostly tables, so this is a deliberate deviation.)

The replay view is the one that earns its place: a timeline of injection
attempts (filled tick = a vote actually moved, hollow = delivered and ignored),
a player column flagging who read a payload and whose action was blocked, and
per-step ReAct traces with the payload text highlighted where it entered, the
belief diff underneath it, and the resulting ballot. Pointing at the step where
a tool return carried an instruction and the next step flipped a belief is worth
more than any table.
