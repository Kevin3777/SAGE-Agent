# Paper-to-Code Mapping

This table maps the method components to the repository’s implementation and observable checks. The paper-to-code relations describe an independent engineering reproduction, not an official implementation.

| Paper component | Reference | Implementation | Validation evidence | Notes |
| --- | --- | --- | --- | --- |
| User / Assistant / Checker roles | Section 3.1 | `src/sage/agents/user.py`, `src/sage/agents/assistant.py`, `src/sage/feedback/checker.py` | `tests/unit/test_reflection_agent.py`, `tests/integration/test_end_to_end.py` | Checker feedback is task-environment evidence, not an LLM judge |
| Iterative feedback | Eq. 1 | `src/sage/agents/agent.py` | A prior feedback ID and message appear in the next backend input | Current feedback is generated only after the current action |
| Evolution goal and policy revision | Eqs. 2–3 | `src/sage/core/types.py`, `src/sage/agents/agent.py` | Versioned instructions and corrective plan are asserted in agent tests | Non-parametric prompt/plan adaptation |
| Reflection and memory insertion | Eqs. 4–5 | `src/sage/reflection/generator.py`, `schema.py`, `validator.py` | Structure, provenance, evidence, and Checker error validation are tested | Deterministic evidence-constrained generator |
| STM | Section 3.2 | `src/sage/memory/store.py`, `src/sage/memory/manager.py` | Capacity, recent trajectory, retrieval, and virtual-time transition tests | Bounded in-process recent trajectory plus timestamped reflection records |
| LTM and retrieval | Section 3.2 | `src/sage/memory/store.py`, `src/sage/memory/manager.py` | SQLite persistence, deduplication, ranking, deletion, reopen, and retrieval tests | SQLite; no separate database service |
| Ebbinghaus curve | Eq. 6 | `src/sage/memory/forgetting.py`, `src/sage/core/clock.py` | Numerical, monotonicity, invalid-input, SystemClock/FakeClock tests | Configurable unit |
| MemorySyntax | Eq. 7 | `src/sage/memory/syntax.py`, `src/sage/core/types.py` | Extraction, fact/entity/constraint preservation, lexical compression proxy, and strength tests | Strength formula is an engineering assumption |
| Threshold lifecycle | Eq. 8 | `src/sage/memory/manager.py` | STM retention, LTM promotion, and searchable deletion at boundaries | Follows prose interpretation of the Eq. 8 ambiguity |
| QA environment | Section 3.1 | `src/sage/environments/qa.py` | Multi-hop answer and required-passage checks | Small, hand-authored local tasks |
| SQL environment | Section 3.1 | `src/sage/environments/sql.py` | Actual SQLite JOIN and unsafe-query denial tests | `mode=ro` plus SQLite authorizer |
| Mock / GGUF / compatible API | Implementation | `src/sage/backends/` | Backend distinction tests and historical local inference trace | Paid API use is opt-in and was not used for validation |
| CLI and baseline comparison | Implementation | `src/sage/cli.py` | CLI integration tests and documented command smoke checks | `base`, `reflection`, and `full` are small functional modes |

## Iteration order

1. Read current task state, STM history, and relevant LTM experiences.
2. Build the Assistant prompt using only known prior feedback and rewards.
3. Generate and parse an action using the configured backend.
4. Evaluate the action in the task Checker.
5. Record the output, feedback, reward, and model-call metadata.
6. If successful, stop; otherwise generate and validate reflection.
7. In full mode, optimize a validated reflection and update STM/LTM state.
8. Revise the policy/prompt and proceed to the next iteration.

## Reproduction boundary

This repository implements core SAGE method behavior and a small functional comparison. It does not reproduce model training, complete benchmark suites, or paper-reported performance results.
