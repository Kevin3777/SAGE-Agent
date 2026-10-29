# Engineering Assumptions

The paper defines high-level mechanisms but leaves several implementation choices open. These choices make this small reproduction executable; they are not claimed as paper-prescribed constants or algorithms.

## Causal timing

The Assistant receives only rewards and Checker feedback from earlier iterations. Current-iteration feedback is produced after the model call and can influence only later iterations.

## Policy adaptation

Eqs. 2–3 do not prescribe a unique optimizer. `EvolutionGoal` records memory and strategy objectives, failure evidence, proposed changes, and references. `PolicyRevision` records a version, preceding version, revised instructions, selected memories, corrective plan, and originating feedback ID. The implementation changes prompts and plans, not model weights.

## Reflection

`ReflectionGenerator` uses a deterministic evidence-constrained template. It specializes cause and correction wording for the local QA and SQL task families. `ReflectionValidator` verifies required fields, feedback provenance, confidence range, evidence references, and consistency with Checker errors. Confidence is engineering metadata, not a calibrated probability. Invalid reflections are not written.

## MemorySyntax and token estimates

The local optimizer extracts salient facts, named entities, constraints, Checker errors, causes, corrections, reusable lessons, and evidence. It removes duplicated wording and validates protected content. Token counts use a lexical proxy rather than a model tokenizer. The before count includes original information, reflection fields, task constraints, and evidence references; the after count measures the optimized representation. A strength increase requires both successful preservation checks and lexical-proxy compression.

## Strength and time units

The paper does not prescribe an exact strength estimator or unit. Seconds are the default; configuration also supports minutes, hours, and days. The estimator uses a normalized weighted score: 0.30 Checker quality, 0.15 constraint importance, 0.20 task importance, 0.20 reflection reusability, 0.10 evidence count, and 0.05 prior reuse. The score maps to an equivalent decay-resistance duration from 60 to 3600 seconds, converted to the selected unit. These weights and range are engineering assumptions.

## STM and LTM strength branch

STM retention uses optimized strength $S^{&#42;}$. On promotion, LTM begins a new decay epoch at migration time and retention uses base strength $S$. The lifecycle event preserves the pre-transition retention that caused promotion.

## Threshold interpretation

The implementation follows the paper’s prose interpretation: $R \ge 0.7$ stays in STM; $0.3 \le R < 0.7$ promotes to LTM; $R < 0.3$ is forgotten and deleted from searchable storage. The Eq. 8 expression/prose mismatch is documented in `EQUATION_AUDIT.md`.

## Retrieval and reinforcement

Retrieval uses lightweight lexical Jaccard overlap with a bonus for matching task type; no embedding service is required. Access increments the reinforcement count, updates access time and retention, and increases active strength by five percent. This is a local engineering rule.

## Tasks and Checker

The QA Checker uses normalized answer comparison plus required evidence passage IDs. The SQL Checker executes one task against a generated local SQLite database and compares column order and rows. SQL uses a `mode=ro` connection and SQLite authorizer allowing read-only query actions; the demo database is not connected to an external service.
