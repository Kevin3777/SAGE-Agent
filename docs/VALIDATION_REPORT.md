# Validation Report

**Validation scope:** independent core-method implementation and small functional tasks. This is not a reproduction of the paper’s benchmark results.

## Environment inspected

| Item | Observed value |
| --- | --- |
| Python | 3.13.5, 64-bit (Anaconda CPython build) |
| Python Launcher inventory | Python 3.10 32-bit only; Python 3.12 was not installed |
| GPU | NVIDIA GeForce RTX 3080, 10,240 MiB |
| Base PyTorch | 2.9.1+cpu; CUDA unavailable |
| Local inference package | `llama-cpp-python` 0.3.36 CPU wheel |
| Existing model format | GGUF, Q4_K_M |
| Model | Qwen2.5-1.5B-Instruct |

The model was loaded from an existing local file and was neither copied into the repository nor downloaded again. Recorded inference used CPU; CUDA inference is not claimed.

## Automated validation

The development copy completed `python -m compileall -q src tests scripts` successfully and the test suite reported **48 passed, 0 failed, 0 skipped**. Tests exercise the mathematical decay function, threshold boundaries, memory persistence and retrieval, reflection validation, Checker feedback propagation, early stop and max-iteration handling, SQLite read-only enforcement, CLI behavior, and end-to-end Mock tasks.

The small base/reflection/full comparison is in `src/sage/cli.py` and can be regenerated with:

```powershell
python -m sage.cli evaluate --backend mock --task all --max-iterations 3 --out outputs\evaluation.json
```

It is a functional comparison on two hand-authored task types, not a benchmark result.

## Historical real-model validation

These facts were cross-checked against local JSON traces before preparing this release. The raw development traces and SQLite files are intentionally excluded from the public repository.

| Run | Evidence |
| --- | --- |
| Clean QA | Local backend loaded; one model call and one Checker call; Qwen returned Wellington with p1 and p2; Checker passed. 250 input tokens, 38 output tokens, approximately 2.46 seconds generation latency. |
| Controlled first attempt | The input explicitly instructed the first attempt to cite p2 only. The actual local model cited p2; Checker feedback required p1 and p2. Three calls and three Checker failures reached the iteration limit; three validated reflections were generated and two distinct STM memories were written. This run failed. |
| Later process using persisted experiences | A separate run loaded the two persisted evidence-grounded memories, retrieved them, and passed the corresponding task in one local-model call. This does not change the result of the preceding failed run. |
| LTM retrieval | A validated reflection originating from a real local-model failure was promoted through a disclosed FakeClock lifecycle and reopened from SQLite. A subsequent local inference retrieved the LTM record and passed QA in one call. The virtual clock demonstrates lifecycle behavior; it is not elapsed wall-clock aging evidence. |

The model’s raw outputs were not edited to manufacture success. No paid API was used.

## MemorySyntax demonstration

For the documented example, $R(30,60)=0.6065306597$. The lexical token estimate changed from 177 to 93 (84 fewer estimated tokens) while the example’s protected fact and constraint checks passed. The computed strengths were approximately $S=2823.56$ seconds and $S^{&#42;}=3672.74$ seconds. In a FakeClock lifecycle, retention 0.69 caused STM-to-LTM promotion and retention 0.29 caused deletion from searchable SQLite storage. Boundary checks retain $R=0.7$ in STM and promote $R=0.3$ to LTM.

These token counts are a lexical proxy, not model-tokenizer counts. Strength values and mapping range are implementation assumptions, not paper constants.

## Release-copy verification

The public release copy was checked separately: `compileall` succeeded; `pytest` reported 48 passed; CLI help, environment/config loading, Memory Demo, Mock QA, Mock SQL, all three modes, evaluation, and memory inspection completed; README local links resolved; and a wheel was built, installed into an isolated target directory, and exercised with CLI help, Memory Demo, Mock QA, and Mock SQL. The local-model run was not repeated during release preparation; its evidence above is historical and explicitly identified as such.
