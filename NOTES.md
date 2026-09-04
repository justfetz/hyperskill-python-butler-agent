# Butler Agent — Working Notes

Raw material for the final README. Two threads: **what an agent is** (concepts)
and **what broke** (failure log).

---

## Failure log

Five distinct failures before stage 2 was green. None were agent-logic bugs —
worth noting in itself. Diagnosing an agent means first asking *which layer*
failed: config, transport, billing, or the loop.

### 1. Wrong environment variable name
- **Symptom:** `openai.OpenAIError: api_key must be set`
- **Cause:** `.env` said `API_KEY=`, code read `OPENAI_API_KEY`.
- **Fix:** rename the var.
- **Lesson:** `load_dotenv()` silently succeeds when it finds nothing useful.
  A missing var and a wrong-named var are indistinguishable at the call site.

### 2. TLS interception by a corporate proxy
- **Symptom:** `APIConnectionError` wrapping
  `SSL: CERTIFICATE_VERIFY_FAILED — unable to get local issuer certificate`
- **Cause:** Cisco Umbrella re-signs HTTPS with a corporate root CA. Windows
  trusts it; `httpx` (used by the OpenAI SDK) validates against **certifi's**
  bundle, which does not.
- **Diagnosis:** validate the same host against different trust stores —
  certifi failed, Windows store succeeded, issuer = `Cisco Umbrella ... SubCA`.
- **Fix (final):** `ca-bundle.pem` = certifi + Windows ROOT/CA stores, passed as
  `httpx.Client(verify=...)`.
- **Lesson:** "it works in the browser" proves nothing. Python ships its own
  trust store.

### 3. Zero credit balance
- **Symptom:** `429 insufficient_quota — credit_balance_exhausted`
- **Cause:** no credits on the OpenAI **organization**.
- **False lead:** made a new key in a new project. Same error — credits are
  org-level, and every project draws from one balance.
- **Fix:** add credits.
- **Lesson:** 401 = who you are. 429/insufficient_quota = what you owe. A
  healthy key that authenticates can still be unable to spend. `/v1/models` is
  unbilled, so it isolates auth from billing.

### 4. Retry budget exceeded the grader's timeout  *(self-inflicted)*
- **Symptom:** hstest: *"running for more than 15 seconds ... infinite loop"*
- **Cause:** raised `max_retries=5, timeout=30.0` to survive #2. The SDK's
  backoff is exponential capped at 8s: 0.5+1+2+4+8 = **15.5s of sleeping**,
  before any request time.
- **Fix:** `max_retries=2` (1.5s backoff), `timeout=8.0` so a hung attempt plus
  its retry still lands inside 15s.
- **Lesson:** retry budgets are bounded by the deadline of whoever is waiting.
  Also: "infinite loop" in a timeout message is a guess, not a diagnosis —
  `MAX_ITERATIONS` made a real infinite loop impossible.

### 5. truststore instability under the grader
- **Symptom:** `APIConnectionError`, always test #3, always at
  `truststore/_windows.py:562` (the Windows cert-store lookup).
- **Ruled out with evidence:** repeated `inject_into_ssl()` (idempotent, tested
  3x); thread safety (tested 4 threaded requests); flakiness (5 clean suites).
- **Cause:** truststore hits the OS crypto API via ctypes on *every handshake* —
  a live stateful call in the hot path.
- **Fix:** bake the certs into a static `.pem` once; handshake becomes a file
  read. Same fix as #2, and it removed the failing code path entirely.
- **Lesson:** 5 passing runs is not proof. The argument that mattered was
  mechanistic — the failing code no longer executes.

### 6. The CA bundle file vanished
- **Symptom:** `CERTIFICATE_VERIFY_FAILED` returned after being fixed; the
  `ca-bundle.pem` that had made 5 suites pass was simply gone.
- **Cause:** the IDE manages the task directory and removes files not declared
  in `task-info.yaml`.
- **Fix:** stop depending on a file. `build_ssl_context()` reads certifi's roots
  plus the Windows store via `ssl.enum_certificates()` at import, in-process.
  Pure stdlib, no truststore, no artifact to lose, and still only one OS read
  per run rather than one per handshake.
- **Lesson:** a fix that depends on a file someone else manages isn't a fix.
  The same problem recurring with a *different* traceback is real information.

### Reading a failure fast
How far the log got localises the fault before the traceback does:

```
[USER]: ...
[ENTERING AGENT LOOP]          <- stopped here = transport. No THINK means the
                                  model never answered; loop logic never ran.
[THINK]: [...]                 <- got here = the API worked; suspect the loop.
```

---

## Concepts

**The definition.** An agent is an LLM in a loop with tools, where the model —
not the programmer — decides each iteration whether to act or to answer.

Everything follows from that sentence:
- the **loop** exists because there is more than one turn;
- **two registries** exist because tools must be both *described* (to the model)
  and *implemented* (in code);
- **context accumulation** exists because the model is stateless — each call
  receives the whole history or knows nothing;
- **MAX_ITERATIONS** exists because "the model decides when to stop" needs a
  backstop.

**The security boundary.** The model never executes anything. It emits a *name*
and a *JSON blob*; your dispatch table decides whether to honour it. An agent
can do exactly what is in `TOOL_NAME_TO_FUNC` and nothing else.

**Latency compounds.** Every iteration is a network call, so an agent's latency
is the sum of its iterations, multiplied by retry policy. See failure #4.

## Stage progression
| Stage | Shape | New idea |
|---|---|---|
| 1 Basic chat loop | `ask -> answer` | THINK only; `output` is a typed list |
| 2 ACT introduced | `ask -> [act, observe]* -> answer` | the loop; tool dispatch; `call_id` |
| 3 Multiple tools | same loop, richer tools | parameters, state, chaining, errors-as-data |


---

## Stage 3 — what multiple tools teach

### Tools come in three kinds
| Tool | Params | Reads | Writes | Kind |
|---|---|---|---|---|
| `check_weather()` | none | outside world | -- | sensor |
| `get_wardrobe_items()` | none | state | -- | state reader |
| `wash_clothing(item_name)` | `item_name` | state | **state** | actuator |

The third is the leap. A read-only agent is a search engine with extra steps.
An agent that writes changes the world, and "don't ask for permission" in the
system prompt stops being convenience and becomes policy.

### Errors are data, not exceptions
`wash_clothing("red shirt")` **returns** `"Item 'red shirt' not found in
wardrobe"`. It does not raise. That string lands in context as a
`function_call_output`, the model reads it, and recovers on its own -- it called
`get_wardrobe_items` to find out what actually exists, then offered
alternatives. Nobody wrote that recovery logic.

A raised exception kills the loop and teaches the model nothing. A returned
error is a teaching signal. **Tool errors should be returned, not raised.**

### Chaining has two shapes
```
Parallel (independent) -- several calls in ONE response:
  THINK -> ['ResponseFunctionToolCall', 'ResponseFunctionToolCall']
           check_weather + get_wardrobe_items together

Sequential (dependent) -- one call per iteration:
  THINK -> get_wardrobe_items   (must see what is dirty...)
  THINK -> wash_clothing        (...before washing it)
```
The model batches what is independent and serialises what is not. This is why
the loop iterates *every* item in `response.output` instead of taking `[0]` --
defensive-looking in stage 2, load-bearing in stage 3.

### Context accumulation is visible
Turn 1 asked about the wardrobe and called a tool. Turn 2 asked again and called
**no tools at all** -- the answer was already in context. Cheaper and faster,
but it also means the agent can answer from stale memory after state changes
underneath it. Memory and freshness are in tension.

### One loop fix stage 3 forced
Stage 2 returned as soon as any `message` appeared. If a response contains a
tool call *and* a message, that tool's result would never reach the model. The
rule is: final only when the response contained **no** function calls.
```python
if answer is not None and not called_tool:
    return answer
```

### The iteration budget is tight
`MAX_ITERATIONS = 5`. The task's own sequential walkthrough uses exactly 5
THINK phases -- zero slack. Parallel calls are what buy headroom: batching
weather+wardrobe turned a 5-iteration chain into 3. If a run ever ends with
"I wasn't able to finish within the iteration limit", that is the cause.

### Measured
Stage 3 suite: 5 tests, ~25-31s total, ~5s per test against a 15s cap. Slower
than stage 2 (~3.3s) because chains mean more round trips. Agent latency is the
sum of its iterations.
