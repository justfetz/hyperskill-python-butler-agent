# Butler Agent

A 185-line LLM agent that checks the weather, looks in your wardrobe, and washes
your clothes — built to answer one question honestly: **what actually makes
something an "agent" rather than a chatbot?**

> An agent is an LLM in a loop with tools, where the model — not the
> programmer — decides each iteration whether to act or to answer.

Everything below falls out of that one sentence.

---

## It looks like this

One sentence in. Five tool calls out. Nothing in the code says that weather
relates to clothing, or that dirty things should be washed before wearing:

```
[USER]: What should I wear today?
[ENTERING AGENT LOOP]
[THINK]: Model decided to return these items: ['ResponseFunctionToolCall', 'ResponseFunctionToolCall']
[ACT]: Calling "check_weather" with arguments {}
[OBSERVE]: Result Cold, rainy
[ACT]: Calling "get_wardrobe_items" with arguments {}
[OBSERVE]: Result Item blue sweater is dirty; Item brown jacket is dirty
[THINK]: Model decided to return these items: ['ResponseFunctionToolCall', 'ResponseFunctionToolCall']
[ACT]: Calling "wash_clothing" with arguments {'item_name': 'blue sweater'}
[OBSERVE]: Result blue sweater is washed
[ACT]: Calling "wash_clothing" with arguments {'item_name': 'brown jacket'}
[OBSERVE]: Result brown jacket is washed
[THINK]: Model decided to return these items: ['ResponseOutputMessage']
[EXITING AGENT LOOP]
[ASSISTANT]: Today's weather is cold and rainy. I've washed your blue sweater
and brown jacket! You can wear the blue sweater for warmth, layered with the
brown jacket if you'd like extra protection from the rain.
```

The model planned that chain. The code only offered the tools and ran the loop.

## The loop

A chatbot is a straight line. An agent is the same call wrapped in a loop with a
conditional exit:

```
chatbot:  ask ──> answer
agent:    ask ──> [ answer? done : act, observe ] ↺
```

That is the entire difference. The model did not get smarter — it got a turn
structure that lets it stop mid-thought, ask for data, and resume.

```python
def run_agent_loop(context):
    answer = None

    for _ in range(MAX_ITERATIONS):
        response = client.responses.create(
            model=MODEL_NAME, instructions=SYSTEM_PROMPT,
            input=context, tools=TOOLS_REGISTRY,
        )
        context += response.output          # the model must see its own calls
        called_tool = False

        for item in response.output:
            if item.type == "function_call":
                called_tool = True
                result = TOOL_NAME_TO_FUNC[item.name](**json.loads(item.arguments))
                context.append({"type": "function_call_output",
                                "call_id": item.call_id, "output": result})
            elif item.type == "message":
                answer = "".join(p.text for p in item.content
                                 if p.type == "output_text")

        if answer is not None and not called_tool:
            return answer               # final only when it stopped asking
```

Four things in there are load-bearing.

**`context` is a list that grows.** The API is stateless. Each call receives the
entire history or the model knows nothing. After one weather exchange:

```
[0] dict                      {"role": "user", "content": "What should I wear today?"}
[1] ResponseFunctionToolCall  name=check_weather, arguments='{}', call_id='call_TU92…'
[2] dict                      {"type": "function_call_output", "call_id": "call_TU92…",
                               "output": "Cold, rainy"}
[3] ResponseOutputMessage     "The weather today is cold and rainy…"
```

Note that `[1]` — the model's *own* tool call — goes back in. It feels
redundant, but without it the model receives an answer to a question it has no
record of asking. `call_id` is the thread stitching `[1]` to `[2]`; with several
calls in flight it is the only thing saying which result belongs to which
request.

**`response.output` is a list of typed items, not a string.** Printing the class
names exposes the model's decision — `ResponseFunctionToolCall` means "I need
data first", `ResponseOutputMessage` means "I'm done". Iterating *every* item
rather than taking `[0]` looks like fussiness until the model returns two calls
at once, which it does routinely.

**The model decides when to stop.** The loop exits when prose arrives instead of
a tool call. That is the defining property — and the reason `MAX_ITERATIONS`
exists, because "the model decides" needs a backstop.

**Final means no tools were called.** If one response carries both a tool call
and a message, returning immediately would strand that tool's result where the
model never sees it.

## Tools come in three kinds

The three tools are picked to cover genuinely different things a tool can *be*:

| Tool | Params | Reads | Writes | Kind |
|---|---|---|---|---|
| `check_weather()` | none | outside world | — | **sensor** |
| `get_wardrobe_items()` | none | state | — | **state reader** |
| `wash_clothing(item_name)` | `item_name` | state | **state** | **actuator** |

The third row is the leap. A read-only agent is a search engine with extra
steps. An agent that *writes* changes the world, and `"Don't ask for permission"`
in the system prompt stops being a convenience and becomes a policy decision.

### The model never executes anything

Tools are declared twice, on purpose:

```python
TOOLS_REGISTRY    = [ {...json schema...} ]            # what the model is TOLD
TOOL_NAME_TO_FUNC = {"check_weather": check_weather}   # what actually RUNS
```

The model emits a *name* and a *JSON blob*. Your dispatch table decides whether
to honour it — a name absent from that dict raises `KeyError` instead of
running. **An agent can do exactly what is in that dictionary and nothing
else.** That split is the security boundary, and it belongs somewhere obvious in
any agent you build.

For parameterised tools, the JSON schema is the model's entire instruction
manual. This `description` is not a comment — it is the only thing telling the
model to pass `"blue sweater"` rather than `"the blue one"`:

```python
"item_name": {"type": "string", "description": "Name of the clothing item to wash"}
```

## Errors are data, not exceptions

The most interesting behaviour in the project is one nobody wrote.

`wash_clothing` **returns** a string on failure. It does not raise:

```python
if item_name not in WARDROBE:
    return f"Item '{item_name}' not found in wardrobe"
```

That string lands in context as an observation, the model reads it, and recovers
by itself:

```
[ACT]: Calling "wash_clothing" with arguments {'item_name': 'red shirt'}
[OBSERVE]: Result Item 'red shirt' not found in wardrobe
[ACT]: Calling "get_wardrobe_items" with arguments {}
[OBSERVE]: Result Item blue sweater is dirty; Item brown jacket is dirty
[ASSISTANT]: There is no red shirt — you have a blue sweater and a brown
jacket, both dirty. Would you like me to wash one of those instead?
```

No recovery logic exists in the code. The loop handed the failure back as an
observation and the model re-planned.

A raised exception kills the loop and teaches the model nothing. A returned
error is a teaching signal. **Tool errors should be returned, not raised.**

## Two shapes of chaining

```
Parallel — independent calls, one response:
  THINK → ['ResponseFunctionToolCall', 'ResponseFunctionToolCall']
          check_weather + get_wardrobe_items together

Sequential — dependent calls, one per iteration:
  THINK → get_wardrobe_items    (must see what is dirty…)
  THINK → wash_clothing         (…before washing it)
```

The model batches what is independent and serialises what is not. This matters
for the iteration budget: batching weather and wardrobe turned a 5-iteration
chain into 3, against a `MAX_ITERATIONS` of 5.

## Memory has a cost

Context persists across turns, so the agent remembers. Ask about the wardrobe
twice and the second answer arrives with **no tool calls at all** — it was
already in context. Cheaper and faster, and a real tradeoff: the agent can now
answer from stale memory if the world changes underneath it. Memory and
freshness pull against each other.

## Latency compounds

Every iteration is a network call, so an agent's latency is the *sum* of its
iterations, multiplied by retry policy. Measured here: a single-tool exchange
runs ~3.3s, a full chained one ~5s. A retry budget chosen without reference to
the caller's deadline will silently blow it — failure #4 below is exactly that
mistake.

## What broke

Six failures before this ran green. **None were agent-logic bugs** — which is
itself the lesson. Full write-ups with root causes in [NOTES.md](NOTES.md).

| # | Symptom | Actually was |
|---|---|---|
| 1 | `api_key must be set` | `.env` said `API_KEY`, code read `OPENAI_API_KEY` |
| 2 | `CERTIFICATE_VERIFY_FAILED` | corporate TLS proxy; Python validates against certifi, not the OS store |
| 3 | `429 insufficient_quota` | credits are **org**-level; a new key in a new project hit the same wall |
| 4 | "infinite loop", 15s timeout | retry backoff summed to 15.5s of *sleeping*; self-inflicted by the fix for #2 |
| 5 | intermittent connection errors | `truststore` calling the Windows cert store on *every* handshake |
| 6 | the fix for #5 vanished | the IDE deletes files not declared in its task manifest |

Diagnosing an agent means first asking *which layer* failed — config, transport,
billing, or the loop. How far the log gets localises it before the traceback
does:

```
[ENTERING AGENT LOOP]     ← stopped here? transport. The loop never ran.
[THINK]: [...]            ← got here? the API worked. Now suspect the loop.
```

## Running it

Requires Python 3.10+ and an OpenAI API key with credits on the organisation.

```bash
pip install -r requirements.txt
cp .env.example "Butler Agent/task/agent/.env"   # then add your key
python "Butler Agent/task/agent/agent.py"
```

Chat until you're done; `q`, `quit`, or `exit` ends the session.

The `.env` needs one of:

```
OPENAI_API_KEY=<your-openai-api-key>          # OpenAI direct
LITELLM_API_KEY=...  LITELLM_BASE_URL=...     # or an OpenAI-compatible gateway
```

A gateway must implement the **Responses API** (`/v1/responses`), which is less
widely supported than `/v1/chat/completions` — local Ollama, for one, does not
serve it.

> The deep path exists because this began as a
> [Hyperskill](https://hyperskill.org/projects/553) project, which requires
> `agent.py` to live there. The course's task descriptions and graders are not
> included in this repo — only my own code and notes.

## Built in three stages

| Stage | Shape | New idea |
|---|---|---|
| 1 — Basic chat loop | `ask → answer` | THINK only; `output` as a typed list |
| 2 — ACT introduced | `ask → [act, observe]* → answer` | the loop, tool dispatch, `call_id` |
| 3 — Multiple tools | same loop, richer tools | parameters, state, chaining, errors-as-data |
