# Workshop eval dogfood: Banking77 and HealthBench

These two containers require provider credentials because they execute real policy
and scoring calls. Never commit keys, paste them into Workshop chat, put them in a
URL, or save them in a run artifact.

## Credentials

Set credentials in the terminal that will own each container process:

```bash
export OPENROUTER_API_KEY='...'
export OPENAI_API_KEY='...'
```

Validate without printing the key or response body:

```bash
curl -sS -o /dev/null -w 'OpenRouter HTTP %{http_code}\n' \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/models

curl -sS -o /dev/null -w 'OpenAI HTTP %{http_code}\n' \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

Both should return HTTP 200. The entrypoints fail closed with exit code 78 when
their required credential is absent.

## Start the containers

From the repository root:

```bash
cookbooks/optimizers/gepa/banking77_container/run_container.sh --port 8765
```

Entrypoint: `cookbooks/optimizers/gepa/banking77_container/run_container.sh`

HealthBench uses the canonical OpenAI physician-rubric scorer:

```bash
cookbooks/optimizers/gepa/healthbench_groq/run_container.sh --port 8114
```

Entrypoint: `cookbooks/optimizers/gepa/healthbench_groq/run_container.sh`

Until the HealthBench cookbook dependency is advanced past its frozen Containers
revision, dogfood the local concurrency fix explicitly:

```bash
PYTHONPATH=/absolute/path/to/containers/src \
  cookbooks/optimizers/gepa/healthbench_groq/run_container.sh --port 8114
```

Confirm readiness:

```bash
curl -fsS http://127.0.0.1:8765/info >/dev/null
curl -fsS http://127.0.0.1:8114/info >/dev/null
```

## Run from packaged Workshop

Use GPT-5.6 Luna with XHigh reasoning in separate chats. Register and probe the
explicit local URL, then start the corresponding packaged recipe:

- `eval.banking77.baseline.v1`: 10 train examples, concurrency 10.
- `eval.healthbench.smoke.v1`: 2 train + 2 heldout, concurrency 2, $0.50 ceiling.

For each run, require a chat-scoped `experiment.overview.v1`, natural terminal
completion, retained rewards and Trace V5 evidence, and explicit usage. These are
baseline evals; they do not establish uplift.

Release acceptance requires every fixed-cardinality rollout to complete, the
visual and terminal record to survive restart, and policy/grader usage to remain
separate for HealthBench.
