# Free-model aggregator

One server, two client APIs, ten free-tier providers. It serves the OpenAI
Chat Completions API (`POST /v1/chat/completions`) and the Anthropic
Messages API (`POST /v1/messages`), in JSON and in server-sent events,
routing every request to the best free model that can serve it.

## How it works

The trial hands the server a brief (`MAKEFUN_PROMPT_FILE`) with the provider
directory, the sealed quality table and the workload manifest. At startup
the server probes every provider's own catalog route, builds one catalog
document from what the listings said, registers its port and that catalog
with the env (`POST $MAKEFUN_ENV_URL/register`), then answers the client's
twenty requests one at a time:

- **Routing by rating.** Candidates are the free models meeting the
  request's requirements (tools, vision, JSON mode, minimum context),
  ranked by the quality table's `rating_milli`. A soft per-provider cap
  spreads primaries across providers so no single free tier is drained;
  an OpenAI client asking for JSON mode is never routed to an Anthropic-wire
  provider, whose wire cannot echo JSON mode.
- **Failover.** A 402, 429 or 5xx from a provider is retried on the
  next-best candidate, other providers first, then other models of the same
  provider. Exactly one successful provider call answers each request.
- **All three wires.** Providers speak the OpenAI, Anthropic and Gemini
  wires; requests and answers are translated both ways, preserving exactly
  what the client compares: system text, user text, tool names, max_tokens,
  stop sequences, image bytes and the JSON-mode flag, with the provider's
  text, usage and stop reason carried over verbatim.
- **Deterministic.** Every choice is a function of the brief, the requests
  and the providers' answers so far: no wall clock, no unseeded randomness,
  so a replay reproduces the same run.

When the env says the workload is done (`GET $MAKEFUN_ENV_URL/work`), the
server exits 0.

## Run

```sh
python3 main.py --list   # print the catalog and exit 0
python3 main.py          # serve (needs MAKEFUN_ENV_URL + MAKEFUN_PROMPT_FILE)
```

Pure Python 3 standard library: no third-party packages, nothing to install.

## Local test

The kit's env and scorer run the practice trials without docker:

```sh
# env on an ephemeral port, then the entry against it, then SIGTERM the env
MAKEFUN_ENV_DATA_DIR=<kit>/practice/trial-0/input/env \
MAKEFUN_STATE_DIR=/tmp/fma-state MAKEFUN_ENV_PORT=0 \
  node <kit>/program/env/server.mjs
MAKEFUN_ENV_URL=http://127.0.0.1:<port> \
MAKEFUN_PROMPT_FILE=<kit>/practice/trial-0/input/prompt.txt \
  python3 main.py
# score the export the env writes on SIGTERM:
MAKEFUN_OUTPUT_DIR=/tmp/fma-state/export \
MAKEFUN_REFERENCE_DIR=<kit>/practice/trial-0/reference \
  node <kit>/program/scorer/score.mjs
```

Practice results on this machine: trial-0, trial-1 and trial-2 each score
**99** (quality 99, served 100, wire 100, failover 100, catalog 100),
20/20 requests valid, 233/233 response checks passed.
