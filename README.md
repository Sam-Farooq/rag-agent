# rag-agent

Retrieval-augmented question answering over a regulatory corpus, built as a
LangGraph state machine that grades its own retrieval, retries when the
passages are thin, and refuses when they stay thin.

The interesting part is not the retrieval. It is the two gates.

## Why two gates

A RAG pipeline fails in two unrelated ways and one check cannot catch both.

**Retrieval can be wrong.** The query pulls passages about the right statute
and the wrong subsection. A relevance gate catches this, and the fix is to
rewrite the query and search again.

**Generation can be ungrounded.** Retrieval was fine, and the model answered
from what it already knew rather than from the passages in front of it. No
retrieval metric sees this, because retrieval worked. It needs a separate check
that reads the answer against the passages after the fact.

So the graph grades twice, with a retry budget shared between them:

```
          ┌──────────────────────────────────────────┐
          ▼                                          │
      rewrite ──► retrieve ──► grade_documents ──────┤ score < 0.6, budget left
                                    │                │
                       score ≥ 0.6  │                │
                                    ▼                │
                               generate              │
                                    │                │
                                    ▼                │
                            grade_grounding ─────────┘
                                    │
                    score ≥ 0.6     │     budget spent
                        ▼           │          ▼
                       END          │        refuse
                                    └──────────►
```

`refuse` is a real terminal state, not an error path. In a regulatory setting
an answer assembled from parametric memory is worse than no answer, because it
arrives with the same confident tone as a correct one and nothing downstream
can tell them apart.

## Retrieval

Dense search in Qdrant pulls 24 candidates, a cross-encoder re-ranks them, and
the top 5 above a score floor go to the model.

Pulling 24 to keep 5 looks wasteful and is not. A bi-encoder embeds the query
and the passage separately, so it never compares them directly and its ordering
is noisy. A cross-encoder reads both together and orders far better, but it
costs a forward pass per pair, so it cannot run over the whole collection. Wide
recall then precise ordering is the standard way to buy both.

`min_rerank_score` matters as much as `final_k`. Without a floor, a query with
no good answer still returns its five least-bad passages, the relevance grader
sees five plausible-looking chunks, and the refusal path never fires.

## MCP server

`src/rag_agent/mcp/server.py` exposes `search_regulations` and
`get_collection_info` over stdio, so Claude Desktop or another agent can query
the corpus without the HTTP API.

The graph is deliberately not exposed as a tool. A caller that wants the
self-correction loop should call `/ask` and pay for it. A caller that wants raw
passages to reason over itself should not.

```json
{
  "mcpServers": {
    "rag-agent": {
      "command": "python",
      "args": ["-m", "rag_agent.mcp.server"],
      "env": {"RAG_QDRANT_URL": "http://localhost:6333"}
    }
  }
}
```

## Evaluation

`evals/dataset.jsonl` is a held-out set. Five questions answerable from the
corpus, two that are not.

The two unanswerable ones carry most of the signal. `refusal_accuracy` is gated
at 1.0 in CI, so a prompt change that makes the model helpful enough to answer
a question about yesterday's share price fails the build. Every other metric
can regress a little and be argued about. That one cannot.

```
$ make eval
{
  "refusal_accuracy": 1.0,
  "citation_precision": 0.8,
  "content_recall": 0.8,
  "mean_grounding": 0.86,
  "p95_latency_ms": 3180.4,
  "n": 7
}
```

Scores are pushed back to Langfuse against the trace id, so retrieval quality
and grounding sit next to token cost and latency on the same span rather than
in a separate spreadsheet.

## Running it

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY
make up                       # qdrant, redis, api
python scripts/ingest.py --path ./corpus --source crr-575-2013
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"What capital ratio does CRR Article 92 require?"}' | jq
```

Langfuse keys are optional. Without them the tracer is a no-op and everything
else runs.

## Layout

```
src/rag_agent/
  graph/        state, nodes, and the conditional edges that do the work
  retrieval/    qdrant store, bge embedder, cross-encoder re-ranker
  mcp/          stdio server exposing retrieval as tools
  observability/langfuse tracing, no-op when unconfigured
  api/          FastAPI: /ask, /ingest, /health
evals/          held-out set, metrics, CI gates
```

## Decisions worth knowing about

**Answers are cached, refusals are not.** Keyed on the normalised question, one
hour TTL. Caching a refusal means a transient Qdrant outage poisons that
question until the key expires.

**The cache key uses the original question, not the rewritten one.** The
rewrite is a function of the retry count, so keying on it would miss every
first attempt.

**The query is only rewritten on retry.** On the first pass the user's wording
is usually the best available signal, and the rewrite costs a model call.

**Model weights are baked into the image.** The embedder and cross-encoder come
to about 500MB together. That is a large image in exchange for keeping cold
start off the request path.

## Tests

`pytest -q`. The routing tests assert the branch predicates directly rather
than driving the graph through a mocked model: a test that stubs the LLM and
checks the stub was called proves nothing about where the graph goes.

## Not done

- **The grader and the generator are the same model.** It marks its own work.
  A smaller separate grader (bge-reranker already loaded, or a 7B judge) would
  be both cheaper and less self-serving, but the calibration work to keep the
  0.6 threshold meaningful across two models has not been done.
- **Chunking is paragraph-aware and still wrong for tables.** Annex tables in
  the CRR come out as a wall of digits with no header. Nothing downstream
  notices because the text is technically present.
- **No hybrid retrieval here.** Dense only. BM25 would help on statute numbers
  and exact article references, which is precisely where dense embeddings are
  weakest. The RRF implementation in `hybrid-search` was meant to be lifted
  into this repo and has not been.
- `max_batch_size` is unenforced on `/ingest`, so a large upsert can hold the
  event loop. Has not mattered yet because ingest is run from a script.
