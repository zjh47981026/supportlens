# Retrieval evaluation

The sample corpus contains 40 synthetic English tickets across account, cloud,
billing, and desktop products. Six tickets are unresolved. No real customer
records or personal data are included.

`supportlens/fixtures/evaluation.json` contains 12 development questions and
human-authored relevant ticket IDs. Labels distinguish expired reset links from
MFA and SSO failures, exhausted storage from network problems, and settled
duplicate payments from pending authorizations. Three cases also exercise product,
category, or status filters. Labels reflect useful resolutions (or the specifically
requested open issue), rather than shared words alone. They were written before
running retrieval, not derived from model rankings.

## Run a real embedding evaluation

Start Ollama, install the configured embedding model (`nomic-embed-text` by
default), then run from the repository root:

```sh
python -m supportlens.evaluation --output evaluation-results.json
```

This creates a separate temporary Qdrant index, imports the sample corpus once,
and measures keyword, vector, and hybrid retrieval against the same labels and
filters. It does not overwrite the application's dataset. Model download and
runtime requirements are described in the README. A failed model connection is
reported as a failure; there is no substitution with mock embeddings.

The report includes every query, relevance label, returned ordering, and mean:

- Precision@3: relevant items among the first three, divided by three. Fewer than
  three results still use the denominator three.
- Recall@5: distinct relevant items found in the first five, divided by the
  number of labeled relevant items.
- MRR@5: reciprocal rank of the first relevant result within five, or zero.
- Retrieval time: elapsed query time. It includes query embedding in vector and
  hybrid modes but excludes initial document embedding/import.

Low precision can be expected when a question has only one labeled answer.
Inspect per-query ranks and recall alongside the averages. The small corpus and
12 questions are development fixtures, not a held-out benchmark or evidence that
one retrieval method always outperforms another. They do not evaluate generated
response correctness, source interpretation, multilingual retrieval, large-index
latency, or real customer data.

## Automated tests

Unit tests inject a clearly labeled deterministic embedding test double. They
check indexing, persistence, filters, ranking mechanics, import atomicity, and
metric calculations without Ollama or a network connection. Their passing scores
are not presented as real embedding-model quality measurements. The live report
is the appropriate evidence for that claim.

## Observed development run

A local run with real `nomic-embed-text` embeddings on October 4, 2026 produced:

| Retrieval | Precision@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|
| Keyword | 0.444 | 0.958 | 1.000 |
| Vector | 0.472 | 1.000 | 1.000 |
| Hybrid | 0.472 | 1.000 | 1.000 |

All three methods put a labeled relevant ticket first on these 12 questions.
Keyword search missed one secondary relevant ticket in the top five; vector and
hybrid found all labeled relevant tickets there. This small development set does
not establish superiority in production, and the labels and examples were
available during development. Rerun the command above for your model/version and
inspect its per-query JSON before comparing changes.
