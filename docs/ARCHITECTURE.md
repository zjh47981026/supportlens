# Architecture

The plain JavaScript browser UI talks to a Python standard-library HTTP server bound to loopback. The server serves local assets and exposes indexing, search, drafting, evaluation, and local review endpoints. There are no remote scripts or frontend build dependencies.

## Indexing

`store.py` validates canonical ticket fields and embeds the combined title, issue, resolution, product, and category using `nomic-embed-text`. `embeddings.py` prefixes documents with `search_document:` and queries with `search_query:`. It sends bounded batches to Ollama and rejects empty, nonfinite, zero, mismatched, or changed-model vectors.

A staged collection receives a named dense cosine vector, a named sparse keyword vector, and canonical ticket payloads. BM25 document weights use corpus statistics stored in the private dataset manifest. SQLite records dataset metadata and the active dataset pointer. The pointer changes only after successful Qdrant insertion and manifest persistence. Old saved search snapshots are independent of active-index replacement.

## Retrieval

Keyword mode queries the sparse vector. Vector mode embeds the query and queries the dense vector. Hybrid mode sends both branches as Qdrant prefetches and requests native RRF fusion. Product/category/status filters apply within each branch. Search snapshots contain the query, selected filters, dataset identity, ranked ticket payloads, timing, scores, and branch ranks.

Local Qdrant persists vectors on disk and performs exact dense search. The app serializes store access and allows one active job; it is designed for a small local corpus, not concurrent enterprise traffic.

## Drafting and review

The browser submits a saved search ID plus selected ticket IDs. The server reads its own immutable evidence snapshot, rejects forged IDs, and passes only selected resolved tickets to `drafting.py`. Evidence mode extracts recorded resolutions. AI mode asks local qwen3:4b for schema-constrained steps with source IDs and quotes.

Every quote is validated against its source's resolution and each source must belong to the selected evidence. A failed check rejects the whole AI draft. Matching quotes do not guarantee that paraphrased advice is correct for the current issue. The reviewer edits the plaintext response and approves it locally. Approval stores one reviewed version, edit status, and timestamp. Markdown export contains the reviewed response and evidence.

## Persistence and recovery

Qdrant data and dataset manifests live under the chosen runtime directory. SQLite stores up to 100 operations, their status, completed results, and approvals. Restart preserves completed snapshots and marks interrupted work failed with a retry message. There is no resumable agent graph. A separate data directory starts a fresh workspace.

## Evaluation

The lab requires the exact synthetic sample fingerprint; it does not overwrite imported data. The CLI evaluator builds an independent temporary index. Both report precision@3, recall@5, MRR@5, per-query rankings, labels, and timings. Automated tests use explicit doubles for mechanics; the committed live report records real-model retrieval.
