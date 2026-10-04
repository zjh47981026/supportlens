"""Persistent Qdrant retrieval with transactional dataset activation."""
from collections import Counter
from pathlib import Path
import csv
import io
import hashlib
import json
import math
import re
import sqlite3
import threading
import time
import uuid
from qdrant_client import QdrantClient, models
from .embeddings import OllamaEmbedder, validate_vectors

FIELDS = ('id', 'title', 'issue', 'resolution', 'product', 'category', 'status')
STOP = frozenset('a an the to of for in on and or is it my i with after when can cannot'.split())


def tokenize(text):
    return [word for word in re.findall(r'[a-z0-9]+(?:[-_][a-z0-9]+)*', text.lower()) if word not in STOP]


def ticket_text(ticket):
    return '\n'.join(ticket[field] for field in ('title', 'issue', 'resolution', 'product', 'category'))


def validate_tickets(tickets):
    if not isinstance(tickets, list) or not 1 <= len(tickets) <= 500:
        raise ValueError('Import between 1 and 500 tickets.')
    normalized, seen = [], set()
    for row in tickets:
        if not isinstance(row, dict) or set(row) != set(FIELDS):
            raise ValueError('Tickets must contain exactly: ' + ', '.join(FIELDS))
        clean = {}
        for field in FIELDS:
            value = row[field]
            if not isinstance(value, str):
                raise ValueError('Ticket fields must be text.')
            value = value.strip()
            if len(value) > (5000 if field in ('issue', 'resolution') else 200):
                raise ValueError('Ticket field exceeds its length limit.')
            if not value and field != 'resolution':
                raise ValueError('Ticket fields cannot be empty.')
            clean[field] = value
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', clean['id']):
            raise ValueError('Ticket IDs must be 1–80 characters: letters, numbers, dots, underscores, or hyphens.')
        if clean['status'] not in ('resolved', 'open'):
            raise ValueError('Status must be resolved or open.')
        if clean['status'] == 'resolved' and not clean['resolution']:
            raise ValueError('Resolved tickets need a resolution.')
        if clean['id'] in seen:
            raise ValueError('Duplicate ticket IDs are not allowed.')
        seen.add(clean['id'])
        normalized.append(clean)
    if len(json.dumps(normalized).encode()) > 2 * 1024 * 1024:
        raise ValueError('Ticket import exceeds 2 MB.')
    return normalized


def parse_tickets(content, kind='json'):
    if not isinstance(content, str) or len(content.encode('utf8')) > 2 * 1024 * 1024:
        raise ValueError('Ticket import must be text smaller than 2 MB.')
    try:
        if kind == 'json':
            tickets = json.loads(content)
        elif kind == 'csv':
            reader = csv.DictReader(io.StringIO(content.lstrip('\ufeff')))
            if reader.fieldnames is None or len(reader.fieldnames) != len(FIELDS) or set(reader.fieldnames) != set(FIELDS):
                raise ValueError('CSV needs the exact ticket column headers.')
            tickets = []
            for row in reader:
                tickets.append(row)
                if len(tickets) > 500:
                    raise ValueError('Import at most 500 tickets.')
        else:
            raise ValueError('Choose JSON or CSV import.')
        return validate_tickets(tickets)
    except (json.JSONDecodeError, csv.Error) as exc:
        raise ValueError('Invalid ticket file.') from exc


class Store:
    def __init__(self, root, embedder=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder if embedder is not None else OllamaEmbedder()
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / 'datasets.sqlite', check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, metadata TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.db.commit()
        self.client = QdrantClient(path=str(self.root / 'qdrant'))
        # An interrupted import cannot become active. Remove only unregistered staging collections.
        registered = {row[0] for row in self.db.execute('SELECT id FROM datasets')}
        for collection in self.client.get_collections().collections:
            if collection.name.startswith('tickets_') and collection.name not in registered:
                self.client.delete_collection(collection.name)

    def _active(self):
        row = self.db.execute("SELECT value FROM settings WHERE key='active'").fetchone()
        if row is None:
            return None
        saved = self.db.execute('SELECT metadata FROM datasets WHERE id=?', (row[0],)).fetchone()
        return json.loads(saved[0]) if saved else None

    def import_tickets(self, name, tickets):
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise ValueError('Dataset name must be 1–80 characters.')
        tickets = validate_tickets(tickets)
        with self.lock:
            identity = self.embedder.identity() if hasattr(self.embedder, 'identity') else {'model': self.embedder.model}
            vectors = self.embedder.embed_documents([ticket_text(t) for t in tickets])
            if hasattr(self.embedder, 'identity') and self.embedder.identity() != identity:
                raise ValueError('Embedding model changed during indexing. Try again.')
            dimension = validate_vectors(vectors, len(tickets))
            documents = [Counter(tokenize(ticket_text(t))) for t in tickets]
            vocabulary = {term: index for index, term in enumerate(sorted({term for doc in documents for term in doc}))}
            df = Counter(term for doc in documents for term in doc)
            idf = {term: math.log(1 + (len(tickets) - count + .5) / (count + .5)) for term, count in df.items()}
            average_length = sum(sum(doc.values()) for doc in documents) / len(documents) or 1
            collection = 'tickets_' + uuid.uuid4().hex
            metadata = {'id': collection, 'name': name.strip(), 'ticket_count': len(tickets),
                'model': self.embedder.model, 'embedding_identity': identity, 'dimension': dimension, 'created_at': time.time(),
                'fingerprint': hashlib.sha256(json.dumps(tickets, sort_keys=True).encode()).hexdigest(),
                'vocabulary': vocabulary, 'idf': idf, 'average_length': average_length,
                'filters': {field: sorted({t[field] for t in tickets}) for field in ('product', 'category', 'status')}}
            try:
                self.client.create_collection(collection, vectors_config={'dense': models.VectorParams(size=dimension, distance=models.Distance.COSINE)}, sparse_vectors_config={'keyword': models.SparseVectorParams()})
                points = []
                for index, (ticket, vector, doc) in enumerate(zip(tickets, vectors, documents)):
                    length = sum(doc.values())
                    terms = sorted(doc, key=lambda term: vocabulary[term])
                    weights = [idf[term] * doc[term] * 2.2 / (doc[term] + 1.2 * (.25 + .75 * length / average_length)) for term in terms]
                    points.append(models.PointStruct(id=index, vector={'dense': vector, 'keyword': models.SparseVector(indices=[vocabulary[t] for t in terms], values=weights)}, payload=ticket))
                self.client.upsert(collection, points=points, wait=True)
                # Durable immutable manifest is useful for auditing the embedding contract.
                manifest = self.root / (collection + '.json')
                manifest.write_text(json.dumps(metadata, indent=2), encoding='utf8')
                with self.db:
                    self.db.execute('INSERT INTO datasets VALUES (?,?)', (collection, json.dumps(metadata)))
                    self.db.execute("INSERT OR REPLACE INTO settings VALUES ('active',?)", (collection,))
            except Exception:
                if self.client.collection_exists(collection):
                    self.client.delete_collection(collection)
                (self.root / (collection + '.json')).unlink(missing_ok=True)
                raise
            # Keep the previous dataset as a rollback base; prune older completed imports.
            stale = self.db.execute('SELECT id FROM datasets ORDER BY rowid DESC LIMIT -1 OFFSET 2').fetchall()
            for (old_id,) in stale:
                with self.db:
                    self.db.execute('DELETE FROM datasets WHERE id=?', (old_id,))
                self.client.delete_collection(old_id)
                (self.root / (old_id + '.json')).unlink(missing_ok=True)
            return self.metadata()

    def load_sample(self):
        source = Path(__file__).parent / 'fixtures' / 'tickets.json'
        return self.import_tickets('Synthetic support tickets', json.loads(source.read_text(encoding='utf8')))

    def use_dataset(self, dataset_id):
        with self.lock:
            if not self.db.execute('SELECT 1 FROM datasets WHERE id=?', (dataset_id,)).fetchone():
                raise ValueError('Dataset does not exist.')
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO settings VALUES ('active',?)", (dataset_id,))
            return self.metadata()

    def delete_dataset(self, dataset_id):
        with self.lock:
            if not self.db.execute('SELECT 1 FROM datasets WHERE id=?', (dataset_id,)).fetchone():
                raise ValueError('Dataset does not exist.')
            with self.db:
                self.db.execute('DELETE FROM datasets WHERE id=?', (dataset_id,))
                self.db.execute("DELETE FROM settings WHERE key='active' AND value=?", (dataset_id,))
            self.client.delete_collection(dataset_id)
            (self.root / (dataset_id + '.json')).unlink(missing_ok=True)
            return self.metadata()

    def metadata(self):
        with self.lock:
            active = self._active()
            datasets = [json.loads(row[0]) for row in self.db.execute('SELECT metadata FROM datasets ORDER BY rowid DESC')]
            public = lambda item: {k: v for k, v in item.items() if k not in ('vocabulary', 'idf', 'average_length')}
            return {'dataset': public(active) if active else None, 'datasets': [public(d) for d in datasets],
                'filters': active['filters'] if active else {'product': [], 'category': [], 'status': []},
                'ticket_count': active['ticket_count'] if active else 0}

    def search(self, query, mode='hybrid', filters=None, limit=5):
        if not isinstance(query, str) or not 2 <= len(query.strip()) <= 2000:
            raise ValueError('Question must be 2–2000 characters.')
        if mode not in ('keyword', 'vector', 'hybrid'):
            raise ValueError('Choose keyword, vector, or hybrid search.')
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ValueError('Result limit must be between 1 and 20.')
        filters = {} if filters is None else filters
        if not isinstance(filters, dict) or set(filters) - {'product', 'category', 'status'} or any(not isinstance(v, str) or len(v) > 200 for v in filters.values()):
            raise ValueError('Unsupported search filter.')
        start = time.perf_counter()
        with self.lock:
            active = self._active()
            if not active:
                raise ValueError('Import tickets before searching.')
            conditions = [models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in filters.items() if v]
            selector = models.Filter(must=conditions) if conditions else None
            terms = sorted(set(tokenize(query)) & set(active['vocabulary']), key=lambda term: active['vocabulary'][term])
            sparse = models.SparseVector(indices=[active['vocabulary'][term] for term in terms], values=[1.0] * len(terms))
            keyword = self.client.query_points(active['id'], query=sparse, using='keyword', query_filter=selector, limit=20, with_payload=True).points if terms else []
            dense, vector = None, []
            if mode != 'keyword':
                if self.embedder.model != active['model']:
                    raise ValueError('Embedding model differs from the indexed dataset. Reindex with the current model.')
                identity = self.embedder.identity() if hasattr(self.embedder, 'identity') else {'model': self.embedder.model}
                if identity != active.get('embedding_identity', {'model': active['model']}):
                    raise ValueError('Embedding weights changed. Reindex this dataset before searching.')
                dense = self.embedder.embed_query(query.strip())
                if hasattr(self.embedder, 'identity') and self.embedder.identity() != identity:
                    raise ValueError('Embedding model changed during search. Try again.')
                validate_vectors([dense], 1, active['dimension'])
                vector = self.client.query_points(active['id'], query=dense, using='dense', query_filter=selector, limit=20, with_payload=True).points
            if mode == 'hybrid' and terms:
                points = self.client.query_points(active['id'], prefetch=[models.Prefetch(query=sparse, using='keyword', filter=selector, limit=20), models.Prefetch(query=dense, using='dense', filter=selector, limit=20)], query=models.FusionQuery(fusion=models.Fusion.RRF), query_filter=selector, limit=limit, with_payload=True).points
            else:
                points = (keyword if mode == 'keyword' else vector)[:limit]
            kr = {p.id: rank for rank, p in enumerate(keyword, 1)}
            vr = {p.id: rank for rank, p in enumerate(vector, 1)}
            return {'results': [{'ticket': p.payload, 'score': p.score, 'keyword_rank': kr.get(p.id), 'vector_rank': vr.get(p.id)} for p in points],
                'dataset': {k: v for k, v in active.items() if k not in ('vocabulary', 'idf', 'average_length')},
                'mode': mode, 'timing_ms': round((time.perf_counter() - start) * 1000, 1)}

    def close(self):
        with self.lock:
            self.client.close()
            self.db.close()
