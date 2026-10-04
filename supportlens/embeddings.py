"""Local Ollama embeddings; no remote fallback or synthetic semantic vectors."""
import json
import math
import urllib.request
import urllib.error


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OllamaEmbedder:
    model = 'nomic-embed-text'

    def __init__(self, model='nomic-embed-text', timeout=90):
        if model != 'nomic-embed-text':
            raise ValueError('This build uses nomic-embed-text for consistent retrieval.')
        self.model = model
        self.timeout = timeout
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def identity(self):
        """Resolve the installed model manifest digest, so changed weights require reindexing."""
        request = urllib.request.Request('http://127.0.0.1:11434/api/tags', method='GET')
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                body = response.read(1024 * 1024 + 1)
            if len(body) > 1024 * 1024:
                raise RuntimeError('Local model list exceeds its size limit.')
            decoded = json.loads(body)
            if not isinstance(decoded, dict) or not isinstance(decoded.get('models'), list):
                raise RuntimeError('Invalid local model list.')
            models = decoded['models']
            target = self.model if ':' in self.model else self.model + ':latest'
            for model in models:
                if isinstance(model, dict) and model.get('name') == target and isinstance(model.get('digest'), str) and model['digest']:
                    return {'model': self.model, 'digest': model['digest']}
            raise RuntimeError('Pull nomic-embed-text before indexing or searching.')
        except (OSError, urllib.error.URLError, ValueError) as exc:
            raise RuntimeError('Local embedding model identity could not be verified.') from exc

    def _embed(self, texts):
        vectors = []
        for start in range(0, len(texts), 24):
            request = urllib.request.Request('http://127.0.0.1:11434/api/embed',
                data=json.dumps({'model': self.model, 'input': texts[start:start+24], 'truncate': False}).encode(),
                headers={'Content-Type': 'application/json'}, method='POST')
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    body = response.read(8 * 1024 * 1024 + 1)
                if len(body) > 8 * 1024 * 1024:
                    raise RuntimeError('Embedding response exceeds the size limit.')
                result = json.loads(body)
                if not isinstance(result, dict):
                    raise RuntimeError('Invalid local embedding response.')
                batch = result.get('embeddings', [])
                if not isinstance(batch, list) or len(batch) != len(texts[start:start+24]):
                    raise RuntimeError('Embedding model returned an incomplete batch.')
                vectors.extend(batch)
            except (OSError, urllib.error.URLError, ValueError) as exc:
                raise RuntimeError('Local embeddings unavailable. Start Ollama and pull nomic-embed-text.') from exc
        return vectors

    def embed_documents(self, texts):
        return self._embed(['search_document: ' + text for text in texts])

    def embed_query(self, text):
        return self._embed(['search_query: ' + text])[0]


def validate_vectors(vectors, count, dimension=None):
    if not isinstance(vectors, list) or len(vectors) != count:
        raise ValueError('Embedding count does not match the ticket count.')
    size = dimension
    for vector in vectors:
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError('Empty embedding vector.')
        size = size or len(vector)
        if len(vector) != size or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vector):
            raise ValueError('Embeddings must have consistent dimensions and finite values.')
        if not any(v != 0 for v in vector):
            raise ValueError('Embedding vectors cannot be zero.')
    return size
