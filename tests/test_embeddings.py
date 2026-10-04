"""Offline transport contract tests: localhost only, bounded responses, strict vectors."""
import json
import unittest
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

from supportlens.embeddings import NoRedirect, OllamaEmbedder, validate_vectors
from supportlens.store import parse_tickets


class EmbeddingTransportTests(unittest.TestCase):
    def setUp(self):
        self.embedder = OllamaEmbedder(timeout=12)
        self.opener = MagicMock()
        self.embedder._opener = self.opener

    def response(self, value=None, raw=None):
        response = MagicMock()
        response.read.return_value = raw if raw is not None else json.dumps(value).encode()
        self.opener.open.return_value.__enter__.return_value = response
        return response

    def test_transport_disables_proxy_and_redirect_handlers(self):
        with patch('supportlens.embeddings.urllib.request.build_opener') as build:
            OllamaEmbedder()
        proxy, redirect = build.call_args.args
        self.assertIsInstance(proxy, urllib.request.ProxyHandler)
        self.assertEqual(proxy.proxies, {})
        self.assertIsInstance(redirect, NoRedirect)
        self.assertIsNone(redirect.redirect_request(None, None, 302, '', {}, 'https://example.com'))

    def test_documents_use_prefix_fixed_endpoint_and_no_truncation(self):
        response = self.response({'embeddings': [[1.0, 2.0], [2.0, 3.0]]})
        self.assertEqual(self.embedder.embed_documents(['One', 'Two']), [[1., 2.], [2., 3.]])
        request = self.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, 'http://127.0.0.1:11434/api/embed')
        self.assertEqual(request.method, 'POST')
        self.assertEqual(json.loads(request.data), {'model': 'nomic-embed-text', 'input': ['search_document: One', 'search_document: Two'], 'truncate': False})
        self.assertEqual(self.opener.open.call_args.kwargs['timeout'], 12)
        response.read.assert_called_once_with(8 * 1024 * 1024 + 1)

    def test_queries_use_distinct_prefix(self):
        self.response({'embeddings': [[.1, .2]]})
        self.assertEqual(self.embedder.embed_query('Question'), [.1, .2])
        self.assertEqual(json.loads(self.opener.open.call_args.args[0].data)['input'], ['search_query: Question'])

    def test_document_batches_do_not_exceed_24(self):
        def respond(request, timeout):
            response = MagicMock()
            count = len(json.loads(request.data)['input'])
            response.__enter__.return_value.read.return_value = json.dumps({'embeddings': [[1.] for _ in range(count)]}).encode()
            return response
        self.opener.open.side_effect = respond
        self.assertEqual(len(self.embedder.embed_documents(['text'] * 25)), 25)
        self.assertEqual([len(json.loads(call.args[0].data)['input']) for call in self.opener.open.call_args_list], [24, 1])

    def test_incomplete_or_malformed_embedding_responses_rejected(self):
        for value in ([], {}, {'embeddings': []}, {'embeddings': {}}, {'embeddings': [[1], [2]]}):
            with self.subTest(value=value):
                self.response(value)
                with self.assertRaises(RuntimeError):
                    self.embedder.embed_query('Question')
        self.response(raw=b'{invalid')
        with self.assertRaises(RuntimeError):
            self.embedder.embed_query('Question')

    def test_oversized_embedding_response_rejected(self):
        self.response(raw=b'x' * (8 * 1024 * 1024 + 1))
        with self.assertRaisesRegex(RuntimeError, 'size limit'):
            self.embedder.embed_query('Question')

    def test_connection_failure_and_redirect_are_not_remote_fallbacks(self):
        for failure in (urllib.error.URLError('offline'), urllib.error.HTTPError('http://127.0.0.1:11434/api/embed', 302, 'redirect', {}, None)):
            self.opener.open.side_effect = failure
            with self.assertRaisesRegex(RuntimeError, 'Local embeddings unavailable'):
                self.embedder.embed_query('Question')
            self.assertEqual(self.opener.open.call_args.args[0].full_url, 'http://127.0.0.1:11434/api/embed')

    def test_identity_uses_installed_manifest_digest(self):
        response = self.response({'models': [{'name': 'other:latest', 'digest': 'wrong'}, {'name': 'nomic-embed-text:latest', 'digest': 'sha256-correct'}]})
        self.assertEqual(self.embedder.identity(), {'model': 'nomic-embed-text', 'digest': 'sha256-correct'})
        request = self.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, 'http://127.0.0.1:11434/api/tags')
        self.assertEqual(request.method, 'GET')
        response.read.assert_called_once_with(1024 * 1024 + 1)

    def test_missing_models_or_digest_rejected(self):
        for value in ([], {}, {'models': None}, {'models': []}, {'models': ['bad']}, {'models': [{'name': 'nomic-embed-text:latest'}]}, {'models': [{'name': 'nomic-embed-text:latest', 'digest': 42}]}):
            with self.subTest(value=value):
                self.response(value)
                with self.assertRaises(RuntimeError):
                    self.embedder.identity()

    def test_identity_response_bounded_and_failure_explicit(self):
        self.response(raw=b'x' * (1024 * 1024 + 1))
        with self.assertRaisesRegex(RuntimeError, 'size limit'):
            self.embedder.identity()
        self.opener.open.side_effect = urllib.error.URLError('offline')
        with self.assertRaisesRegex(RuntimeError, 'identity could not be verified'):
            self.embedder.identity()

    def test_other_model_disallowed(self):
        with self.assertRaises(ValueError):
            OllamaEmbedder('remote-model')


class VectorAndIdentifierTests(unittest.TestCase):
    def test_invalid_vectors_rejected(self):
        for vectors, count, dimension in (([[1]], 2, None), ([[]], 1, None), ([[1, 2], [1]], 2, None), ([[1]], 1, 2), ([[float('inf')]], 1, None), ([[False]], 1, None), ([[0.0]], 1, None), ([['1']], 1, None)):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                validate_vectors(vectors, count, dimension)
        self.assertEqual(validate_vectors([[1., 2.], [3., 4.]], 2), 2)

    def test_import_identifiers_match_draft_source_contract(self):
        row = {'id': 'Ticket-1.test_2', 'title': 'Title', 'issue': 'Issue', 'resolution': 'Resolution', 'product': 'cloud', 'category': 'sync', 'status': 'resolved'}
        self.assertEqual(parse_tickets(json.dumps([row]))[0]['id'], row['id'])
        for identifier in ('contains spaces', '[T001]', '/path', 'x' * 81, '中文'):
            with self.subTest(identifier=identifier), self.assertRaisesRegex(ValueError, 'Ticket IDs'):
                parse_tickets(json.dumps([dict(row, id=identifier)]))


if __name__ == '__main__':
    unittest.main()
