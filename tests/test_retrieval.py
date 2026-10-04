"""Offline mechanics tests; the embedder below is a test double, not a model."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from supportlens.evaluation import evaluate, retrieval_metrics
from supportlens.store import Store, parse_tickets, validate_tickets


class TopicEmbeddingDouble:
    model = "test-double-topic-v1"
    topics = (('reset', 'password', 'token'), ('mfa', 'authenticator', 'phone'),
              ('sso', 'identity', 'callback'), ('quota', 'storage', 'trash'),
              ('network', 'proxy', 'sync'), ('billing', 'payment', 'card'),
              ('desktop', 'application', 'window'))

    def embed_query(self, text):
        text = text.lower()
        return [float(sum(text.count(word) for word in group)) for group in self.topics] + [0.1]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


def ticket(identifier, issue, *, product='account', category='authentication', status='resolved', resolution='Use the newest password reset link.'):
    return {'id': identifier, 'title': identifier, 'issue': issue, 'resolution': resolution if status == 'resolved' else '',
            'product': product, 'category': category, 'status': status}


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.directory.name), embedder=TopicEmbeddingDouble())
        self.rows = [ticket('reset', 'Password reset token has expired.'),
                     ticket('mfa', 'MFA authenticator phone code is invalid.', resolution='Correct the phone clock.'),
                     ticket('cloud', 'Storage quota exhausted.', product='cloud', category='storage', resolution='Empty cloud trash.'),
                     ticket('open', 'Password reset delivery is pending.', status='open')]
        self.store.import_tickets('Test fixture', self.rows)

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_keyword_exact_issue(self):
        found = self.store.search('authenticator', mode='keyword')
        self.assertEqual(found['results'][0]['ticket']['id'], 'mfa')
        self.assertIsNone(found['results'][0]['vector_rank'])

    def test_keyword_unknown_words_returns_no_results(self):
        self.assertEqual(self.store.search('unicorn zeppelin', mode='keyword')['results'], [])

    def test_vector_uses_query_embedding(self):
        found = self.store.search('authenticator phone', mode='vector')
        self.assertEqual(found['results'][0]['ticket']['id'], 'mfa')
        self.assertEqual(found['results'][0]['vector_rank'], 1)

    def test_product_category_and_status_filters_in_every_mode(self):
        for mode in ('keyword', 'vector', 'hybrid'):
            with self.subTest(mode=mode):
                found = self.store.search('storage quota', mode=mode, filters={'product': 'cloud', 'category': 'storage', 'status': 'resolved'})
                self.assertEqual([r['ticket']['id'] for r in found['results']], ['cloud'])
                self.assertEqual(self.store.search('storage', mode=mode, filters={'product': 'unknown'})['results'], [])

    def test_open_status_is_not_silently_excluded(self):
        for mode in ('keyword', 'vector', 'hybrid'):
            found = self.store.search('password reset', mode=mode, filters={'status': 'open'})
            self.assertEqual([r['ticket']['id'] for r in found['results']], ['open'])

    def test_query_does_not_seed_or_change_dataset(self):
        before = self.store.metadata()
        self.store.search('password reset')
        self.assertEqual(before, self.store.metadata())

    def test_duplicate_ids_rejected_before_embedding(self):
        before = self.store.metadata()
        with patch.object(self.store.embedder, 'embed_documents') as embed:
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                self.store.import_tickets('Duplicate', [self.rows[0], self.rows[0]])
            embed.assert_not_called()
        self.assertEqual(before, self.store.metadata())

    def test_failed_embeddings_leave_active_dataset_unchanged(self):
        before = self.store.metadata()
        with patch.object(self.store.embedder, 'embed_documents', side_effect=RuntimeError('Test failure')):
            with self.assertRaises(RuntimeError):
                self.store.import_tickets('Broken import', self.rows)
        self.assertEqual(before, self.store.metadata())
        self.assertEqual(self.store.search('authenticator', mode='keyword')['results'][0]['ticket']['id'], 'mfa')

    def test_failed_upsert_removes_staging_collection(self):
        before = self.store.metadata()
        collections = {c.name for c in self.store.client.get_collections().collections}
        with patch.object(self.store.client, 'upsert', side_effect=RuntimeError('Test interrupted write')):
            with self.assertRaises(RuntimeError):
                self.store.import_tickets('Broken import', self.rows)
        self.assertEqual(before, self.store.metadata())
        self.assertEqual(collections, {c.name for c in self.store.client.get_collections().collections})

    def test_persistence_reopens_vector_index(self):
        before = self.store.search('authenticator phone', mode='vector')
        self.store.close()
        self.store = Store(Path(self.directory.name), embedder=TopicEmbeddingDouble())
        after = self.store.search('authenticator phone', mode='vector')
        self.assertEqual([r['ticket']['id'] for r in before['results']], [r['ticket']['id'] for r in after['results']])
        self.assertEqual(self.store.metadata()['ticket_count'], 4)

    def test_model_mismatch_requires_reindex(self):
        self.store.embedder.model = 'different-model'
        for mode in ('vector', 'hybrid'):
            with self.assertRaisesRegex(ValueError, 'model differs'):
                self.store.search('reset', mode=mode)
        self.assertTrue(self.store.search('reset', mode='keyword')['results'])

    def test_same_model_name_changed_digest_requires_reindex(self):
        digest = ['weights-v1']
        self.store.embedder.identity = lambda: {'model': self.store.embedder.model, 'digest': digest[0]}
        self.store.import_tickets('Versioned vectors', self.rows)
        digest[0] = 'weights-v2'
        with self.assertRaises(ValueError):
            self.store.search('reset', mode='vector')
        self.assertTrue(self.store.search('reset', mode='keyword')['results'])

    def test_model_change_during_import_cannot_activate_vectors(self):
        digest = ['weights-v1']
        self.store.embedder.identity = lambda: {'model': self.store.embedder.model, 'digest': digest[0]}
        before = self.store.metadata()
        embed = self.store.embedder.embed_documents
        def changed(texts):
            result = embed(texts)
            digest[0] = 'weights-v2'
            return result
        with patch.object(self.store.embedder, 'embed_documents', side_effect=changed):
            with self.assertRaises(ValueError):
                self.store.import_tickets('Changed weights', self.rows)
        self.assertEqual(before, self.store.metadata())

    def test_model_change_during_query_rejects_mixed_versions(self):
        digest = ['weights-v1']
        self.store.embedder.identity = lambda: {'model': self.store.embedder.model, 'digest': digest[0]}
        self.store.import_tickets('Versioned vectors', self.rows)
        embed = self.store.embedder.embed_query
        def changed(text):
            result = embed(text)
            digest[0] = 'weights-v2'
            return result
        with patch.object(self.store.embedder, 'embed_query', side_effect=changed):
            with self.assertRaises(ValueError):
                self.store.search('reset', mode='hybrid')

    def test_dimension_mismatch_requires_reindex(self):
        with patch.object(self.store.embedder, 'embed_query', return_value=[1.0, 0.0]):
            with self.assertRaises(ValueError):
                self.store.search('reset', mode='vector')

    def test_nonfinite_embeddings_cannot_activate_import(self):
        before = self.store.metadata()
        with patch.object(self.store.embedder, 'embed_documents', return_value=[[float('nan')]] * 4):
            with self.assertRaises(ValueError):
                self.store.import_tickets('Bad numbers', self.rows)
        self.assertEqual(before, self.store.metadata())

    def test_previous_dataset_can_be_restored(self):
        first = self.store.metadata()['dataset']['id']
        self.store.import_tickets('Replacement', [self.rows[2]])
        self.assertEqual(self.store.metadata()['ticket_count'], 1)
        self.store.use_dataset(first)
        self.assertEqual(self.store.metadata()['ticket_count'], 4)

    def test_input_validation(self):
        for kwargs in ({'mode': 'wrong'}, {'limit': True}, {'limit': 0}, {'filters': {'arbitrary': 'value'}}, {'filters': []}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.store.search('reset', **kwargs)
        with self.assertRaises(ValueError):
            self.store.search('x')

    def test_hybrid_fuses_rankings_instead_of_adding_raw_scores(self):
        class ControlledEmbeddingDouble:
            model = 'test-double-controlled-v1'
            def embed_documents(self, texts):
                return [[0.7, 0.7], [1.0, 0.0], [0.0, 1.0]]
            def embed_query(self, text):
                return [0.0, 1.0]
        self.store.embedder = ControlledEmbeddingDouble()
        self.store.import_tickets('RRF test', [
            ticket('balanced', 'reset reset reset reset', resolution='Resolve issue.'),
            ticket('keyword', 'reset reset', resolution='Resolve issue.'),
            ticket('vector', 'reset', resolution='Resolve issue.')])
        self.assertEqual(self.store.search('reset', mode='keyword')['results'][0]['ticket']['id'], 'balanced')
        self.assertEqual(self.store.search('reset', mode='vector')['results'][0]['ticket']['id'], 'vector')
        hybrid = self.store.search('reset', mode='hybrid')['results']
        self.assertEqual(hybrid[0]['ticket']['id'], 'balanced')
        self.assertEqual((hybrid[0]['keyword_rank'], hybrid[0]['vector_rank']), (1, 2))
        self.assertGreater(hybrid[0]['score'], hybrid[1]['score'])


class FixtureAndMetricTests(unittest.TestCase):
    def test_synthetic_corpus_and_labels_consistent(self):
        root = Path(__file__).resolve().parents[1] / 'supportlens' / 'fixtures'
        corpus = validate_tickets(json.loads((root / 'tickets.json').read_text()))
        cases = json.loads((root / 'evaluation.json').read_text())
        self.assertEqual(len(corpus), 40)
        self.assertEqual(sum(t['status'] == 'open' for t in corpus), 6)
        ids = {t['id'] for t in corpus}
        for case in cases:
            self.assertTrue(set(case['relevant_ids']) <= ids)
            for relevant_id in case['relevant_ids']:
                row = next(t for t in corpus if t['id'] == relevant_id)
                self.assertTrue(all(row[k] == v for k, v in case['filters'].items()))

    def test_metrics_known_ordering_and_no_duplicate_credit(self):
        self.assertEqual(retrieval_metrics(['wrong', 'a', 'a', 'b'], ['a', 'b']),
                         {'precision_at_3': 2 / 3, 'recall_at_5': 1.0, 'mrr_at_5': 0.5})
        self.assertEqual(retrieval_metrics([], ['a'])['mrr_at_5'], 0)
        with self.assertRaises(ValueError):
            retrieval_metrics(['a'], [])

    def test_csv_and_json_import_validation(self):
        row = ticket('sample', 'Reset link expired.')
        self.assertEqual(parse_tickets(json.dumps([row])), [row])
        with self.assertRaises(ValueError):
            parse_tickets('id,title\nx,y', kind='csv')
        with self.assertRaises(ValueError):
            validate_tickets([dict(row, resolution='')])
        with self.assertRaises(ValueError):
            validate_tickets([dict(row, unexpected='extra')])

    def test_evaluation_does_not_seed_and_reports_query_ranks(self):
        class StoreDouble:
            def metadata(self): return {'dataset': {'name': 'test double'}}
            def search(self, query, mode, filters, limit):
                return {'results': [{'ticket': {'id': 'b'}}, {'ticket': {'id': 'a'}}], 'timing_ms': 1}
        case = {'id': 'fixture', 'query': 'Question', 'filters': {}, 'relevant_ids': ['a']}
        report = evaluate(StoreDouble(), [case])
        self.assertEqual(report['case_count'], 1)
        self.assertEqual(report['modes']['hybrid']['mean']['mrr_at_5'], 0.5)
        self.assertEqual(report['modes']['vector']['cases'][0]['ranked_ids'], ['b', 'a'])


if __name__ == '__main__':
    unittest.main()
