"""Real loopback HTTP tests with actual Qdrant and an explicitly fake embedder."""
import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from uuid import uuid4

from supportlens.server import Application, MAX_BODY, make_handler
from supportlens.store import Store
from tests.test_retrieval import TopicEmbeddingDouble, ticket


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        try:
            self.start()
        except Exception:
            self.temp.cleanup()
            raise

    def start(self, drafter=None):
        self.store = Store(self.root / "knowledge", embedder=TopicEmbeddingDouble())
        self.app = Application(self.root, store=self.store, drafter=drafter)
        try:
            self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.app, 0))
        except Exception:
            self.app.close()
            raise
        self.port = self.server.server_address[1]
        self.server.RequestHandlerClass = make_handler(self.app, self.port)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.app.close()

    def tearDown(self):
        self.stop()
        self.temp.cleanup()

    def request(self, method, path, payload=None, headers=None, raw=None):
        defaults = {"Host": f"127.0.0.1:{self.port}"}
        if method == "POST":
            defaults.update({"Content-Type": "application/json", "X-App-Token": self.app.token,
                             "Origin": f"http://127.0.0.1:{self.port}"})
        defaults.update(headers or {})
        body = raw if raw is not None else json.dumps(payload).encode() if payload is not None else None
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=defaults)
            response = connection.getresponse()
            data = response.read()
            mime = response.getheader("Content-Type", "")
            return response.status, json.loads(data) if mime.startswith("application/json") else data, dict(response.getheaders())
        finally:
            connection.close()

    def wait_job(self, identifier):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status, result, _ = self.request("GET", f"/api/jobs/{identifier}")
            self.assertEqual(status, 200)
            if result["status"] in ("complete", "failed"):
                # Completion is saved just before the worker releases its semaphore.
                while self.app.active is not None and time.monotonic() < deadline:
                    time.sleep(.005)
                return result
            time.sleep(.005)
        self.fail("Local operation did not finish within five seconds.")

    def submit(self, path, payload):
        status, result, _ = self.request("POST", path, payload)
        self.assertEqual(status, 202, result)
        return self.wait_job(result["id"])

    def import_small(self):
        rows = [ticket("SL-001", "Password reset token expired."),
                ticket("SL-002", "MFA authenticator phone invalid.", resolution="Correct the phone clock and try again."),
                ticket("SL-003", "Password reset is still pending.", status="open")]
        return self.submit("/api/import", {"name": "Private test dataset", "kind": "json", "content": json.dumps(rows)})

    def search(self):
        self.import_small()
        return self.submit("/api/search", {"query": "Password reset token", "mode": "hybrid", "filters": {"status": "resolved"}})

    def test_sample_import_and_search_use_actual_qdrant(self):
        loaded = self.submit("/api/sample", {})
        self.assertEqual(loaded["status"], "complete")
        self.assertGreater(loaded["result"]["ticket_count"], 10)
        search = self.submit("/api/search", {"query": "password reset", "mode": "vector"})
        self.assertEqual(search["status"], "complete")
        self.assertTrue(search["result"]["results"])
        self.assertEqual(search["result"]["dataset"]["model"], "test-double-topic-v1")

    def test_selected_saved_evidence_draft_edit_approval_survives_restart(self):
        search = self.search()
        selected = search["result"]["results"][0]["ticket"]["id"]
        draft = self.submit("/api/draft", {"search_id": search["id"], "mode": "evidence", "ticket_ids": [selected]})
        self.assertEqual(draft["status"], "complete")
        self.assertEqual([s["id"] for s in draft["result"]["sources"]], [selected])
        self.assertFalse(draft["result"]["reviewed"])
        approved = "I reviewed this response. Please use the newest password reset link."
        status, review, _ = self.request("POST", f"/api/review/{draft['id']}", {"text": approved})
        self.assertEqual(status, 200)
        self.assertTrue(review["result"]["reviewed"])
        self.assertTrue(review["result"]["edited"])
        self.stop()
        self.start()
        status, saved, _ = self.request("GET", f"/api/jobs/{draft['id']}")
        self.assertEqual(status, 200)
        self.assertEqual(saved["result"]["approved_text"], approved)
        self.assertEqual(saved["result"]["dataset"]["id"], search["result"]["dataset"]["id"])
        self.assertTrue(self.app.store.metadata()["ticket_count"])
        self.assertEqual(self.request("POST", f"/api/review/{draft['id']}", {"text": approved})[0], 400)

    def test_client_cannot_replace_saved_evidence(self):
        search = self.search()
        status, result, _ = self.request("POST", "/api/draft", {"search_id": search["id"], "ticket_ids": ["forged"], "results": [{"resolution": "invented"}]})
        self.assertEqual(status, 400)
        self.assertIn("saved search", result["error"])
        draft = self.submit("/api/draft", {"search_id": search["id"], "results": [{"resolution": "invented"}]})
        self.assertNotIn("invented", draft["result"]["draft"])

    def test_ai_failure_is_saved_without_fake_fallback(self):
        search = self.search()
        def unavailable(*args, **kwargs):
            raise ValueError("Local AI is unavailable.")
        self.app.drafter = unavailable
        draft = self.submit("/api/draft", {"search_id": search["id"], "mode": "ai"})
        self.assertEqual(draft["status"], "failed")
        self.assertIsNone(draft["result"])
        self.assertIn("unavailable", draft["error"])
        self.stop()
        self.start()
        self.assertEqual(self.app.job(draft["id"])["status"], "failed")

    def test_search_missing_dataset_is_failed_job(self):
        job = self.submit("/api/search", {"query": "password reset", "mode": "keyword"})
        self.assertEqual(job["status"], "failed")
        self.assertIsNone(job["result"])

    def test_incomplete_saved_job_marked_failed_on_restart(self):
        identifier = str(uuid4())
        self.app.db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?)", (identifier, "search", "running", "interrupted", "", "", "2026-01-01"))
        self.app.db.commit()
        self.stop()
        self.start()
        record = self.app.job(identifier)
        self.assertEqual(record["status"], "failed")
        self.assertIn("process stopped", record["error"])

    def test_unknown_and_invalid_ids_are_not_found(self):
        for identifier in (str(uuid4()), "invalid", "../workspace.sqlite"):
            self.assertEqual(self.request("GET", "/api/jobs/" + identifier)[0], 404)

    def test_host_origin_and_token_checks(self):
        self.assertEqual(self.request("GET", "/api/config", headers={"Host": "evil.example"})[0], 403)
        for headers in ({"Origin": "https://evil.example"}, {"X-App-Token": "wrong"}, {"Host": "evil.example"}):
            self.assertEqual(self.request("POST", "/api/sample", {}, headers=headers)[0], 403)
        self.assertEqual(self.request("GET", "/api/config")[1]["token"], self.app.token)

    def test_malformed_input_and_body_limits(self):
        for body in (b"broken", b"[]", b"null"):
            self.assertEqual(self.request("POST", "/api/search", raw=body)[0], 400)
        for headers in ({"Content-Type": "text/plain"}, {"Content-Length": "0"},
                        {"Content-Length": str(MAX_BODY + 1)}, {"Transfer-Encoding": "chunked"}):
            self.assertEqual(self.request("POST", "/api/search", {}, headers=headers)[0], 400)
        for payload in ({"query": "x"}, {"query": "valid query", "limit": True},
                        {"query": "valid query", "filters": {"unknown": "x"}},
                        {"query": "valid query", "mode": "invented"}):
            self.assertEqual(self.request("POST", "/api/search", payload)[0], 400)

    def test_private_ticket_text_stays_out_of_static_page_and_state(self):
        secret = '<img src=x onerror="alert(1)"> customer-private-issue'
        rows = [ticket("SL-001", secret)]
        loaded = self.submit("/api/import", {"name": "Private data", "kind": "json", "content": json.dumps(rows)})
        self.assertEqual(loaded["status"], "complete")
        status, state, headers = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertNotIn(secret, json.dumps(state))
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        status, page, _ = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertNotIn(secret.encode(), page)

    def test_draft_from_open_ticket_has_no_approvable_response(self):
        self.import_small()
        search = self.submit("/api/search", {"query": "password reset", "mode": "keyword", "filters": {"status": "open"}})
        draft = self.submit("/api/draft", {"search_id": search["id"], "ticket_ids": ["SL-003"]})
        self.assertEqual(draft["result"]["status"], "insufficient_evidence")
        self.assertEqual(self.request("POST", f"/api/review/{draft['id']}", {"text": "Please approve this unsupported response."})[0], 400)


if __name__ == "__main__":
    unittest.main()
