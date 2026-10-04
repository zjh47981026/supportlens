import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from supportlens.drafting import MAX_RESPONSE_BYTES, draft_response, _call_model


RESOLUTION = "Open the password reset link in a new private browser window. Then sign in again."


def ticket(source_id="SL-001", status="resolved", resolution=RESOLUTION):
    return {"ticket": {"id": source_id, "title": "Password reset", "status": status, "resolution": resolution}, "score": .8}


def model_output(**step_changes):
    return {"intro": "Thanks for contacting support.", "steps": [{
        "text": "Try the password reset link in a private window.",
        "source_id": "SL-001", "quote": RESOLUTION,
        **step_changes,
    }]}


class DraftTests(unittest.TestCase):
    def test_evidence_copies_resolved_resolution_without_model(self):
        with patch("supportlens.drafting._call_model") as call:
            result = draft_response("Login issue", [ticket(), ticket("SL-002", "open")])
        call.assert_not_called()
        self.assertEqual(result["status"], "draft")
        self.assertEqual(result["steps"][0]["text"], RESOLUTION)
        self.assertIn("[SL-001]", result["draft"])
        self.assertIn("no generative AI", result["notice"])

    def test_no_resolved_evidence_returns_no_draft(self):
        for results in ([], [ticket(status="open")], [ticket(resolution="")]):
            with self.subTest(results=results), patch("supportlens.drafting._call_model") as call:
                result = draft_response("Login issue", results, "ai")
                self.assertEqual(result["status"], "insufficient_evidence")
                self.assertEqual(result["draft"], "")
                call.assert_not_called()

    def test_bounds_and_deduplicates_source_context(self):
        results = [ticket(), ticket()] + [ticket(f"SL-{i:03}", resolution="x" * 3000) for i in range(2, 9)]
        result = draft_response("Login issue", results)
        self.assertEqual(len(result["steps"]), 5)
        self.assertEqual(len(result["steps"][1]["text"]), 2000)

    def test_flat_tickets_are_supported(self):
        result = draft_response("Login issue", [ticket()["ticket"]])
        self.assertEqual(result["sources"][0]["id"], "SL-001")

    def test_ai_output_verified_and_generated_intro_not_rendered(self):
        output = model_output()
        output["intro"] = "Your account is guaranteed to be fixed."
        with patch("supportlens.drafting._call_model", return_value=output):
            result = draft_response("Login issue", [ticket()], "ai")
        self.assertEqual(result["status"], "draft")
        self.assertNotIn("guaranteed", result["draft"])
        self.assertEqual(result["sources"][0]["quote"], RESOLUTION)
        self.assertIn("does not prove", result["notice"])

    def test_whitespace_normalization_matches_quote(self):
        with patch("supportlens.drafting._call_model", return_value=model_output(quote=RESOLUTION.replace(" ", "\n"))):
            self.assertEqual(draft_response("Login issue", [ticket()], "ai")["status"], "draft")

    def test_unretrieved_or_unresolved_source_rejected(self):
        for source_id in ("invented", "SL-002"):
            with self.subTest(source_id=source_id), patch("supportlens.drafting._call_model", return_value=model_output(source_id=source_id)):
                with self.assertRaisesRegex(ValueError, "outside"):
                    draft_response("Login issue", [ticket(), ticket("SL-002", "open")], "ai")

    def test_fabricated_quote_rejects_all_steps(self):
        output = model_output()
        output["steps"].append({"text": "Disable account security.", "source_id": "SL-001", "quote": "Disable all account security checks."})
        with patch("supportlens.drafting._call_model", return_value=output):
            with self.assertRaisesRegex(ValueError, "could not be verified"):
                draft_response("Login issue", [ticket()], "ai")

    def test_invalid_model_output_rejected(self):
        bad = [None, {}, {"intro": "", "steps": []}, model_output(text="x" * 501),
               model_output(quote="short"), model_output(quote=" " * 600 + RESOLUTION),
               model_output(text="Reset it [SL-999]"), model_output(extra="ignored")]
        for output in bad:
            with self.subTest(output=output), patch("supportlens.drafting._call_model", return_value=output):
                with self.assertRaises(ValueError):
                    draft_response("Login issue", [ticket()], "ai")

    def test_invalid_inputs_rejected(self):
        for query, results, mode, model in [
            ("", [], "evidence", "qwen3:4b"), ("x" * 4001, [], "evidence", "qwen3:4b"),
            ("query", [], "unknown", "qwen3:4b"), ("query", [], "ai", "../bad\n"),
            ("query", [ticket()] * 101, "ai", "qwen3:4b"),
            ("query", [ticket("<script>")], "evidence", "qwen3:4b"),
        ]:
            with self.subTest(mode=mode, model=model):
                with self.assertRaises(ValueError):
                    draft_response(query, results, mode, model)

    def test_model_transport_has_fixed_endpoint_and_bounds(self):
        content = json.dumps({"message": {"content": json.dumps(model_output())}}).encode()
        with patch("supportlens.drafting.urllib.request.build_opener") as build:
            build.return_value.open.return_value = io.BytesIO(content)
            output = _call_model("query", [{"id": "SL-001", "resolution": RESOLUTION}], "qwen3:4b")
            request = build.return_value.open.call_args.args[0]
            self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
            self.assertEqual(build.return_value.open.call_args.kwargs["timeout"], 45)
            self.assertEqual(build.call_args.args[0].proxies, {})
            payload = json.loads(request.data)
            self.assertFalse(payload["think"])
            self.assertEqual(payload["options"]["temperature"], 0)
            self.assertIsInstance(payload["format"], dict)
            self.assertEqual(output, model_output())

    def test_model_transport_rejects_oversize_and_bad_json(self):
        for content in (b"x" * (MAX_RESPONSE_BYTES + 1), b"not json", b'{}', b'\xff'):
            with self.subTest(size=len(content)), patch("supportlens.drafting.urllib.request.build_opener") as build:
                build.return_value.open.return_value = io.BytesIO(content)
                with self.assertRaises(ValueError):
                    _call_model("query", [], "qwen3:4b")

    def test_model_transport_error_hides_server_details(self):
        with patch("supportlens.drafting.urllib.request.build_opener") as build:
            build.return_value.open.side_effect = urllib.error.URLError("sensitive server details")
            with self.assertRaises(ValueError) as caught:
                _call_model("query", [], "qwen3:4b")
            self.assertNotIn("sensitive", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
