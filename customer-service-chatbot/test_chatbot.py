"""Regression tests for the chatbot's highest-value user journeys."""
import unittest

from app import ChatRequest, _sessions, chat, clear_session


class ChatbotBehaviorTests(unittest.TestCase):
    def setUp(self):
        _sessions.clear()

    def test_order_message_returns_intent_entity_and_rankings(self):
        result = chat(ChatRequest(message="Where is my order #45210?", session_id="orders"))

        self.assertEqual(result.intent, "order_status")
        self.assertEqual(result.entities["order_number"], "order #45210")
        self.assertGreaterEqual(len(result.alternatives), 3)
        self.assertFalse(result.needs_clarification)
        self.assertGreater(result.processing_ms, 0)

    def test_order_number_is_carried_into_follow_up(self):
        chat(ChatRequest(message="Track order #88213", session_id="follow-up"))
        result = chat(ChatRequest(message="Can I cancel it?", session_id="follow-up"))

        self.assertEqual(result.intent, "cancel_order")
        self.assertEqual(result.entities["order_number"], "order #88213")

    def test_unrecognized_message_requests_clarification(self):
        result = chat(ChatRequest(message="What is the weather on Mars?", session_id="unknown"))

        self.assertEqual(result.intent, "fallback")
        self.assertTrue(result.needs_clarification)

    def test_session_reset_removes_context(self):
        chat(ChatRequest(message="Track order #88213", session_id="reset-me"))
        clear_session("reset-me")

        self.assertNotIn("reset-me", _sessions)


if __name__ == "__main__":
    unittest.main()