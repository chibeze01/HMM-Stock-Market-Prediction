"""Unit tests for api.model_registry."""

import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.model_registry import ModelEntry, ModelRegistry


class TestModelRegistry(unittest.TestCase):
    def _entry(self, model="fake", evaluation=None, ticker="AAPL"):
        return ModelEntry(model=model, evaluation=evaluation or {}, ticker=ticker)

    def test_add_and_get(self):
        reg = ModelRegistry()
        model_id = reg.add(self._entry())
        self.assertIsInstance(model_id, str)
        entry = reg.get(model_id)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.model, "fake")
        self.assertEqual(entry.ticker, "AAPL")

    def test_get_missing(self):
        reg = ModelRegistry()
        self.assertIsNone(reg.get("nonexistent"))

    def test_delete(self):
        reg = ModelRegistry()
        model_id = reg.add(self._entry())
        self.assertTrue(reg.delete(model_id))
        self.assertIsNone(reg.get(model_id))

    def test_delete_missing(self):
        reg = ModelRegistry()
        self.assertFalse(reg.delete("nonexistent"))

    def test_list_ids(self):
        reg = ModelRegistry()
        id1 = reg.add(self._entry(ticker="A"))
        id2 = reg.add(self._entry(ticker="B"))
        ids = reg.list_ids()
        self.assertIn(id1, ids)
        self.assertIn(id2, ids)

    def test_eviction(self):
        reg = ModelRegistry(max_models=3)
        first = reg.add(self._entry())
        for _ in range(3):
            reg.add(self._entry())
        self.assertIsNone(reg.get(first))
        self.assertEqual(len(reg.list_ids()), 3)

    def test_thread_safety(self):
        reg = ModelRegistry()
        ids = []
        errors = []
        lock = threading.Lock()

        def worker():
            try:
                for _ in range(10):
                    mid = reg.add(ModelEntry(model="m", evaluation={}, ticker="T"))
                    with lock:
                        ids.append(mid)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(ids), 50)


if __name__ == "__main__":
    unittest.main()
