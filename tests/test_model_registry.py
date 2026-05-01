"""Tests for the thread-safe model registry."""

import threading


def _make_entry(**kwargs):
    from api.model_registry import ModelEntry

    defaults = {"model": "fake_model", "ticker": "AAPL", "evaluation": {"score": 1.0}}
    defaults.update(kwargs)
    return ModelEntry(**defaults)


def test_add_and_get():
    from api.model_registry import ModelRegistry

    registry = ModelRegistry()
    entry = _make_entry()
    model_id = registry.add(entry)
    retrieved = registry.get(model_id)
    assert retrieved is entry
    assert retrieved.model == "fake_model"
    assert retrieved.ticker == "AAPL"


def test_get_nonexistent_returns_none():
    from api.model_registry import ModelRegistry

    registry = ModelRegistry()
    assert registry.get("nonexistent-id") is None


def test_delete_existing():
    from api.model_registry import ModelRegistry

    registry = ModelRegistry()
    model_id = registry.add(_make_entry())
    assert registry.delete(model_id) is True
    assert registry.get(model_id) is None


def test_delete_nonexistent_returns_false():
    from api.model_registry import ModelRegistry

    registry = ModelRegistry()
    assert registry.delete("nonexistent-id") is False


def test_list_ids():
    from api.model_registry import ModelRegistry

    registry = ModelRegistry()
    id1 = registry.add(_make_entry())
    id2 = registry.add(_make_entry())
    ids = registry.list_ids()
    assert set(ids) == {id1, id2}


def test_evicts_oldest_when_at_capacity():
    from api.model_registry import ModelRegistry

    registry = ModelRegistry(max_models=2)
    id1 = registry.add(_make_entry())
    _id2 = registry.add(_make_entry())
    id3 = registry.add(_make_entry())
    assert registry.get(id1) is None
    assert registry.get(id3) is not None
    assert len(registry.list_ids()) == 2


def test_thread_safety():
    from api.model_registry import ModelRegistry

    registry = ModelRegistry()
    errors = []

    def add_models():
        try:
            for i in range(20):
                registry.add(_make_entry(model=f"m{i}"))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=add_models) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # Default max_models=50, so at most 50 retained
    assert len(registry.list_ids()) == 50


def test_update_entry():
    from api.model_registry import ModelRegistry

    registry = ModelRegistry()
    model_id = registry.add(_make_entry())
    updated = registry.update(model_id, model="new_model", evaluation={"v": 2})
    assert updated is True
    entry = registry.get(model_id)
    assert entry.model == "new_model"
    assert entry.evaluation == {"v": 2}


def test_update_nonexistent_returns_false():
    from api.model_registry import ModelRegistry

    registry = ModelRegistry()
    assert registry.update("nonexistent", model="m") is False
