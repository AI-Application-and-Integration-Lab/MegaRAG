import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from megarag.kg import nano_mm_vector_db_impl as storage


def seed(path):
    db = storage.DotProductVectorDB(2, storage_file=str(path))
    db.upsert([{'__id__': 'a', '__vector__': np.array([10, 0], dtype=np.float32)},
               {'__id__': 'b', '__vector__': np.array([1, 1], dtype=np.float32)}])
    db.save()
    return db


def test_dot_filter_threshold_and_reload_updates(tmp_path):
    path = tmp_path / 'vectors.json'
    db = seed(path)
    query = np.array([1, 1], dtype=np.float32)
    assert db.query(query, filter_lambda=lambda row: False) == []
    assert [row['__id__'] for row in db.query(query, better_than_threshold=3)] == ['a']
    reloaded = storage.DotProductVectorDB(2, storage_file=str(path))
    reloaded.upsert([{'__id__': 'a', '__vector__': np.array([20, 0], dtype=np.float32)}])
    assert reloaded.query(query, top_k=1)[0]['__metrics__'] == 20


@pytest.mark.parametrize('operation', ['reload', 'save_conflict', 'drop'])
def test_storage_keeps_dot_metric(operation, tmp_path, monkeypatch):
    async def embed(**kwargs):
        return np.array([[1, 1]], dtype=np.float32)
    embed.embedding_dim = 2
    owner = storage.NanoMMVectorDBStorage(namespace='chunks', workspace='',
        global_config={'working_dir': str(tmp_path), 'embedding_batch_num': 2,
                       'vector_db_storage_cls_kwargs': {'cosine_better_than_threshold': 0.2}},
        embedding_func=embed)
    owner._client = seed(tmp_path / 'vdb_chunks.json')
    owner._storage_lock = asyncio.Lock()
    owner.storage_updated = SimpleNamespace(value=True)
    async def notify(*args):
        pass
    monkeypatch.setattr('lightrag.kg.nano_vector_db_impl.set_all_update_flags', notify)
    if hasattr(storage, 'set_all_update_flags'):
        monkeypatch.setattr(storage, 'set_all_update_flags', notify)
    async def run():
        if operation == 'reload':
            await owner._get_client()
        elif operation == 'save_conflict':
            assert await owner.index_done_callback() is False
        else:
            assert (await owner.drop())['status'] == 'success'
            assert await owner.query('question', top_k=1) == []
            owner._client.upsert([{'__id__': 'a', '__vector__': np.array([10, 0], dtype=np.float32)},
                                  {'__id__': 'b', '__vector__': np.array([1, 1], dtype=np.float32)}])
        assert owner._client.metric == 'dot'
        assert (await owner.query('question', top_k=1))[0]['id'] == 'a'
        assert (await owner.query('question', top_k=1, ids=['b']))[0]['id'] == 'b'
        assert await owner.query('question', top_k=1, ids=['missing']) == []
        assert await owner.query('question', top_k=1, ids=[]) == []
    asyncio.run(run())
