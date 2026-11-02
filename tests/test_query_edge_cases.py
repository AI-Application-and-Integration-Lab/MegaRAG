import asyncio
from types import SimpleNamespace

import pytest
from lightrag.base import QueryParam
from megarag import operate
from megarag.prompt import PROMPTS


async def answer(*args, **kwargs):
    return 'A user compares model results for question. ' * 1000


@pytest.fixture
def query_setup(monkeypatch):
    async def keywords(*args, **kwargs):
        return ['topic'], ['entity']
    async def context(*args, **kwargs):
        return 'evidence', []
    monkeypatch.setattr(operate, 'get_keywords_from_query', keywords)
    monkeypatch.setattr(operate, '_build_query_context_with_image', context)
    return {'llm_model_func': answer, 'tokenizer': SimpleNamespace(encode=list)}


def test_empty_extraction_is_valid():
    asyncio.run(operate.merge_nodes_and_edges(
        [({}, {})], None, None, None, {'llm_model_max_async': 1},
        pipeline_status={'history_messages': []}, pipeline_status_lock=asyncio.Lock()))
    assert asyncio.run(operate.extract_entities({}, {})) == []


def test_empty_graph_returns_no_evidence(query_setup, monkeypatch):
    async def empty(*args, **kwargs):
        return None
    monkeypatch.setattr(operate, '_build_query_context_with_image', empty)
    assert asyncio.run(operate.kg_query('question', None, None, None, None,
        QueryParam(mode='hybrid'), query_setup)) == PROMPTS['fail_response']


def test_graph_answer_preserved_without_cache(query_setup):
    result = asyncio.run(operate.kg_query('question', None, None, None, None,
        QueryParam(mode='hybrid'), query_setup))
    assert result == asyncio.run(answer())


@pytest.mark.parametrize('flag', ['only_need_context', 'only_need_prompt', None])
def test_two_step_preserves_options(query_setup, monkeypatch, flag):
    params = []
    async def graph(*args, **kwargs):
        params.append(args[5])
        return 'graph evidence'
    async def vector(*args, **kwargs):
        params.append(args[3])
        return 'vector evidence'
    calls = []
    async def model(*args, **kwargs):
        calls.append(kwargs)
        return await answer()
    monkeypatch.setattr(operate, 'kg_query', graph)
    monkeypatch.setattr(operate, 'naive_query', vector)
    query_setup['llm_model_func'] = model
    original = QueryParam(mode='mix_two_step', stream=True, enable_rerank=True,
                          ids=['chunk-a'], max_total_tokens=4000,
                          conversation_history=[{'role': 'user', 'content': 'Earlier question'}])
    if flag:
        setattr(original, flag, True)
    result = asyncio.run(operate.kg_two_step_query('question', None, None, None, None,
                                                original, query_setup))
    for param in params:
        assert param.ids == original.ids
        assert param.conversation_history == original.conversation_history
        assert param.max_total_tokens == 4000
        assert param.enable_rerank
        assert not param.stream  # Fusion needs completed branch answers.
        if flag:
            assert getattr(param, flag)
    assert original.mode == 'mix_two_step' and original.stream
    if flag:
        assert not calls
        assert 'graph evidence' in result and 'vector evidence' in result
    else:
        assert calls[0]['stream'] is True
        assert result == asyncio.run(answer())


@pytest.mark.parametrize('truncated', [True, False])
def test_vector_context_empty_or_text_only(query_setup, monkeypatch, truncated):
    async def search(*args, **kwargs):
        return [{'id': 'a', 'content': 'evidence', 'file_path': 'source'}]
    async def process(*args, **kwargs):
        return [] if truncated else await search()
    async def get(*args):
        return [{'content': 'evidence'}]
    monkeypatch.setattr(operate, '_get_vector_context', search)
    monkeypatch.setattr(operate, 'process_chunks_unified', process)
    result = asyncio.run(operate.naive_query('question', None, SimpleNamespace(get_by_ids=get),
                                            QueryParam(mode='naive'), query_setup))
    assert result == (PROMPTS['fail_response'] if truncated else asyncio.run(answer()))


def test_keyword_fallback_keeps_caller_and_cache_mode(query_setup, monkeypatch):
    async def keywords(*args, **kwargs):
        return ['topic'], []
    saved = []
    async def save(cache, data):
        saved.append(data)
    async def get(*args):
        return None
    monkeypatch.setattr(operate, 'get_keywords_from_query', keywords)
    monkeypatch.setattr(operate, 'save_to_cache', save)
    param = QueryParam(mode='hybrid')
    cache = SimpleNamespace(global_config={'enable_llm_cache': True}, get_by_id=get)
    asyncio.run(operate.kg_query('question', None, None, None, None, param, query_setup, hashing_kv=cache))
    assert param.mode == 'hybrid'
    assert saved[0].mode == 'hybrid'


def test_stream_is_returned_without_serializing_iterator(query_setup, monkeypatch):
    async def stream():
        yield 'First '
        yield 'second'
    async def model(*args, **kwargs):
        return stream()
    async def get(*args):
        return None
    async def save(*args):
        pytest.fail('An unfinished stream cannot be stored as a complete answer')
    monkeypatch.setattr(operate, 'save_to_cache', save)
    query_setup['llm_model_func'] = model
    cache = SimpleNamespace(global_config={'enable_llm_cache': True}, get_by_id=get)
    async def run():
        result = await operate.kg_query('question', None, None, None, None,
            QueryParam(mode='hybrid', stream=True), query_setup, hashing_kv=cache)
        return ''.join([part async for part in result])
    assert asyncio.run(run()) == 'First second'
