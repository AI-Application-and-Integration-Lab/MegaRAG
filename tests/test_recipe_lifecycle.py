import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_recipe(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'egs/utils' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('failed', [False, True])
def test_build_recipe_uses_one_loop_and_closes_storage(tmp_path, monkeypatch, failed):
    recipe = load_recipe('construct_mmkg')
    events = []
    loops = []
    config = tmp_path / 'config.yaml'
    config.write_text('{}')
    source = tmp_path / 'pages.json'
    source.write_text('{}')
    async def initialize(*args, **kwargs):
        loops.append(asyncio.get_running_loop())
        return SimpleNamespace(ainsert=insert, finalize_storages=finalize,
                               doc_status=SimpleNamespace(get_docs_by_status=status)), 'usage'
    async def insert(**kwargs):
        events.append('insert')
        loops.append(asyncio.get_running_loop())
    async def status(*args):
        return {'failed-document': {}} if failed else {}
    async def finalize():
        events.append('finalize')
        loops.append(asyncio.get_running_loop())
    monkeypatch.setattr(recipe, 'initialize_rag', initialize)
    args = SimpleNamespace(working_dir=tmp_path / 'index', config_file=config,
                           input_dir=source, max_retries=1, retry_delay=0)
    if failed:
        with pytest.raises(RuntimeError, match='exceeding'):
            asyncio.run(recipe.run(args))
    else:
        asyncio.run(recipe.run(args))
    assert events == ['insert', 'finalize']
    assert len({id(loop) for loop in loops}) == 1


def test_query_recipe_finalizes_after_input_failure(tmp_path, monkeypatch):
    recipe = load_recipe('query_mmkg')
    config = tmp_path / 'config.yaml'
    config.write_text('{}')
    questions = tmp_path / 'questions.txt'
    questions.write_text('')
    closed = []
    async def finalize():
        closed.append(True)
    async def initialize(*args, **kwargs):
        return SimpleNamespace(finalize_storages=finalize), 'usage'
    monkeypatch.setattr(recipe, 'initialize_rag', initialize)
    monkeypatch.setattr(recipe, '_torch_speed_tweaks', lambda: None)
    monkeypatch.setattr(recipe, 'parse_args', lambda: SimpleNamespace(
        working_dir=tmp_path / 'index', config_file=config, input_queries=questions))
    with pytest.raises(SystemExit, match='No questions'):
        asyncio.run(recipe.async_main())
    assert closed == [True]
