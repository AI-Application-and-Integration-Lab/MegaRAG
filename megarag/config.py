"""Configuration and workspace compatibility checks, without model imports."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import yaml


DEFAULT_ADDON = {
    "language": "English",
    "example_number": 1,
    "insert_batch_size": 4,
    "entity_extract_max_gleaning": 0,
    "entity_refine_max_times": 1,
    "refine_subgraph_top_k": 120,
    "refine_subgraph_max_token_for_global_context": 8000,
    "refine_subgraph_max_token_for_local_context": 8000,
    "refine_subgraph_max_token_for_text_unit": 8000,
    "chunk_top_k": 6,
    "embed_parallel_limit": 1,
}


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    structured_output: bool = True
    max_async: int = 4
    timeout: float = 120.0
    extra_body: dict = field(default_factory=dict)

    def resolve_api_key(self):
        """An empty variable name explicitly selects an unauthenticated server."""
        if self.api_key_env == "":
            return "EMPTY"
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ValueError(f"Set {self.api_key_env} before running inference.")
        return key


@dataclass
class EmbeddingConfig:
    # This adapter requires the GME get_*_embeddings methods.
    model: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
    revision: str = "main"
    dimension: int = 1536
    max_tokens: int = 32768
    device: str = "auto"
    batch_size: int = 2


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    addon_params: dict = field(default_factory=lambda: dict(DEFAULT_ADDON))

    def validate(self):
        for name, value in {
            "llm.max_async": self.llm.max_async,
            "llm.timeout": self.llm.timeout,
            "embedding.dimension": self.embedding.dimension,
            "embedding.max_tokens": self.embedding.max_tokens,
            "embedding.batch_size": self.embedding.batch_size,
            "addon_params.embed_parallel_limit": self.addon_params.get(
                "embed_parallel_limit", 1
            ),
            "addon_params.chunk_top_k": self.addon_params.get("chunk_top_k", 6),
            "addon_params.insert_batch_size": self.addon_params.get(
                "insert_batch_size", 4
            ),
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be positive")
            if name != "llm.timeout" and not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        for name, value in {
            "llm.model": self.llm.model,
            "llm.base_url": self.llm.base_url,
            "embedding.model": self.embedding.model,
            "embedding.revision": self.embedding.revision,
            "embedding.device": self.embedding.device,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.llm.structured_output, bool):
            raise ValueError("llm.structured_output must be true or false")
        if not isinstance(self.llm.api_key_env, str) or (
            self.llm.api_key_env and not self.llm.api_key_env.strip()
        ):
            raise ValueError("llm.api_key_env must be an environment variable name or an empty string")
        if not isinstance(self.llm.extra_body, dict) or not all(
            isinstance(key, str) for key in self.llm.extra_body
        ):
            raise ValueError("llm.extra_body must be a mapping with string keys")
        reserved = {"model", "messages", "stream", "response_format"} & self.llm.extra_body.keys()
        if reserved:
            raise ValueError(f"llm.extra_body cannot override: {', '.join(sorted(reserved))}")
        try:
            json.dumps(self.llm.extra_body, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("llm.extra_body must contain JSON-compatible values") from exc
        if not all(isinstance(key, str) for key in self.addon_params):
            raise ValueError("addon_params keys must be strings")
        if any(key.startswith("_") for key in self.addon_params):
            raise ValueError("addon_params keys starting with '_' are reserved")
        return self


def load_config(path=None):
    if path is None:
        return Config().validate()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    if not all(isinstance(key, str) for key in data):
        raise ValueError("Configuration keys must be strings")
    # Existing recipe files contain addon parameters at the top level.
    if not any(key in data for key in ("llm", "embedding", "addon_params")):
        data = {"addon_params": data}
    unknown = set(data) - {"llm", "embedding", "addon_params"}
    if unknown:
        raise ValueError(
            f"Unknown configuration sections: {', '.join(sorted(unknown))}"
        )
    for key in ("llm", "embedding", "addon_params"):
        if key in data and not isinstance(data[key], dict):
            raise ValueError(f"{key} must be a YAML mapping")
    try:
        config = Config(
            llm=LLMConfig(**data.get("llm", {})),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            addon_params={**DEFAULT_ADDON, **data.get("addon_params", {})},
        )
    except TypeError as exc:
        raise ValueError(f"Invalid model configuration: {exc}") from exc
    return config.validate()


def config_fingerprint(config):
    data = asdict(config)
    # Execution limits do not change the embedding space or model selection.
    for key in ("device", "batch_size"):
        data["embedding"].pop(key)
    for key in ("api_key_env", "max_async", "timeout"):
        data["llm"].pop(key)
    data["addon_params"].pop("embed_parallel_limit", None)
    # Prompt changes invalidate persisted results even if model names match.
    data["prompts"] = hashlib.sha256(
        Path(__file__).with_name("prompt.py").read_bytes()
    ).hexdigest()
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def prepare_workspace(directory, config, *, ingest=False):
    """Reject incompatible/unversioned stores; rotate query cache on ingestion.

    CLI workspaces support one writer at a time. No existing index is deleted.
    """
    directory = Path(directory)
    manifest_path = directory / "megarag_manifest.json"
    fingerprint = config_fingerprint(config)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != 1
            or manifest.get("config_fingerprint") != fingerprint
        ):
            raise ValueError(
                "Workspace configuration changed. Use a new --workspace and ingest again; the existing index is preserved."
            )
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(
                "Workspace has no model manifest. Use a new, empty --workspace; legacy indexes are not adopted automatically."
            )
        if not ingest:
            raise ValueError(
                "Workspace is not initialized. Run 'megarag ingest' first."
            )
        manifest = {
            "schema": 1,
            "config_fingerprint": fingerprint,
            "embedding": {
                key: value
                for key, value in asdict(config.embedding).items()
                if key not in ("device", "batch_size")
            },
        }
    if ingest:
        manifest["cache_revision"] = uuid.uuid4().hex
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f".manifest-{uuid.uuid4().hex}.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
    else:
        # A manifest is created before parsing/model loading; it does not prove
        # that ingestion finished. Check the persisted document statuses first.
        status_path = directory / "kv_store_doc_status.json"
        statuses = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        if not isinstance(statuses, dict) or not statuses or any(
            not isinstance(doc, dict) or doc.get("status") != "processed"
            for doc in statuses.values()
        ):
            raise ValueError("Workspace ingestion is incomplete. Run 'megarag ingest' successfully before querying.")
    if not isinstance(manifest.get("cache_revision"), str):
        raise ValueError("Workspace manifest has no valid cache revision. Rerun ingest.")
    return manifest["cache_revision"]
