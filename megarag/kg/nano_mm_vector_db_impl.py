import time
import os
import PIL
import asyncio
import numpy as np
import pipmaster as pm

from PIL import Image
from typing import Any, final
from dataclasses import dataclass, asdict

if not pm.is_installed("nano-vectordb"):
    pm.install("nano-vectordb")

from lightrag.utils import (
    logger,
    compute_mdhash_id,
)
from nano_vectordb import NanoVectorDB
from nano_vectordb.dbs import (
    DataBase,
    load_storage,
    ConditionLambda,
    f_METRICS,
)

from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage

PIL.Image.MAX_IMAGE_PIXELS = None

from dataclasses import dataclass
from typing import Literal

@dataclass
class DotProductVectorDB(NanoVectorDB):
    """A NanoVectorDB variant that uses raw dot-product for similarity."""
    # Allow selecting "dot" (default) or falling back to "cosine" if needed.
    metric: Literal["dot", "cosine"] = "dot"

    def __post_init__(self):
        # Re-implement init to register our "dot" metric BEFORE asserting support.
        default_storage = {
            "embedding_dim": self.embedding_dim,
            "data": [],
            "matrix": np.array([], dtype=np.float32).reshape(0, self.embedding_dim),
        }
        storage: DataBase = load_storage(self.storage_file) or default_storage
        assert (
            storage["embedding_dim"] == self.embedding_dim
        ), f"Embedding dim mismatch, expected: {self.embedding_dim}, but loaded: {storage['embedding_dim']}"

        # Access the parent's private slot via name-mangling.
        self._NanoVectorDB__storage = storage

        # Register available metrics (include cosine to stay compatible).
        self.usable_metrics = {
            "dot": self._dot_query,
            "cosine": self._cosine_query,  # inherited behavior, optional
        }
        assert self.metric in self.usable_metrics, f"Metric {self.metric} not supported"

        # Only normalize if explicitly using cosine.
        self.pre_process()

        logger.info(f"Init {asdict(self)} {len(self._NanoVectorDB__storage['data'])} data")

    # --- Dot-product query implementation ---
    def _dot_query(
        self,
        query: np.ndarray,
        top_k: int,
        better_than_threshold: float | None,
        filter_lambda: ConditionLambda = None,
    ):
        # No normalization here — dot product relies on both direction and norm.
        if filter_lambda is None:
            use_matrix = self._NanoVectorDB__storage["matrix"]
            filter_index = np.arange(len(self._NanoVectorDB__storage["data"]))
        else:
            filter_index = np.array(
                [
                    i
                    for i, data in enumerate(self._NanoVectorDB__storage["data"])
                    if filter_lambda(data)
                ]
            )
            use_matrix = self._NanoVectorDB__storage["matrix"][filter_index]

        # Ensure numeric stability / dtype alignment.
        q = query.astype(np.float32, copy=False)
        scores = use_matrix @ q  # shape: (num_candidates,)
        # Take top_k by score (descending).
        if top_k <= 0:
            return []
        # Use argpartition for efficiency on large arrays; then sort those top indices.
        k = min(top_k, scores.shape[0])
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]  # descending

        sort_abs_index = filter_index[top_idx]
        results = []
        for abs_i, rel_i in zip(sort_abs_index, top_idx):
            s = float(scores[rel_i])
            # if better_than_threshold is not None and s < better_than_threshold:
            #     break
            results.append({**self._NanoVectorDB__storage["data"][abs_i], f_METRICS: s})
        return results

@final
@dataclass
class NanoMMVectorDBStorage(NanoVectorDBStorage):
    def __post_init__(self):
        # Initialize basic attributes
        self._client = None
        self._storage_lock = None
        self.storage_updated = None

        # Use global config value if specified, otherwise use default
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        working_dir = self.global_config["working_dir"]
        if self.workspace:
            # Include workspace in the file path for data isolation
            workspace_dir = os.path.join(working_dir, self.workspace)
            os.makedirs(workspace_dir, exist_ok=True)
            self._client_file_name = os.path.join(
                workspace_dir, f"vdb_{self.namespace}.json"
            )
        else:
            # Default behavior when workspace is empty
            self._client_file_name = os.path.join(
                working_dir, f"vdb_{self.namespace}.json"
            )
        self._max_batch_size = self.global_config["embedding_batch_num"]

        self._client = DotProductVectorDB(
            self.embedding_func.embedding_dim,
            storage_file=self._client_file_name,
            metric='dot',
        )

    # This class uses cosine for similarity calculate, which may cause some problems.
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """
        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """
        logger.debug(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        current_time = int(time.time())
        list_data = [
            {
                "__id__": k,
                "__created_at__": current_time,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        keys = list(list(data.values())[0].keys())
        contents, page_imgs, fig_imgs = [], [], []
        if "content" in keys:
            contents  = [v["content"] for v in data.values()]
        if "page_img" in keys:
            page_imgs = [Image.open(v["page_img"]).convert("RGB") for v in data.values()]
        if "fig_imgs" in keys:
            fig_imgs = [v["fig_imgs"] for v in data.values()]

        if len(page_imgs) > 0:
            batches = [
                {
                    "images":  page_imgs[i : i + self._max_batch_size],
                }
                for i in range(0, len(contents), self._max_batch_size)
            ]
        else:
            batches = [
                {
                    "texts" : contents[i : i + self._max_batch_size],
                }
                for i in range(0, len(contents), self._max_batch_size)
            ]
        # Execute embedding outside of lock to avoid long lock times
        embedding_tasks = [self.embedding_func(**batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        if len(embeddings) == len(list_data):
            for i, d in enumerate(list_data):
                d["__vector__"] = embeddings[i]
            client = await self._get_client()
            results = client.upsert(datas=list_data)
            return results
        else:
            # sometimes the embedding is not returned correctly. just log it.
            logger.error(
                f"embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}"
            )

    async def query(
        self, query: str, top_k: int, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        # Execute embedding outside of lock to avoid improve cocurrent
        embedding = await self.embedding_func(
            texts=[query], is_query=True
        )  # higher priority for query
        embedding = embedding[0]
        client = await self._get_client()
        results = client.query(
            query=embedding,
            top_k=top_k,
        )
        results = [
            {
                **dp,
                "id": dp["__id__"],
                "distance": dp["__metrics__"],
                "created_at": dp.get("__created_at__"),
            }
            for dp in results
        ]
        return results