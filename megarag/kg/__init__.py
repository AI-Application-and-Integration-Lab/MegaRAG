from lightrag.kg import (
    STORAGE_IMPLEMENTATIONS,
    STORAGE_ENV_REQUIREMENTS,
    STORAGES,
)

STORAGE_IMPLEMENTATIONS['VECTOR_STORAGE']['implementations'].append('NanoMMVectorDBStorage')
STORAGE_ENV_REQUIREMENTS['NanoMMVectorDBStorage'] = []

STORAGES = {key: f'lightrag{STORAGES[key]}' for key in STORAGES}
STORAGES['NanoMMVectorDBStorage'] = "megarag.kg.nano_mm_vector_db_impl"

def verify_storage_implementation(storage_type: str, storage_name: str) -> None:
    """Verify if storage implementation is compatible with specified storage type

    Args:
        storage_type: Storage type (KV_STORAGE, GRAPH_STORAGE etc.)
        storage_name: Storage implementation name

    Raises:
        ValueError: If storage implementation is incompatible or missing required methods
    """
    if storage_type not in STORAGE_IMPLEMENTATIONS:
        raise ValueError(f"Unknown storage type: {storage_type}")

    storage_info = STORAGE_IMPLEMENTATIONS[storage_type]
    if storage_name not in storage_info["implementations"]:
        raise ValueError(
            f"Storage implementation '{storage_name}' is not compatible with {storage_type}. "
            f"Compatible implementations are: {', '.join(storage_info['implementations'])}"
        )
