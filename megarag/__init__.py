"""MegaRAG public API. Load inference dependencies only when needed."""

__all__ = ["MegaRAG"]


def __getattr__(name):
    if name == "MegaRAG":
        from .megarag import MegaRAG
        return MegaRAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
