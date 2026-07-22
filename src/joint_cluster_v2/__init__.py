"""Joint clustering V2 (type-identity GT + fixed K=4)."""

__all__ = ["run", "main"]


def __getattr__(name: str):
    if name in {"run", "main"}:
        from . import pipeline_v2

        return getattr(pipeline_v2, name)
    raise AttributeError(name)
