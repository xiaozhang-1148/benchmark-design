"""Joint DeepSeek-OCR2 embedding + 5 GT features → K-Means."""

__all__ = ["run"]


def __getattr__(name: str):
    if name == "run":
        from .pipeline import run

        return run
    if name == "main":
        from .pipeline import main

        return main
    raise AttributeError(name)
