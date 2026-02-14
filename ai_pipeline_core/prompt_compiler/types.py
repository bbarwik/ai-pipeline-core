"""Types for prompt specifications."""


class Phase(str):
    """Pipeline phase identifier for PromptSpec subclasses.

    Must be a non-empty string when creating a Phase (validated by PromptSpec at class definition time).
    Any non-empty string is valid (e.g., Phase('review'), Phase('analysis'), Phase('writing')).
    Used as a class parameter: ``class MySpec(PromptSpec, phase=Phase('review'))``.
    """


__all__ = ["Phase"]
