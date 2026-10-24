"""
cerebellum_viz : napari visualisation of the cerebellum volumes, and the blinded vermis
annotation tool built on it (vault : Run 20, annotation-vermis).

    from cerebellum_viz import view, annotate
    viewer, sub = view("/path/to/dataset", "sub-XXXX", ["t1w", "zscore"])
    viewer, session, form = annotate("/path/to/project", "rater")
"""

from .data import (
    DEFAULT_LAYOUT, MODALITIES_TYPES, SES_LAYOUT,
    BinaryMask, Dataset, LabelVolume, Modality, ScalarVolume, Subject, Surface, T1w, ZScore,
    reorient,
)
from .annotation import (
    ANNOTATION_VERSION, AnnotationProject, AnnotationSession, Protocol, protocol_file,
)
from .viewer import LAYOUTS, annotate, view

__version__ = "0.1.0"
