"""Detectors sub-package for specialized entity detection modules."""

from .code_detector import CodeEntityDetector
from .numeric_detector import NumericalEntityDetector
from .web_detector import WebEntityDetector

__all__ = ["CodeEntityDetector", "NumericalEntityDetector", "WebEntityDetector"]
