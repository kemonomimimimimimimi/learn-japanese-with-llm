"""
LLM Learn Japanese Plugin

A plugin for learning Japanese with spaced repetition and AI-based exercises.
"""

from . import db
from . import scheduler  
from . import exercises
from . import structured
from . import plugin

__version__ = "0.1.0"
__all__ = ["db", "scheduler", "exercises", "structured", "plugin"]