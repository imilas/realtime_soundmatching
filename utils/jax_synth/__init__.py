"""
JAX-based differentiable Faust synthesis.

This module is *only* importable from `.venv-gd` (Python 3.10 + JAX +
DawDreamer). The main venv (Python 3.14) cannot import this — keep gradient-
free code free of these imports.

Ported with minor edits from `/home/imilas/code/audio_nexting/helper_funcs/`.
"""
