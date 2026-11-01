"""Phase 5 local research application: frozen pipeline, decision rule, diagnostics and Gradio UI.

``config``, ``inputs``, ``decision``, ``report``, ``server`` and ``tempfiles`` need no model stack;
``face``, ``pipeline`` and ``diagnostics`` load models lazily; ``ui`` imports gradio only when the interface is built.
"""
