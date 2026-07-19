"""
Metrics Description page.

Renders the actual markdown docs from `training/` describing the metrics
and feature-importance methods used elsewhere in the app (macro-F1,
multiclass ROC/AUC, standardized coefficients, permutation importance) —
reads the real files rather than duplicating their content, so this page
can never drift out of sync with the docs.
"""

from pathlib import Path
from typing import ClassVar

import streamlit as st


class MetricsDescriptionPage:
    """Renders each training/*.md doc in its own tab.

    Entry points:
        render: Draw the full page.

    Args:
        training_dir: Directory containing the markdown docs to render.
    """

    # ClassVar: the set of docs to show is a property of the page, not of
    # any particular instance.
    DOC_FILES: ClassVar[list[str]] = [
        "macro_f1.md",
        "multiclass_auc.md",
        "feature_importance_methods.md",
        "metrics_and_target_encoding_ideas.md",
    ]

    def __init__(self, training_dir: Path) -> None:
        self._training_dir = training_dir

    def _read(self, filename: str) -> str:
        return (self._training_dir / filename).read_text()

    def _tab_label(self, content: str) -> str:
        # The first line of every doc is a "# Title" heading — strip the
        # markdown heading marker to use as this tab's short label.
        first_line = content.splitlines()[0]
        return first_line.lstrip("#").strip()

    def render(self) -> None:
        """Draw one tab per doc in `DOC_FILES`, each showing its full content."""
        contents = [self._read(filename) for filename in self.DOC_FILES]
        tabs = st.tabs([self._tab_label(content) for content in contents])
        for tab, content in zip(tabs, contents):
            with tab:
                st.markdown(content)


st.set_page_config(page_title="Metrics Description", page_icon="📚")
st.title("📚 Metrics Description")
st.caption(
    "Background on the metrics and feature-importance methods shown on the "
    "model pages — rendered directly from the docs in `training/`."
)

_repo_root = Path(__file__).resolve().parent.parent.parent
MetricsDescriptionPage(_repo_root / "training").render()
