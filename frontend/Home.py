"""
Streamlit entry point / landing page.

Run with `streamlit run frontend/Home.py`. The four pages under
`frontend/pages/` are auto-discovered by Streamlit's multipage support.
"""

import streamlit as st

st.set_page_config(page_title="Employee Well-Being Analysis", page_icon="📊")

st.title("Employee Well-Being Analysis")
st.write(
    "Explore what drives employee satisfaction with remote work, and "
    "predict satisfaction for an individual employee."
)

st.page_link("pages/1_Descriptive_Analysis.py", label="📈 Descriptive Analysis")
st.page_link("pages/2_Logistic_Model_Features.py", label="📊 Logistic Model — Top Features")
st.page_link("pages/3_Tree_Model_Features.py", label="🌳 Tree Model — Top Features")
st.page_link("pages/4_Predict.py", label="🔮 Predict")
