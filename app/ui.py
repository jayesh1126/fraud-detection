"""Fraud triage demo UI — thin client over the FastAPI service.

Run (with the API already up on :8000):
    uv run streamlit run app/ui.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="Fraud Triage", page_icon="🔍", layout="wide")
st.title("🔍 Fraud claim triage — scored & explained")


@st.cache_data(ttl=600)
def get_demo_claims():
    return pd.DataFrame(requests.get(f"{API}/demo_claims", timeout=10).json())


def reasons_chart(reasons):
    """Horizontal bar chart of SHAP contributions — the waterfall, webified."""
    df = pd.DataFrame(reasons)[::-1]                     # biggest bar on top
    labels = [f"{r.feature} = {r.value if r.value is not None else '(missing)'}"
              for r in df.itertuples()]
    colors = ["#d62728" if c > 0 else "#1f77b4" for c in df.contribution_logodds]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(df) + 0.5))
    ax.barh(labels, df.contribution_logodds, color=colors)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("contribution (log-odds)  ·  red → fraud, blue → legit")
    fig.tight_layout()
    return fig


def show_result(res):
    score, thr = res["fraud_score"], res["threshold"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud score", f"{score:.1%}")
    c2.metric("Decision", "🚨 INVESTIGATE" if res["decision"] == "investigate"
              else "✅ clear")
    c3.metric("Threshold (val-tuned)", f"{thr:.1%}")
    st.progress(min(score, 1.0))
    st.subheader("Why? — top contributions")
    st.pyplot(reasons_chart(res["top_reasons"]))


tab_demo, tab_whatif = st.tabs(["Score a held-out claim", "What-if (partial claim)"])

with tab_demo:
    claims = get_demo_claims()
    st.caption(f"{len(claims)} real claims from the held-out validation window — "
               "the model never trained on them.")
    tid = st.selectbox("Pick a claim", claims["TransactionID"].sort_values())
    if st.button("Score it", type="primary"):
        res = requests.get(f"{API}/demo_score/{tid}", timeout=30).json()
        show_result(res)
        with st.expander("Reveal ground truth"):
            actual = res["actually_fraud"]
            verdict = res["decision"] == "investigate"
            st.markdown(f"**Actually fraud: {actual}** — the model "
                        + ("✅ got it right." if actual == verdict
                           else "❌ got it wrong (see report: ~45% recall at this budget)."))

with tab_whatif:
    st.caption("Send a *partial* claim — unknown fields are treated as missing, "
               "exactly as in training.")
    col1, col2 = st.columns(2)
    amt = col1.number_input("TransactionAmt", 1.0, 10_000.0, 350.0)
    card6 = col1.selectbox("card6 (card type)", ["credit", "debit", None])
    product = col2.selectbox("ProductCD", ["W", "C", "R", "H", "S", None])
    r_email = col2.text_input("R_emaildomain (recipient)", "mail.com") or None

    if st.button("Score partial claim", type="primary"):
        payload = {"features": {k: v for k, v in {
            "TransactionAmt": amt, "card6": card6,
            "ProductCD": product, "R_emaildomain": r_email,
        }.items() if v is not None}}
        res = requests.post(f"{API}/score", json=payload, timeout=30).json()
        show_result(res)
