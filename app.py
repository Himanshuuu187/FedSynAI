"""FedSynAI demo dashboard (Review-1 prototype).  Run:  streamlit run app.py"""
import json, os, sys
import pandas as pd, streamlit as st
sys.path.insert(0, os.path.dirname(__file__))
from fedsynai.data import prepare, partition, describe_partition
from fedsynai.fl import train_federated, train_centralized
from fedsynai.attacks import membership_auc

st.set_page_config(page_title="FedSynAI", page_icon="🔐", layout="wide")
RES = os.path.join(os.path.dirname(__file__), "results", "results.json")


@st.cache_data(show_spinner=False)
def load_results():
    return json.load(open(RES)) if os.path.exists(RES) else None


@st.cache_resource(show_spinner="Loading & preprocessing Adult dataset…")
def get_prep():
    return prepare(42)


R = load_results()
st.title("🔐 FedSynAI")
st.caption("Privacy-preserving collaborative AI using Federated Learning + Generative Synthetic Data ·") 
tabs = st.tabs(["Overview", "Clients & non-IID", "▶ Live FL demo", "Experiments", "Synthetic data", "Privacy", "Roadmap"])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clients (simulated orgs)", 4); c2.metric("Dataset", "UCI Adult")
    c3.metric("Algorithms", "FedAvg · FedProx"); c4.metric("Generator", "CTGAN")
    st.markdown("""
**Pipeline (as in the Phase-0 architecture):** 4 private client datasets → local training → server aggregation (FedAvg/FedProx)
→ optional privacy layer (client-level DP: clipping + Gaussian noise) → global model → evaluation engine
(Accuracy/F1/AUC, communication cost, membership inference, synthetic fidelity & utility) → this dashboard.

**Implemented so far:** data pipeline & Dirichlet non-IID partitioner · FL engine with FedAvg/FedProx · client-level DP with
RDP ε-accountant · loss-threshold membership-inference attack · per-client CTGAN generators with fidelity, TSTR utility and
DCR privacy proxy · FedSynAI-lite (synthetic augmentation to repair label skew) · experiments E1–E5-lite · this dashboard.
""")
    if R:
        st.info(f"Last full run: {len(R['experiments'])} experiments, {R['config']['rounds']} rounds, "
                f"{R['config']['train_rows']:,} train rows, {R['runtime_sec']:.0f}s on CPU.")

with tabs[1]:
    st.subheader("How the data is split across the 4 clients")
    if R:
        a, b = st.columns(2)
        for col, key, title in [(a, "partition_iid", "IID split"), (b, "partition_noniid", f"Non-IID split (Dirichlet α={R['config']['alpha']})")]:
            df = pd.DataFrame(R[key]).set_index("client")
            col.markdown(f"**{title}**"); col.bar_chart(df["rows"]); col.bar_chart(df["positive_rate"])
            col.dataframe(df.style.format({"positive_rate": "{:.1%}"}), width="stretch")
        st.caption("Positive rate = share of income >50K. Under non-IID, some clients almost never see the minority class — the root cause of client drift.")

with tabs[2]:
    st.subheader("Run federated learning live")
    c = st.columns(5)
    algo = c[0].selectbox("Algorithm", ["fedavg", "fedprox"])
    alpha = c[1].select_slider("Non-IID α (low = skewed)", [0.3, 0.5, 1.0, 5.0, 100.0], value=1.0)
    rounds = c[2].slider("Rounds", 3, 20, 8)
    use_dp = c[3].checkbox("Client-level DP")
    sigma = c[4].select_slider("Noise σ", [0.05, 0.1, 0.25, 0.5, 1.0], value=0.1, disabled=not use_dp)
    mu = st.slider("FedProx μ", 0.0, 1.0, 0.1, disabled=algo != "fedprox")
    if st.button("Start federated training", type="primary"):
        prep = get_prep()
        parts = partition(prep.y_train, 4, alpha, 42)
        st.dataframe(describe_partition(parts, prep.y_train).style.format({"positive_rate": "{:.1%}"}), width="stretch")
        with st.spinner("Training…"):
            m, h, comm = train_federated(prep, parts, rounds, algo=algo, mu=mu,
                                         dp_clip=1.0 if use_dp else None, dp_noise=sigma if use_dp else 0.0, seed=42)
        d = pd.DataFrame(h).set_index("round")
        x, y = st.columns(2)
        x.line_chart(d[["accuracy", "f1", "auc"]]); y.line_chart(d[["comm_mb"]])
        last = h[-1]
        k = st.columns(5)
        k[0].metric("Accuracy", f"{last['accuracy']:.3f}"); k[1].metric("F1", f"{last['f1']:.3f}")
        k[2].metric("AUC", f"{last['auc']:.3f}"); k[3].metric("Comm. cost", f"{comm/1e6:.1f} MB")
        k[4].metric("ε (δ=1e-5)", f"{last['epsilon']:.1f}" if use_dp else "∞ (no DP)")
        st.caption(f"Membership-inference AUC on final model: {membership_auc(m, prep.X_train, prep.y_train, prep.X_test, prep.y_test):.3f}  (0.5 = no leakage)")

with tabs[3]:
    if R:
        df = pd.DataFrame([{k: e[k] for k in ["id", "desc", "accuracy", "f1", "auc", "mia_auc", "comm_mb", "epsilon"]} for e in R["experiments"]])
        st.dataframe(df.style.format({"accuracy": "{:.3f}", "f1": "{:.3f}", "auc": "{:.3f}", "mia_auc": "{:.3f}", "comm_mb": "{:.1f}", "epsilon": "{:.1f}"}, na_rep="—"),
                     width="stretch", hide_index=True)
        st.markdown("**F1 per experiment** — the minority class is where non-IID hurts most")
        st.bar_chart(df.set_index("id")["f1"])
        st.markdown("**Convergence (test AUC per round)**")
        sel = st.multiselect("Experiments", [e["id"] for e in R["experiments"]], default=["E1", "E2", "E3a", "E3b", "E5-lite"])
        st.line_chart(pd.DataFrame({e["id"]: [r["auc"] for r in e["history"]] for e in R["experiments"] if e["id"] in sel}))

with tabs[4]:
    if R:
        st.subheader("Per-client CTGAN generators")
        s = pd.DataFrame(R["synthetic"]).set_index("client")
        st.dataframe(s.style.format("{:.3f}"), width="stretch")
        a, b = st.columns(2)
        a.markdown("**Column-wise fidelity (Client 1 synthetic vs Client 1 real)** — 1.0 = identical distribution")
        a.bar_chart(pd.Series(R["fidelity_per_column_client1"]))
        t = R["tstr"]
        b.markdown("**Utility (TSTR):** train on synthetic → test on real")
        b.bar_chart(pd.DataFrame({"AUC": [t["train_real_auc"], t["train_synthetic_auc"]], "F1": [t["train_real_f1"], t["train_synthetic_f1"]]},
                                 index=["trained on real", "trained on synthetic"]), stack=False)
        st.caption("DCR ratio ≈ 1 → synthetic rows are no closer to training rows than to unseen rows (no sign of memorisation).")
        p = os.path.join(os.path.dirname(__file__), "results", "synthetic_sample_client1.csv")
        if os.path.exists(p):
            st.markdown("**Sample of synthetic records (no real person)**"); st.dataframe(pd.read_csv(p).head(15), width="stretch")

with tabs[5]:
    if R:
        st.subheader("Privacy–utility trade-off")
        dp = [e for e in R["experiments"] if e["id"].startswith("E4")]
        d = pd.DataFrame({"σ (noise)": [float(e["id"].split("z")[1]) for e in dp], "ε": [e["epsilon"] for e in dp],
                          "AUC": [e["auc"] for e in dp], "F1": [e["f1"] for e in dp]})
        st.dataframe(d.style.format({"ε": "{:.1f}", "AUC": "{:.3f}", "F1": "{:.3f}"}), hide_index=True, width="stretch")
        st.line_chart(d.set_index("σ (noise)")[["AUC", "F1"]])
        st.markdown("""
- More noise → smaller ε (stronger privacy) but lower accuracy. With only **4 clients**, client-level DP is expensive — the key finding motivating
  synthetic data and secure aggregation as complementary tools.
- **Honest caveat:** ε values above ~10 give no meaningful formal guarantee. At σ ≤ 0.25 utility is preserved but ε is in the hundreds; at σ = 1 ε ≈ 31 and the model collapses.
  Closing this gap (more clients, subsampling, secure aggregation, better tuning) is the Review-2 goal.
- **Membership inference (loss-threshold)** on these small MLPs is ≈ 0.5 AUC, i.e. weak leakage; stronger attacks (shadow models, gradient inversion) are planned for Review 2.""")

with tabs[6]:
    st.subheader("Progress vs. plan")
    st.dataframe(pd.DataFrame([
        ("Data pipeline + IID/non-IID partitioner", "Done", "Review 0–1"),
        ("FL engine: FedAvg, FedProx", "Done", "Review 1"),
        ("Experiments E1, E2, E3 (+ comm. cost)", "Done", "Review 1"),
        ("Client-level DP + ε accountant (E4)", "Prototype", "Review 1→2"),
        ("Per-client CTGAN + fidelity / TSTR / DCR", "Prototype", "Review 1"),
        ("FedSynAI-lite (synthetic augmentation, E5)", "Prototype", "Review 1→2"),
        ("Membership inference (loss-threshold)", "Prototype", "Review 1→2"),
        ("Streamlit dashboard", "Prototype", "Review 1→2"),
        ("Secure aggregation (masking / SecAgg)", "Planned", "Review 2"),
        ("Federated / DP generator training, CTAB-GAN+", "Planned", "Review 2"),
        ("Opacus DP-SGD, Flower runtime, MLflow", "Planned", "Review 2"),
        ("Stronger attacks (shadow-model MIA, gradient inversion)", "Planned", "Review 2"),
        ("Multiple datasets, fairness & robustness study, final report", "Planned", "Review 2"),
    ], columns=["Work item", "Status", "Target"]), hide_index=True, width="stretch")
