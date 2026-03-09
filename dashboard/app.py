"""
PharmaDash - Pharmaceutical Tablet Manufacturing Intelligence Dashboard
=====================================================================
Multi-output XGBoost model for predicting tablet quality attributes
and optimizing manufacturing process parameters.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import io
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PharmaDash - Tablet Manufacturing Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141428 0%, #1c1c3a 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 16px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
    }
    div[data-testid="stMetric"] label {
        color: #a5b4fc !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e0e7ff !important;
        font-weight: 700;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 30, 60, 0.5);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #a5b4fc;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.2) !important;
        color: #e0e7ff !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(99, 102, 241, 0.08);
        border-radius: 8px;
        color: #c7d2fe;
    }

    /* Headers */
    h1, h2, h3 {
        color: #e0e7ff !important;
    }

    /* Section divider */
    .section-header {
        background: linear-gradient(90deg, rgba(99,102,241,0.15) 0%, transparent 100%);
        border-left: 4px solid #6366f1;
        padding: 12px 20px;
        border-radius: 0 8px 8px 0;
        margin: 24px 0 16px 0;
    }
    .section-header h2 {
        margin: 0;
        font-size: 1.4rem;
        color: #c7d2fe !important;
    }

    /* KPI card */
    .kpi-card {
        background: rgba(99, 102, 241, 0.06);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 14px;
        padding: 24px;
        text-align: center;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #818cf8;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Hide hamburger and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Plotly chart background overrides */
    .js-plotly-plot .plotly .main-svg {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
    "accent": "#06b6d4",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "bg_dark": "#0f0c29",
    "bg_card": "rgba(30,30,60,0.6)",
    "text": "#e0e7ff",
    "text_muted": "#94a3b8",
    "palette": ["#6366f1", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"],
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,12,41,0.4)",
    font=dict(color="#c7d2fe", family="Inter, system-ui, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_default_data():
    """Load the default batch production data shipped with the project."""
    import os

    # Look for the data file in parent directory or current
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "_h_batch_production_data.xlsx"),
        os.path.join(os.path.dirname(__file__), "_h_batch_production_data.xlsx"),
        "_h_batch_production_data.xlsx",
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_excel(path)
    return None


@st.cache_data
def load_process_data():
    """Load the batch process (time-series) data."""
    import os

    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "_h_batch_process_data.xlsx"),
        os.path.join(os.path.dirname(__file__), "_h_batch_process_data.xlsx"),
        "_h_batch_process_data.xlsx",
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_excel(path)
    return None


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
TARGETS = [
    "Dissolution_Rate",
    "Hardness",
    "Content_Uniformity",
    "Tablet_Weight",
    "Friability",
    "Energy_Index",
]

FEATURE_ORDER = [
    "Granulation_Time",
    "Binder_Amount",
    "Drying_Temp",
    "Drying_Time",
    "Compression_Force",
    "Moisture_Content",
]

FEATURE_RANGES = {
    "Granulation_Time": (5.0, 25.0),
    "Binder_Amount": (2.0, 12.0),
    "Drying_Temp": (40.0, 80.0),
    "Drying_Time": (10.0, 45.0),
    "Compression_Force": (5.0, 20.0),
    "Moisture_Content": (0.5, 5.0),
}


@st.cache_resource
def build_model(data: pd.DataFrame):
    """Train the multi-output XGBoost model and return artefacts."""
    df = data.copy()
    df["Energy_Index"] = (
        0.4 * df["Machine_Speed"]
        + 0.3 * df["Compression_Force"]
        + 0.3 * df["Drying_Temp"]
    )

    X_all = df.drop(columns=["Batch_ID"] + [t for t in TARGETS if t in df.columns])
    y_all = df[TARGETS]

    # Feature selection
    selector = SelectKBest(score_func=f_regression, k=6)
    selector.fit(X_all, y_all.mean(axis=1))
    selected = X_all.columns[selector.get_support()].tolist()

    X = X_all[FEATURE_ORDER]  # keep consistent order
    y = y_all

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MultiOutputRegressor(
        XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=4,
                     random_state=42, n_jobs=1, tree_method="hist")
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # Metrics
    metrics = {}
    for i, t in enumerate(TARGETS):
        metrics[t] = {
            "r2": r2_score(y_test.iloc[:, i], preds[:, i]),
            "mae": mean_absolute_error(y_test.iloc[:, i], preds[:, i]),
            "rmse": np.sqrt(mean_squared_error(y_test.iloc[:, i], preds[:, i])),
        }

    # Cross-val for Dissolution_Rate (lightweight)
    cv_scores = cross_val_score(
        XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=4,
                     random_state=42, n_jobs=1, tree_method="hist"),
        X,
        y["Dissolution_Rate"],
        cv=3,
        scoring="r2",
    )

    # Feature importances per target
    importances = {}
    for i, t in enumerate(TARGETS):
        importances[t] = dict(
            zip(FEATURE_ORDER, model.estimators_[i].feature_importances_)
        )

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preds": preds,
        "metrics": metrics,
        "cv_scores": cv_scores,
        "importances": importances,
        "full_X": X,
        "full_y": y,
        "data": df,
    }


def forecast(model, params: dict) -> dict:
    sample = pd.DataFrame([params])[FEATURE_ORDER]
    pred = model.predict(sample)[0]
    return dict(zip(TARGETS, pred))


def detect_risk(pred: dict) -> list:
    risks = []
    if pred["Friability"] > 1.0:
        risks.append(("Tablet breakage risk -- Friability > 1.0%", "danger"))
    if pred["Hardness"] < 70:
        risks.append(("Low tablet strength -- Hardness < 70 N", "danger"))
    if pred["Content_Uniformity"] < 95 or pred["Content_Uniformity"] > 105:
        risks.append(("Content uniformity out of spec (95-105%)", "warning"))
    if pred["Dissolution_Rate"] < 80:
        risks.append(("Dissolution rate below 80%", "warning"))
    if not risks:
        risks.append(("Process stable -- all parameters within spec", "success"))
    return risks


def recommend_params(model, current: dict, trials: int = 500) -> tuple:
    best_score = -1e9
    best_params = None
    best_pred = None
    for _ in range(trials):
        candidate = current.copy()
        for f in FEATURE_RANGES:
            candidate[f] = np.random.uniform(*FEATURE_RANGES[f])
        pred = forecast(model, candidate)
        score = pred["Dissolution_Rate"] - 0.1 * pred["Energy_Index"]
        if score > best_score:
            best_score = score
            best_params = candidate
            best_pred = pred
    return best_params, best_pred


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 20px 0;">
            <h1 style="font-size:1.8rem; margin:0; background: linear-gradient(135deg, #6366f1, #06b6d4);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                PharmaDash
            </h1>
            <p style="color:#94a3b8; font-size:0.85rem; margin-top:4px;">
                Tablet Manufacturing Intelligence
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Data source
    data_source = st.radio(
        "Data source",
        ["Default dataset", "Upload your own"],
        index=0,
    )

    if data_source == "Upload your own":
        uploaded = st.file_uploader("Upload Excel / CSV", type=["xlsx", "csv"])
        if uploaded:
            if uploaded.name.endswith(".csv"):
                raw_data = pd.read_csv(uploaded)
            else:
                raw_data = pd.read_excel(uploaded)
        else:
            raw_data = load_default_data()
    else:
        raw_data = load_default_data()

    if raw_data is None:
        st.error("Could not find the data file. Place `_h_batch_production_data.xlsx` next to the dashboard folder.")
        st.stop()

    st.markdown("---")
    st.caption(f"Dataset: **{len(raw_data)} batches**, **{len(raw_data.columns)} features**")

    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Dataset Insights",
            "Preprocessing",
            "Model Performance",
            "Predict & Simulate",
            "Optimizer",
            "3D Explorer",
            "How to Use",
        ],
        index=0,
    )

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
art = build_model(raw_data)
model = art["model"]
data = art["data"]
process_data = load_process_data()


# ====================================================================
# PAGE: Overview
# ====================================================================
if page == "Overview":
    st.markdown(
        """
        <div style="text-align:center; padding: 30px 0 10px 0;">
            <h1 style="font-size:2.8rem; margin:0;
                        background: linear-gradient(135deg, #6366f1, #06b6d4);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                PharmaDash
            </h1>
            <p style="color:#94a3b8; font-size:1.1rem; margin-top:8px;">
                AI-Powered Pharmaceutical Tablet Manufacturing Intelligence
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # KPI row
    avg_r2 = np.mean([m["r2"] for m in art["metrics"].values()])
    cv_mean = art["cv_scores"].mean()
    best_diss = data["Dissolution_Rate"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Batches Analysed", f"{len(data)}")
    c2.metric("Avg Model R\u00b2", f"{avg_r2:.3f}")
    c3.metric("CV Score (Dissolution)", f"{cv_mean:.3f}")
    c4.metric("Best Dissolution", f"{best_diss:.1f}%")

    st.markdown("")

    # What this dashboard does
    st.markdown('<div class="section-header"><h2>What This Dashboard Does</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **Manufacturing Process Optimization**
            - Predicts 6 tablet quality attributes simultaneously
            - Uses a multi-output XGBoost regression model
            - Trained on real pharmaceutical batch production data
            - Handles 6 input process parameters

            **Key Quality Targets**
            | Target | Description |
            |--------|-------------|
            | Dissolution Rate | Drug release performance |
            | Hardness | Mechanical strength |
            | Content Uniformity | Dose consistency |
            | Tablet Weight | Weight control |
            | Friability | Breakage resistance |
            | Energy Index | Process energy efficiency |
            """
        )
    with col2:
        st.markdown(
            """
            **Dashboard Capabilities**
            - Interactive dataset exploration with 3D visualizations
            - Real-time quality predictions from process parameters
            - What-if scenario simulation & comparison
            - Automated parameter optimization (max dissolution, min energy)
            - Risk detection and process stability alerts
            - Full Pareto-front visualization for multi-objective trade-offs

            **Technology Stack**
            - Streamlit, Plotly, XGBoost, scikit-learn
            - Deployable on Streamlit Cloud, HuggingFace Spaces, Render
            """
        )

    # Quick target summary radar
    st.markdown('<div class="section-header"><h2>Model Performance Snapshot</h2></div>', unsafe_allow_html=True)

    r2_vals = [art["metrics"][t]["r2"] for t in TARGETS]
    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=r2_vals + [r2_vals[0]],
            theta=TARGETS + [TARGETS[0]],
            fill="toself",
            fillcolor="rgba(99,102,241,0.15)",
            line=dict(color=COLORS["primary"], width=2),
            name="R\u00b2 Score",
        )
    )
    fig_radar.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor="rgba(15,12,41,0.4)",
            radialaxis=dict(visible=True, range=[0.85, 1.0], color="#94a3b8"),
            angularaxis=dict(color="#c7d2fe"),
        ),
        title="R\u00b2 Score Across All Targets",
        height=420,
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ====================================================================
# PAGE: Dataset Insights
# ====================================================================
elif page == "Dataset Insights":
    st.markdown('<div class="section-header"><h2>Dataset Insights</h2></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Raw Data", "Distributions", "Correlations"])

    with tab1:
        st.dataframe(data.style.format(precision=2), use_container_width=True, height=400)
        csv = data.to_csv(index=False)
        st.download_button("Download CSV", csv, "batch_data.csv", "text/csv")

    with tab2:
        num_cols = data.select_dtypes(include=np.number).columns.tolist()
        num_cols = [c for c in num_cols if c != "Batch_ID"]
        selected_dist = st.multiselect("Select columns", num_cols, default=num_cols[:4])
        if selected_dist:
            fig = make_subplots(
                rows=1,
                cols=len(selected_dist),
                subplot_titles=selected_dist,
            )
            for i, col in enumerate(selected_dist):
                fig.add_trace(
                    go.Histogram(
                        x=data[col],
                        marker_color=COLORS["palette"][i % len(COLORS["palette"])],
                        opacity=0.8,
                        name=col,
                    ),
                    row=1,
                    col=i + 1,
                )
            fig.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        corr = data[num_cols].corr()
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu_r",
                zmid=0,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                textfont=dict(size=9),
            )
        )
        fig_corr.update_layout(**PLOTLY_LAYOUT, height=550, title="Feature Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

    # Process data section
    if process_data is not None:
        st.markdown('<div class="section-header"><h2>Batch Process Time-Series</h2></div>', unsafe_allow_html=True)
        batch_ids = process_data["Batch_ID"].unique().tolist()
        sel_batch = st.selectbox("Select Batch", batch_ids)
        batch_ts = process_data[process_data["Batch_ID"] == sel_batch]
        ts_cols = [c for c in process_data.columns if c not in ["Batch_ID", "Time_Minutes", "Phase"]]
        sel_ts = st.multiselect("Sensors", ts_cols, default=ts_cols[:3])
        if sel_ts:
            fig_ts = go.Figure()
            for j, col in enumerate(sel_ts):
                fig_ts.add_trace(
                    go.Scatter(
                        x=batch_ts["Time_Minutes"],
                        y=batch_ts[col],
                        mode="lines+markers",
                        name=col,
                        line=dict(color=COLORS["palette"][j % len(COLORS["palette"])], width=2),
                    )
                )
            fig_ts.update_layout(
                **PLOTLY_LAYOUT,
                title=f"Process Sensors -- Batch {sel_batch}",
                xaxis_title="Time (min)",
                yaxis_title="Value",
                height=400,
            )
            st.plotly_chart(fig_ts, use_container_width=True)


# ====================================================================
# PAGE: Preprocessing
# ====================================================================
elif page == "Preprocessing":
    st.markdown('<div class="section-header"><h2>Data Preprocessing Pipeline</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Pipeline Steps**
            1. Engineered `Energy_Index = 0.4*Machine_Speed + 0.3*Compression_Force + 0.3*Drying_Temp`
            2. Defined 6 target variables (quality + energy)
            3. Dropped `Batch_ID` and targets from feature matrix
            4. Applied `SelectKBest` (f-regression, k=6) for feature selection
            5. 80/20 train-test split (random_state=42)
            """
        )

        st.markdown("**Selected Features (6/9)**")
        for f in FEATURE_ORDER:
            st.markdown(f"- `{f}`")

    with col2:
        # Feature selection scores
        X_all = data.drop(columns=["Batch_ID"] + [t for t in TARGETS if t in data.columns])
        selector = SelectKBest(score_func=f_regression, k=6)
        selector.fit(X_all, data[TARGETS].mean(axis=1))
        scores = pd.Series(selector.scores_, index=X_all.columns).sort_values(ascending=True)
        fig_sel = go.Figure(
            go.Bar(
                x=scores.values,
                y=scores.index,
                orientation="h",
                marker_color=[
                    COLORS["primary"] if c in FEATURE_ORDER else COLORS["text_muted"]
                    for c in scores.index
                ],
            )
        )
        fig_sel.update_layout(
            **PLOTLY_LAYOUT,
            title="Feature Selection Scores (f-regression)",
            xaxis_title="F-Score",
            height=360,
        )
        st.plotly_chart(fig_sel, use_container_width=True)

    # Box plot of features
    st.markdown('<div class="section-header"><h2>Feature Distributions (Box Plots)</h2></div>', unsafe_allow_html=True)

    fig_box = go.Figure()
    for i, f in enumerate(FEATURE_ORDER):
        fig_box.add_trace(
            go.Box(
                y=art["full_X"][f],
                name=f,
                marker_color=COLORS["palette"][i % len(COLORS["palette"])],
            )
        )
    fig_box.update_layout(**PLOTLY_LAYOUT, height=400, title="Input Feature Distributions")
    st.plotly_chart(fig_box, use_container_width=True)


# ====================================================================
# PAGE: Model Performance
# ====================================================================
elif page == "Model Performance":
    st.markdown('<div class="section-header"><h2>Model Performance Metrics</h2></div>', unsafe_allow_html=True)

    # Metric cards
    cols = st.columns(3)
    for i, t in enumerate(TARGETS):
        m = art["metrics"][t]
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-value">{m['r2']:.4f}</div>
                    <div class="kpi-label">R\u00b2 &mdash; {t}</div>
                    <div style="color:#94a3b8; font-size:0.75rem; margin-top:6px;">
                        MAE: {m['mae']:.3f} &nbsp;|&nbsp; RMSE: {m['rmse']:.3f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")

    # Actual vs Predicted
    st.markdown('<div class="section-header"><h2>Actual vs Predicted</h2></div>', unsafe_allow_html=True)

    sel_target = st.selectbox("Target", TARGETS, index=0)
    idx = TARGETS.index(sel_target)
    y_actual = art["y_test"].iloc[:, idx]
    y_pred = art["preds"][:, idx]

    fig_ap = go.Figure()
    fig_ap.add_trace(
        go.Scatter(
            x=y_actual,
            y=y_pred,
            mode="markers",
            marker=dict(color=COLORS["primary"], size=10, opacity=0.8,
                        line=dict(width=1, color="#fff")),
            name="Predictions",
            hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
        )
    )
    mn, mx = min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())
    fig_ap.add_trace(
        go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines",
            line=dict(color=COLORS["accent"], dash="dash"),
            name="Perfect prediction",
        )
    )
    fig_ap.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Actual vs Predicted -- {sel_target} (R\u00b2 = {art['metrics'][sel_target]['r2']:.4f})",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        height=450,
    )
    st.plotly_chart(fig_ap, use_container_width=True)

    # Residual plot
    residuals = y_pred - y_actual.values
    fig_res = go.Figure()
    fig_res.add_trace(
        go.Bar(
            x=list(range(len(residuals))),
            y=residuals,
            marker_color=[COLORS["success"] if r >= 0 else COLORS["danger"] for r in residuals],
        )
    )
    fig_res.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Residuals -- {sel_target}",
        xaxis_title="Test Sample",
        yaxis_title="Residual",
        height=300,
    )
    st.plotly_chart(fig_res, use_container_width=True)

    # Feature importance
    st.markdown('<div class="section-header"><h2>Feature Importance</h2></div>', unsafe_allow_html=True)

    imp = art["importances"][sel_target]
    imp_s = pd.Series(imp).sort_values(ascending=True)
    fig_imp = go.Figure(
        go.Bar(
            x=imp_s.values,
            y=imp_s.index,
            orientation="h",
            marker=dict(
                color=imp_s.values,
                colorscale=[[0, COLORS["accent"]], [1, COLORS["primary"]]],
            ),
        )
    )
    fig_imp.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Feature Importance -- {sel_target}",
        xaxis_title="Importance",
        height=350,
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Cross-validation
    st.markdown('<div class="section-header"><h2>Cross-Validation (Dissolution Rate)</h2></div>', unsafe_allow_html=True)

    fig_cv = go.Figure()
    fig_cv.add_trace(
        go.Bar(
            x=[f"Fold {i+1}" for i in range(len(art["cv_scores"]))],
            y=art["cv_scores"],
            marker_color=COLORS["palette"][:len(art["cv_scores"])],
            text=[f"{s:.4f}" for s in art["cv_scores"]],
            textposition="outside",
        )
    )
    fig_cv.add_hline(
        y=art["cv_scores"].mean(),
        line_dash="dash",
        line_color=COLORS["warning"],
        annotation_text=f"Mean: {art['cv_scores'].mean():.4f}",
    )
    fig_cv.update_layout(**PLOTLY_LAYOUT, title=f"{len(art['cv_scores'])}-Fold Cross-Validation R\u00b2", height=350)
    st.plotly_chart(fig_cv, use_container_width=True)


# ====================================================================
# PAGE: Predict & Simulate
# ====================================================================
elif page == "Predict & Simulate":
    st.markdown('<div class="section-header"><h2>Predict Tablet Quality</h2></div>', unsafe_allow_html=True)

    st.markdown("Adjust the process parameters below to get real-time quality predictions.")

    col1, col2, col3 = st.columns(3)
    params = {}
    with col1:
        params["Granulation_Time"] = st.slider("Granulation Time (min)", 5.0, 25.0, 15.0, 0.5)
        params["Binder_Amount"] = st.slider("Binder Amount (%)", 2.0, 12.0, 8.0, 0.1)
    with col2:
        params["Drying_Temp"] = st.slider("Drying Temp (\u00b0C)", 40.0, 80.0, 60.0, 1.0)
        params["Drying_Time"] = st.slider("Drying Time (min)", 10.0, 45.0, 25.0, 1.0)
    with col3:
        params["Compression_Force"] = st.slider("Compression Force (kN)", 5.0, 20.0, 12.0, 0.5)
        params["Moisture_Content"] = st.slider("Moisture Content (%)", 0.5, 5.0, 2.0, 0.1)

    pred = forecast(model, params)

    # Results
    st.markdown("")
    pred_cols = st.columns(3)
    for i, (t, v) in enumerate(pred.items()):
        with pred_cols[i % 3]:
            st.metric(t, f"{v:.2f}")

    # Risk detection
    risks = detect_risk(pred)
    for msg, level in risks:
        if level == "success":
            st.success(msg)
        elif level == "warning":
            st.warning(msg)
        else:
            st.error(msg)

    # Scenario comparison
    st.markdown('<div class="section-header"><h2>Scenario Comparison</h2></div>', unsafe_allow_html=True)
    st.markdown("Adjust the **scenario** parameters and compare against the current prediction above.")

    col1b, col2b, col3b = st.columns(3)
    params2 = {}
    with col1b:
        params2["Granulation_Time"] = st.slider("Scenario: Granulation Time", 5.0, 25.0, 15.0, 0.5, key="s_gt")
        params2["Binder_Amount"] = st.slider("Scenario: Binder Amount", 2.0, 12.0, 8.0, 0.1, key="s_ba")
    with col2b:
        params2["Drying_Temp"] = st.slider("Scenario: Drying Temp", 40.0, 80.0, 65.0, 1.0, key="s_dt")
        params2["Drying_Time"] = st.slider("Scenario: Drying Time", 10.0, 45.0, 25.0, 1.0, key="s_dtm")
    with col3b:
        params2["Compression_Force"] = st.slider("Scenario: Compression Force", 5.0, 20.0, 12.0, 0.5, key="s_cf")
        params2["Moisture_Content"] = st.slider("Scenario: Moisture Content", 0.5, 5.0, 2.0, 0.1, key="s_mc")

    pred2 = forecast(model, params2)

    # Side-by-side bar chart
    fig_cmp = go.Figure()
    fig_cmp.add_trace(
        go.Bar(
            name="Current",
            x=TARGETS,
            y=[pred[t] for t in TARGETS],
            marker_color=COLORS["primary"],
        )
    )
    fig_cmp.add_trace(
        go.Bar(
            name="Scenario",
            x=TARGETS,
            y=[pred2[t] for t in TARGETS],
            marker_color=COLORS["accent"],
        )
    )
    fig_cmp.update_layout(**PLOTLY_LAYOUT, barmode="group", title="Current vs Scenario", height=400)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Impact table
    impact = {t: pred2[t] - pred[t] for t in TARGETS}
    impact_df = pd.DataFrame(
        {"Target": TARGETS, "Current": [pred[t] for t in TARGETS],
         "Scenario": [pred2[t] for t in TARGETS], "Delta": [impact[t] for t in TARGETS]}
    )
    st.dataframe(impact_df.style.format({"Current": "{:.2f}", "Scenario": "{:.2f}", "Delta": "{:+.2f}"}).applymap(
        lambda v: "color: #10b981" if isinstance(v, (int, float)) and v > 0 else "color: #ef4444" if isinstance(v, (int, float)) and v < 0 else "",
        subset=["Delta"],
    ), use_container_width=True)


# ====================================================================
# PAGE: Optimizer
# ====================================================================
elif page == "Optimizer":
    st.markdown('<div class="section-header"><h2>Parameter Optimizer</h2></div>', unsafe_allow_html=True)

    st.markdown(
        "Runs a randomized search to find parameters that **maximize Dissolution Rate** "
        "while **minimizing Energy Index** (objective = Dissolution - 0.1 * Energy)."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        n_trials = st.slider("Optimization trials", 100, 2000, 500, 100)
    with col2:
        run_opt = st.button("Run Optimization", type="primary", use_container_width=True)

    if run_opt:
        current = {f: art["full_X"][f].mean() for f in FEATURE_ORDER}
        with st.spinner("Optimizing..."):
            best_params, best_pred = recommend_params(model, current, trials=n_trials)

        st.markdown("### Recommended Parameters")
        bp_cols = st.columns(3)
        for i, (k, v) in enumerate(best_params.items()):
            with bp_cols[i % 3]:
                st.metric(k, f"{v:.2f}")

        st.markdown("### Predicted Quality")
        bq_cols = st.columns(3)
        for i, (k, v) in enumerate(best_pred.items()):
            with bq_cols[i % 3]:
                st.metric(k, f"{v:.2f}")

        risks = detect_risk(best_pred)
        for msg, level in risks:
            if level == "success":
                st.success(msg)
            elif level == "warning":
                st.warning(msg)
            else:
                st.error(msg)

    # Pareto front
    st.markdown('<div class="section-header"><h2>Pareto Front: Dissolution vs Energy</h2></div>', unsafe_allow_html=True)

    solutions = []
    for temp in range(50, 75):
        for mc in np.linspace(1.0, 4.0, 8):
            p = {
                "Granulation_Time": 15,
                "Binder_Amount": 8,
                "Drying_Temp": float(temp),
                "Drying_Time": 25,
                "Compression_Force": 12,
                "Moisture_Content": float(mc),
            }
            pr = forecast(model, p)
            solutions.append({
                "Drying_Temp": temp,
                "Moisture_Content": mc,
                "Dissolution": pr["Dissolution_Rate"],
                "Energy": pr["Energy_Index"],
            })

    sol_df = pd.DataFrame(solutions)

    # Identify pareto front
    pareto = []
    for i, row in sol_df.iterrows():
        dominated = False
        for j, row2 in sol_df.iterrows():
            if (
                row2["Dissolution"] >= row["Dissolution"]
                and row2["Energy"] <= row["Energy"]
                and (row2["Dissolution"] > row["Dissolution"] or row2["Energy"] < row["Energy"])
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(row)
    pareto_df = pd.DataFrame(pareto) if pareto else pd.DataFrame()

    fig_par = go.Figure()
    fig_par.add_trace(
        go.Scatter(
            x=sol_df["Energy"],
            y=sol_df["Dissolution"],
            mode="markers",
            marker=dict(color=COLORS["primary"], opacity=0.3, size=6),
            name="All solutions",
            hovertemplate="Energy: %{x:.1f}<br>Dissolution: %{y:.2f}<extra></extra>",
        )
    )
    if not pareto_df.empty:
        par_sorted = pareto_df.sort_values("Energy")
        fig_par.add_trace(
            go.Scatter(
                x=par_sorted["Energy"],
                y=par_sorted["Dissolution"],
                mode="markers+lines",
                marker=dict(color=COLORS["danger"], size=10, symbol="diamond"),
                line=dict(color=COLORS["danger"], width=2),
                name="Pareto front",
                hovertemplate="Energy: %{x:.1f}<br>Dissolution: %{y:.2f}<extra></extra>",
            )
        )
    fig_par.update_layout(
        **PLOTLY_LAYOUT,
        title="Pareto Optimization: Dissolution Rate vs Energy Index",
        xaxis_title="Energy Index",
        yaxis_title="Dissolution Rate (%)",
        height=480,
    )
    st.plotly_chart(fig_par, use_container_width=True)


# ====================================================================
# PAGE: 3D Explorer
# ====================================================================
elif page == "3D Explorer":
    st.markdown('<div class="section-header"><h2>3D Interactive Explorer</h2></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x_ax = st.selectbox("X axis", FEATURE_ORDER, index=2)
    with c2:
        y_ax = st.selectbox("Y axis", FEATURE_ORDER, index=5)
    with c3:
        z_ax = st.selectbox("Z axis (target)", TARGETS, index=0)
    with c4:
        color_by = st.selectbox("Color by", TARGETS, index=4)

    # 3D surface from grid prediction
    st.markdown('<div class="section-header"><h2>3D Prediction Surface</h2></div>', unsafe_allow_html=True)

    x_range = np.linspace(FEATURE_RANGES[x_ax][0], FEATURE_RANGES[x_ax][1], 25)
    y_range = np.linspace(FEATURE_RANGES[y_ax][0], FEATURE_RANGES[y_ax][1], 25)
    xx, yy = np.meshgrid(x_range, y_range)

    grid_params = []
    defaults = {f: art["full_X"][f].mean() for f in FEATURE_ORDER}
    for xi, yi in zip(xx.ravel(), yy.ravel()):
        row = defaults.copy()
        row[x_ax] = xi
        row[y_ax] = yi
        grid_params.append(row)

    grid_df = pd.DataFrame(grid_params)[FEATURE_ORDER]
    grid_preds = model.predict(grid_df)
    z_idx = TARGETS.index(z_ax)
    zz = grid_preds[:, z_idx].reshape(xx.shape)

    fig_surf = go.Figure(
        data=[
            go.Surface(
                x=xx,
                y=yy,
                z=zz,
                colorscale="Viridis",
                colorbar=dict(title=z_ax),
                opacity=0.85,
                hovertemplate=f"{x_ax}: %{{x:.1f}}<br>{y_ax}: %{{y:.1f}}<br>{z_ax}: %{{z:.2f}}<extra></extra>",
            )
        ]
    )
    fig_surf.update_layout(
        **PLOTLY_LAYOUT,
        title=f"3D Surface: {z_ax} vs {x_ax} & {y_ax}",
        scene=dict(
            xaxis_title=x_ax,
            yaxis_title=y_ax,
            zaxis_title=z_ax,
            bgcolor="rgba(15,12,41,0.3)",
        ),
        height=550,
    )
    st.plotly_chart(fig_surf, use_container_width=True)

    # 3D scatter of actual data
    st.markdown('<div class="section-header"><h2>3D Data Scatter</h2></div>', unsafe_allow_html=True)

    fig_3d = go.Figure(
        data=[
            go.Scatter3d(
                x=data[x_ax] if x_ax in data.columns else art["full_X"][x_ax],
                y=data[y_ax] if y_ax in data.columns else art["full_X"][y_ax],
                z=data[z_ax],
                mode="markers",
                marker=dict(
                    size=6,
                    color=data[color_by],
                    colorscale="Plasma",
                    colorbar=dict(title=color_by),
                    opacity=0.9,
                    line=dict(width=0.5, color="#fff"),
                ),
                hovertemplate=f"{x_ax}: %{{x:.1f}}<br>{y_ax}: %{{y:.1f}}<br>{z_ax}: %{{z:.2f}}<br>{color_by}: %{{marker.color:.2f}}<extra></extra>",
            )
        ]
    )
    fig_3d.update_layout(
        **PLOTLY_LAYOUT,
        title=f"3D Scatter: {z_ax} coloured by {color_by}",
        scene=dict(
            xaxis_title=x_ax,
            yaxis_title=y_ax,
            zaxis_title=z_ax,
            bgcolor="rgba(15,12,41,0.3)",
        ),
        height=550,
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # Animated 3D - rotating view via frames
    st.markdown('<div class="section-header"><h2>Animated 3D Trends</h2></div>', unsafe_allow_html=True)

    anim_feature = st.selectbox("Animate across", FEATURE_ORDER, index=0, key="anim_feat")
    anim_range = np.linspace(FEATURE_RANGES[anim_feature][0], FEATURE_RANGES[anim_feature][1], 12)

    frames = []
    for val in anim_range:
        p = defaults.copy()
        p[anim_feature] = val
        pr = forecast(model, p)
        frames.append({anim_feature: val, **pr})
    anim_df = pd.DataFrame(frames)

    fig_anim = go.Figure()
    for t in TARGETS:
        fig_anim.add_trace(
            go.Scatter(
                x=anim_df[anim_feature],
                y=anim_df[t],
                mode="lines+markers",
                name=t,
                line=dict(width=2),
                hovertemplate=f"{anim_feature}: %{{x:.1f}}<br>{t}: %{{y:.2f}}<extra></extra>",
            )
        )
    fig_anim.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Target Trends as {anim_feature} Varies",
        xaxis_title=anim_feature,
        yaxis_title="Predicted Value",
        height=450,
    )
    st.plotly_chart(fig_anim, use_container_width=True)


# ====================================================================
# PAGE: How to Use
# ====================================================================
elif page == "How to Use":
    st.markdown('<div class="section-header"><h2>How to Use PharmaDash</h2></div>', unsafe_allow_html=True)

    st.markdown(
        """
### Getting Started

1. **Select a data source** in the sidebar -- use the default dataset or upload your own Excel/CSV file.
2. **Navigate** between pages using the sidebar menu.

### Page Guide

| Page | What It Does |
|------|-------------|
| **Overview** | High-level KPIs and model performance radar |
| **Dataset Insights** | Explore raw data, distributions, correlations, and time-series sensor data |
| **Preprocessing** | View the feature engineering & selection pipeline |
| **Model Performance** | Detailed R\u00b2, MAE, RMSE, actual-vs-predicted charts, residuals, feature importance, cross-validation |
| **Predict & Simulate** | Adjust sliders to predict quality; compare two scenarios side-by-side |
| **Optimizer** | Automated parameter search to maximize dissolution and minimize energy |
| **3D Explorer** | Interactive 3D surfaces, scatter plots, and animated trend lines |

### Uploading Custom Data

Your uploaded file must contain these columns (same as the default dataset):

```
Batch_ID, Granulation_Time, Binder_Amount, Drying_Temp, Drying_Time,
Compression_Force, Machine_Speed, Lubricant_Conc, Moisture_Content,
Tablet_Weight, Hardness, Friability, Disintegration_Time,
Dissolution_Rate, Content_Uniformity
```

The model will retrain automatically on your data.

### Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py
```

### Deploying

**Streamlit Cloud**
1. Push this folder to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set `dashboard/app.py` as the main file.

**HuggingFace Spaces**
1. Create a new Space (SDK: Streamlit).
2. Upload `app.py`, `requirements.txt`, and data files.

**Render / Vercel**
1. Use a `Dockerfile` or set the start command to `streamlit run app.py --server.port $PORT`.

### Input Parameter Ranges

| Parameter | Min | Max | Unit |
|-----------|-----|-----|------|
| Granulation Time | 5 | 25 | min |
| Binder Amount | 2 | 12 | % |
| Drying Temp | 40 | 80 | \u00b0C |
| Drying Time | 10 | 45 | min |
| Compression Force | 5 | 20 | kN |
| Moisture Content | 0.5 | 5.0 | % |

### Model Details

- **Algorithm**: Multi-output XGBoost Regressor
- **Hyperparameters**: 300 estimators, learning rate 0.05, max depth 4
- **Feature selection**: SelectKBest with f-regression (top 6)
- **Validation**: 5-fold cross-validation, 80/20 train-test split
        """
    )
