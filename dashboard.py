import pickle
from models.decision_tree import evaluate_decision_tree
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from utils import plot_histogram, plot_boxplot, plot_scatter
import pickle
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide", page_title="Patient Classification Dashboard")

st.markdown("""
<style>
    .block-container { padding: 1rem 1.2rem; }
    div[data-testid="column"] > div {
        background: var(--background-color);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 12px;
        padding: 14px 16px;
        height: 100%;
    }
    .panel-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: gray;
        margin-bottom: 8px;
    }
    .metric-row {
        display: flex;
        gap: 8px;
        margin-top: 6px;
    }
    .metric-box {
        flex: 1;
        background: rgba(128,128,128,0.08);
        border-radius: 8px;
        padding: 8px 10px;
    }
    .metric-box .mlabel { font-size: 10px; color: gray; }
    .metric-box .mvalue { font-size: 20px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


#load dataframes cached
@st.cache_data
def load_dataframes():
    historical_df = pd.read_csv("data/historical_data.csv")
    current_df    = pd.read_csv("data/current_data.csv")
    return historical_df, current_df

historical_df, current_df = load_dataframes()
_, _, _, historical_df = train_test_split(historical_df, historical_df["HIGH_EMERGENCY_RISK"], test_size=0.2, random_state=42)
_, _, _, current_df    = train_test_split(current_df, current_df["HIGH_EMERGENCY_RISK"], test_size=0.2, random_state=42)

COLUMNS = current_df.columns.drop("PATIENT").tolist()

def get_model_results(model_name: str) -> dict:
    """
    REPLACE THIS with your real model results.
    Must return:
        {
            "confusion_matrix": [[TP, FN], [FP, TN]],
            "accuracy":  float,
            "precision": float,
            "recall":    float,
            "f1":        float,
        }
    """
    model_filepath_map = {
        "Decision Tree Historical": 'saved_models/best_decision_tree_historical.pkl',
        "SVC Historical":           'saved_models/best_svc_historical.pkl',
        "MLP Historical":           'saved_models/best_mlp_historical.pkl',
        "Decision Tree Current":    'saved_models/finetuned_decision_tree_current.pkl',
        "SVC Current":              'saved_models/finetuned_svc_current.pkl',
        "MLP Current":              'saved_models/finetuned_mlp_current.pkl',
    }

    model = pickle.load(open(model_filepath_map.get(model_name, ''), 'rb')) if model_name in model_filepath_map else None
    df = historical_df if "Historical" in model_name else current_df
    if model_name=="Decision Tree Historical":
        matrix, report = evaluate_decision_tree(model, df.drop(columns=["EMERGENCY_ENCOUNTERS", "HIGH_EMERGENCY_RISK", "PATIENT"]), df["HIGH_EMERGENCY_RISK"]) 
        return {
            "confusion_matrix": matrix,
            "accuracy": report["accuracy"],
            "precision": report['macro avg']["precision"],
            "recall": report['macro avg']["recall"],
            "f1": report['macro avg']["f1-score"],
        }
    elif model_name=="SVC Historical":
        matrix, report = evaluate_decision_tree(model, df.drop(columns=["EMERGENCY_ENCOUNTERS", "HIGH_EMERGENCY_RISK", "PATIENT"]), df["HIGH_EMERGENCY_RISK"]) 
        return {
            "confusion_matrix": matrix,
            "accuracy": report["accuracy"],
            "precision": report['macro avg']["precision"],
            "recall": report['macro avg']["recall"],
            "f1": report['macro avg']["f1-score"],
        }
    elif model_name=="MLP Historical":
        matrix, report = evaluate_decision_tree(model, df.drop(columns=["EMERGENCY_ENCOUNTERS", "HIGH_EMERGENCY_RISK", "PATIENT"]), df["HIGH_EMERGENCY_RISK"]) 
        return {
            "confusion_matrix": matrix,
            "accuracy": report["accuracy"],
            "precision": report['macro avg']["precision"],
            "recall": report['macro avg']["recall"],
            "f1": report['macro avg']["f1-score"],
        }
    elif model_name=="Decision Tree Current":
        matrix, report = evaluate_decision_tree(model, df.drop(columns=["EMERGENCY_ENCOUNTERS", "HIGH_EMERGENCY_RISK", "PATIENT"]), df["HIGH_EMERGENCY_RISK"]) 
        return {
            "confusion_matrix": matrix,
            "accuracy": report["accuracy"],
            "precision": report['macro avg']["precision"],
            "recall": report['macro avg']["recall"],
            "f1": report['macro avg']["f1-score"],
        }
    elif model_name=="SVC Current":
        matrix, report = evaluate_decision_tree(model, df.drop(columns=["EMERGENCY_ENCOUNTERS", "HIGH_EMERGENCY_RISK", "PATIENT"]), df["HIGH_EMERGENCY_RISK"]) 
        return {
            "confusion_matrix": matrix,
            "accuracy": report["accuracy"],
            "precision": report['macro avg']["precision"],
            "recall": report['macro avg']["recall"],
            "f1": report['macro avg']["f1-score"],
        }
    elif model_name=="MLP Current":
        matrix, report = evaluate_decision_tree(model, df.drop(columns=["EMERGENCY_ENCOUNTERS", "HIGH_EMERGENCY_RISK", "PATIENT"]), df["HIGH_EMERGENCY_RISK"]) 
        return {
            "confusion_matrix": matrix,
            "accuracy": report["accuracy"],
            "precision": report['macro avg']["precision"],
            "recall": report['macro avg']["recall"],
            "f1": report['macro avg']["f1-score"],
        }

    return model_filepath_map.get(model_name, {})


def get_plot_figure(plot_type: str, dataset: str, col1: str, col2: str = None) -> go.Figure:
    """
    REPLACE THIS with your real dataframes and plotting logic.
    Parameters:
        plot_type : "Histogram" | "Box" | "Scatter"
        dataset   : "Historical" | "Current"
        col1      : column name string
        col2      : column name string (only for Scatter)
    Must return a plotly Figure.
    """

    df = historical_df if dataset == "Historical" else current_df

    color = "#1D9E75" if dataset == "Historical" else "#378ADD"
    #fig = go.Figure()

    if plot_type == "Histogram":
        fig = plot_histogram(df, col1)
        fig.update_layout(xaxis_title=col1, yaxis_title="Count")

    elif plot_type == "Box":
        fig = plot_boxplot(df, col1)
        fig.update_layout(yaxis_title=col1)

    elif plot_type == "Scatter" and col2:
        fig = plot_scatter(df, col1, col2)
        fig.update_layout(xaxis_title=col1, yaxis_title=col2)

    fig.update_layout(
        margin=dict(t=20, r=10, b=40, l=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
        showlegend=False,
        height=260,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


# ─────────────────────────────────────────────
# CONFUSION MATRIX HELPER
# ─────────────────────────────────────────────

def plot_confusion_matrix(cm: list) -> go.Figure:
    [[tp, fn], [fp, tn]] = cm
    z     = [[tp, fn], [fp, tn]]
    text  = [["TP", "FN"], ["FP", "TN"]]
    vals  = [[f"<b>{tp}</b><br><sup>TP</sup>", f"<b>{fn}</b><br><sup>FN</sup>"],
             [f"<b>{fp}</b><br><sup>FP</sup>", f"<b>{tn}</b><br><sup>TN</sup>"]]

    colorscale = [
        [0.0, "#FAECE7"], [0.49, "#FAECE7"],
        [0.5, "#E1F5EE"], [1.0,  "#E1F5EE"],
    ]

    fig = go.Figure(go.Heatmap(
        z=[[1, 0], [0, 1]],
        text=vals,
        texttemplate="%{text}",
        textfont=dict(size=16),
        colorscale=colorscale,
        showscale=False,
        xgap=4, ygap=4,
    ))
    fig.update_layout(
        xaxis=dict(tickvals=[0, 1], ticktext=["Predicted Positive", "Predicted Negative"], side="top"),
        yaxis=dict(tickvals=[0, 1], ticktext=["Actual Positive", "Actual Negative"], autorange="reversed"),
        margin=dict(t=40, r=10, b=10, l=120),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200,
        font=dict(size=11),
    )
    return fig


# ─────────────────────────────────────────────
# METRIC CARDS HELPER
# ─────────────────────────────────────────────

def render_metrics(accuracy, precision, recall, f1):
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <div class="mlabel">Accuracy</div>
            <div class="mvalue">{accuracy*100:.1f}%</div>
        </div>
        <div class="metric-box">
            <div class="mlabel">Precision</div>
            <div class="mvalue">{precision*100:.1f}%</div>
        </div>
        <div class="metric-box">
            <div class="mlabel">Recall</div>
            <div class="mvalue">{recall*100:.1f}%</div>
        </div>
        <div class="metric-box">
            <div class="mlabel">F1 score</div>
            <div class="mvalue">{f1*100:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PLOT PANEL HELPER
# ─────────────────────────────────────────────

def render_plot_panel(panel_id: str):
    st.markdown(f'<div class="panel-label">Plot {panel_id} — exploratory analysis</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.2])
    with c1:
        plot_type = st.selectbox("Plot type", ["", "Histogram", "Box", "Scatter"],
                                 key=f"plot_type_{panel_id}", label_visibility="collapsed")
        st.caption("Plot type")
    with c2:
        dataset = st.selectbox("Dataset", ["", "Historical", "Current"],
                               key=f"dataset_{panel_id}", label_visibility="collapsed")
        st.caption("Dataset")
    with c3:
        col1 = st.selectbox("Column 1", [""] + COLUMNS,
                            key=f"col1_{panel_id}", label_visibility="collapsed")
        st.caption("Column 1")
    with c4:
        scatter_active = plot_type == "Scatter"
        col2_opts = [""] + COLUMNS if scatter_active else ["— scatter only —"]
        col2 = st.selectbox("Column 2", col2_opts,
                            key=f"col2_{panel_id}",
                            disabled=not scatter_active,
                            label_visibility="collapsed")
        st.caption("Column 2")

    ready = (plot_type and dataset and col1 and
             (plot_type != "Scatter" or (col2 and col2 != "— scatter only —")))

    if ready:
        fig = get_plot_figure(plot_type, dataset, col1,
                              col2 if plot_type == "Scatter" else None)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown(
            "<div style='height:200px;display:flex;align-items:center;justify-content:center;"
            "border:1px dashed rgba(128,128,128,0.3);border-radius:8px;"
            "color:gray;font-size:12px;'>Select plot type, dataset and column</div>",
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────
# MODEL PANEL HELPER
# ─────────────────────────────────────────────

def render_model_panel(panel_id: str):
    label = "A" if panel_id == "1" else "B"
    st.markdown(f'<div class="panel-label">Model {label} — confusion matrix &amp; metrics</div>',
                unsafe_allow_html=True)

    model = st.selectbox(
        "Model", ["", "Decision Tree Historical", "SVC Historical", "MLP Historical", "Decision Tree Current", "SVC Current", "MLP Current"],
        key=f"model_{panel_id}", label_visibility="collapsed"
    )
    st.caption("Select model")

    if model:
        result = get_model_results(model)
        if result:
            cm_fig = plot_confusion_matrix(result["confusion_matrix"])
            st.plotly_chart(cm_fig, use_container_width=True, config={"displayModeBar": False})
            render_metrics(result["accuracy"], result["precision"],
                           result["recall"], result["f1"])
    else:
        st.markdown(
            "<div style='height:180px;display:flex;align-items:center;justify-content:center;"
            "border:1px dashed rgba(128,128,128,0.3);border-radius:8px;"
            "color:gray;font-size:12px;'>Select a model to view confusion matrix</div>",
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────
# LAYOUT  —  left 2fr | right 3fr
# ─────────────────────────────────────────────

top_left, top_right = st.columns([2, 3])
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
bot_left, bot_right = st.columns([2, 3])

with top_left:
    render_model_panel("1")

with top_right:
    render_plot_panel("A")

with bot_left:
    render_model_panel("2")

with bot_right:
    render_plot_panel("B")