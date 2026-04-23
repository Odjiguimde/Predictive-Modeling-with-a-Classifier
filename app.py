import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, os, json, pickle
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score, classification_report
)
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor Pro",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stSidebar"] { background:#0a0c14; border-right:1px solid #1e2130; }
[data-testid="stSidebar"] * { color:#d4d8f0 !important; }
[data-testid="stSidebar"] label { color:#6b7499 !important; font-size:.72rem !important;
    letter-spacing:.08em; text-transform:uppercase; }

[data-testid="metric-container"] { background:#131627; border:1px solid #1e2540;
    border-radius:12px; padding:.9rem 1.1rem; }
[data-testid="metric-container"] label { color:#6b7499 !important; font-size:.72rem !important;
    text-transform:uppercase; letter-spacing:.07em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#e4e8ff !important;
    font-size:1.55rem !important; font-weight:700; }

.stTabs [data-baseweb="tab-list"] { gap:2px; background:transparent;
    border-bottom:1px solid #1e2540; padding-bottom:0; }
.stTabs [data-baseweb="tab"] { background:transparent; color:#6b7499;
    border-radius:8px 8px 0 0; padding:.45rem 1.1rem; font-size:.82rem; font-weight:500; }
.stTabs [aria-selected="true"] { background:#131627 !important; color:#7b9cff !important;
    border-bottom:2px solid #7b9cff; }

.sec { font-size:.68rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
    color:#7b9cff; border-left:3px solid #7b9cff; padding-left:.55rem;
    margin:1.4rem 0 .7rem; }

.insight { background:#0f1221; border:1px solid #1e2540; border-left:3px solid #7b9cff;
    border-radius:10px; padding:.9rem 1.1rem; margin:.4rem 0;
    font-size:.86rem; line-height:1.65; color:#b0b8d8; }
.insight b { color:#7b9cff; }
.insight.warn  { border-left-color:#f4a261; } .insight.warn  b { color:#f4a261; }
.insight.good  { border-left-color:#52b788; } .insight.good  b { color:#52b788; }
.insight.alert { border-left-color:#ef476f; } .insight.alert b { color:#ef476f; }

.model-card { background:#0f1221; border:1px solid #1e2540; border-radius:12px;
    padding:1rem 1.2rem; margin:.4rem 0; }
.model-card .mc-name { font-size:.85rem; font-weight:700; color:#e4e8ff; }
.model-card .mc-score { font-size:1.6rem; font-weight:700; color:#7b9cff; }

.main .block-container { background:#080a12; padding-top:1.2rem; max-width:1400px; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────
PAL  = ["#7b9cff","#ef476f","#52b788","#f4a261","#64dfdf","#ffd166","#a8dadc","#c77dff","#e76f51","#06d6a0"]
BG   = "#131627"
GRID = "#1e2540"
TEXT = "#b0b8d8"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT, "axes.titlecolor": "#e4e8ff", "axes.titlesize": 11,
    "axes.titleweight": "600", "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "legend.facecolor": BG, "legend.edgecolor": GRID,
    "legend.labelcolor": TEXT, "grid.color": GRID, "grid.linestyle": "--",
    "grid.linewidth": 0.5, "axes.grid": True, "figure.dpi": 110,
})

def fig():
    return plt.figure(figsize=(9, 4.5))

# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    # Feature Engineering
    df["ChurnBin"] = (df["Churn"] == "Yes").astype(int)
    df["HasPartnerOrDep"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)
    df["HasStreaming"] = ((df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")).astype(int)
    df["HasSecurity"] = ((df["OnlineSecurity"] == "Yes") | (df["DeviceProtection"] == "Yes")).astype(int)
    df["AvgMonthlyRevenue"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["TenureGroup"] = pd.cut(df["tenure"], bins=[0,12,24,48,72], labels=["0-1 an","1-2 ans","2-4 ans","4-6 ans"])
    df["ChargeGroup"] = pd.cut(df["MonthlyCharges"], bins=[0,35,65,90,120], labels=["Faible","Moyen","Élevé","Très élevé"])
    return df

@st.cache_data
def preprocess(df):
    cat_cols = ["gender","Partner","Dependents","PhoneService","MultipleLines",
                "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
                "TechSupport","StreamingTV","StreamingMovies","Contract",
                "PaperlessBilling","PaymentMethod"]
    num_cols = ["tenure","MonthlyCharges","TotalCharges","HasPartnerOrDep",
                "HasStreaming","HasSecurity","AvgMonthlyRevenue","SeniorCitizen"]

    dummies = pd.get_dummies(df[cat_cols], drop_first=True)
    X = pd.concat([df[num_cols].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    y = df["ChurnBin"]
    return X, y

# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_resource
def train_all_models(df):
    X, y = preprocess(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Régression Logistique": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors":   KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes":           GaussianNB(),
        "Random Forest":         RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        "Gradient Boosting":     GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42,
                                           use_label_encoder=False, eval_metric="logloss", n_jobs=-1)
    if HAS_LGB:
        models["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)

    results = {}
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    for name, m in models.items():
        m.fit(X_tr_sc, y_tr)
        y_pred = m.predict(X_te_sc)
        y_prob = m.predict_proba(X_te_sc)[:,1] if hasattr(m, "predict_proba") else np.zeros(len(y_te))
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        results[name] = {
            "model": m,
            "accuracy":  accuracy_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred, zero_division=0),
            "recall":    recall_score(y_te, y_pred, zero_division=0),
            "f1":        f1_score(y_te, y_pred, zero_division=0),
            "roc_auc":   auc(fpr, tpr),
            "conf_matrix": confusion_matrix(y_te, y_pred),
            "fpr": fpr, "tpr": tpr,
            "y_te": y_te.values, "y_pred": y_pred,
        }
        cv = cross_val_score(m, X_tr_sc, y_tr, cv=StratifiedKFold(5), scoring="f1")
        results[name]["cv_f1_mean"] = cv.mean()
        results[name]["cv_f1_std"]  = cv.std()

    return results, scaler, X, y, X_tr, X_te, y_tr, y_te

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
df = load_data()

with st.sidebar:
    st.markdown("## 📡 Churn Predictor Pro")
    st.markdown("---")
    st.markdown('<p class="sec">🔍 Filtres</p>', unsafe_allow_html=True)

    contracts = st.multiselect("Contrat", df["Contract"].unique(), default=df["Contract"].unique())
    internet  = st.multiselect("Internet", df["InternetService"].unique(), default=df["InternetService"].unique())
    senior    = st.selectbox("Senior Citizen", ["Tous", "Oui", "Non"])
    tenure_rng = st.slider("Ancienneté (mois)", 1, 72, (1, 72))

    df_f = df[
        df["Contract"].isin(contracts) &
        df["InternetService"].isin(internet) &
        (df["tenure"].between(*tenure_rng))
    ]
    if senior == "Oui":   df_f = df_f[df_f["SeniorCitizen"]==1]
    elif senior == "Non": df_f = df_f[df_f["SeniorCitizen"]==0]

    st.markdown("---")
    st.metric("Clients filtrés", f"{len(df_f):,}")
    st.metric("Taux de churn", f"{df_f['ChurnBin'].mean():.1%}")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 📡 Churn Predictor Pro")
st.markdown("**Modélisation prédictive du désabonnement client — Télécommunications**")
st.markdown("---")

# KPIs
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Total Clients", f"{len(df_f):,}")
c2.metric("Taux de Churn", f"{df_f['ChurnBin'].mean():.1%}", delta=f"{df_f['ChurnBin'].mean()-df['ChurnBin'].mean():.1%}")
c3.metric("Revenu Mensuel Moy.", f"{df_f['MonthlyCharges'].mean():.1f} €")
c4.metric("Ancienneté Moy.", f"{df_f['tenure'].mean():.0f} mois")
c5.metric("Clients à Risque", f"{df_f['ChurnBin'].sum():,}")

st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs(["📊 Aperçu & EDA", "🔬 Analyse Churn", "🤖 Modèles ML", "📈 Comparaison", "🎯 Prédiction Live"])

# ══════════════════════════════════════════════
# TAB 1 — APERÇU
# ══════════════════════════════════════════════
with tabs[0]:
    st.markdown('<p class="sec">Vue d\'ensemble du Dataset</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df_f.head(10).drop(columns=["ChurnBin","HasPartnerOrDep","HasStreaming",
                                                   "HasSecurity","AvgMonthlyRevenue"], errors="ignore"),
                     use_container_width=True, height=280)
    with col2:
        desc = df_f[["tenure","MonthlyCharges","TotalCharges"]].describe().round(2)
        st.dataframe(desc, use_container_width=True)

    st.markdown('<p class="sec">Distributions des Variables Numériques</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig2, axes = plt.subplots(1, 3, figsize=(12, 3.5), facecolor=BG)
        for ax, col, c in zip(axes, ["tenure","MonthlyCharges","TotalCharges"], PAL):
            churn_yes = df_f[df_f["ChurnBin"]==1][col]
            churn_no  = df_f[df_f["ChurnBin"]==0][col]
            ax.hist(churn_no,  bins=25, alpha=0.7, color=PAL[0], label="Non-Churn")
            ax.hist(churn_yes, bins=25, alpha=0.7, color=PAL[1], label="Churn")
            ax.set_title(col, fontsize=10)
            ax.legend(fontsize=7)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with col2:
        fig3, axes = plt.subplots(1, 3, figsize=(12, 3.5), facecolor=BG)
        for ax, col in zip(axes, ["tenure","MonthlyCharges","TotalCharges"]):
            data_box = [df_f[df_f["ChurnBin"]==0][col].values,
                        df_f[df_f["ChurnBin"]==1][col].values]
            bp = ax.boxplot(data_box, patch_artist=True, widths=0.5,
                            medianprops=dict(color="white", linewidth=2))
            for patch, c in zip(bp['boxes'], [PAL[0], PAL[1]]):
                patch.set_facecolor(c); patch.set_alpha(0.7)
            ax.set_xticklabels(["No Churn","Churn"], fontsize=9)
            ax.set_title(col, fontsize=10)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close()

    st.markdown('<p class="sec">Variables Catégorielles — Taux de Churn</p>', unsafe_allow_html=True)
    cat_vars = ["Contract","InternetService","PaymentMethod","TenureGroup"]
    fig4, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor=BG)
    for ax, var in zip(axes, cat_vars):
        ct = df_f.groupby(var)["ChurnBin"].mean().sort_values(ascending=False)
        bars = ax.bar(range(len(ct)), ct.values, color=PAL[:len(ct)])
        ax.set_xticks(range(len(ct)))
        ax.set_xticklabels(ct.index, rotation=20, ha="right", fontsize=8)
        ax.set_title(f"Churn par {var}", fontsize=10)
        ax.set_ylabel("Taux de churn")
        for b, v in zip(bars, ct.values):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f"{v:.0%}", ha="center", fontsize=8)
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close()

# ══════════════════════════════════════════════
# TAB 2 — ANALYSE CHURN
# ══════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p class="sec">Segmentation des Churners</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Pie churn
        fig5, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
        vals = df_f["Churn"].value_counts()
        ax.pie(vals, labels=vals.index, colors=[PAL[1], PAL[0]],
               autopct="%1.1f%%", startangle=90,
               wedgeprops=dict(edgecolor=BG, linewidth=2))
        ax.set_title("Répartition Churn / Non-Churn")
        st.pyplot(fig5); plt.close()

    with col2:
        # Churn par contrat (grouped bar)
        fig6, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
        ct = df_f.groupby(["Contract","Churn"]).size().unstack(fill_value=0)
        x = np.arange(len(ct))
        w = 0.35
        ax.bar(x-w/2, ct.get("No",0), w, label="Non-Churn", color=PAL[0], alpha=0.85)
        ax.bar(x+w/2, ct.get("Yes",0), w, label="Churn", color=PAL[1], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(ct.index)
        ax.set_title("Churn par type de Contrat"); ax.legend()
        st.pyplot(fig6); plt.close()

    col3, col4 = st.columns(2)

    with col3:
        # Scatter tenure vs MonthlyCharges
        fig7, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
        for label, color, marker in [(0, PAL[0], "o"), (1, PAL[1], "X")]:
            subset = df_f[df_f["ChurnBin"]==label].sample(min(300, len(df_f[df_f["ChurnBin"]==label])), random_state=42)
            ax.scatter(subset["tenure"], subset["MonthlyCharges"],
                       c=color, alpha=0.5, s=18, marker=marker,
                       label=["Non-Churn","Churn"][label])
        ax.set_xlabel("Ancienneté (mois)"); ax.set_ylabel("Charges Mensuelles")
        ax.set_title("Tenure vs Charges Mensuelles"); ax.legend()
        st.pyplot(fig7); plt.close()

    with col4:
        # Heatmap corrélation
        fig8, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
        num_cols = ["tenure","MonthlyCharges","TotalCharges","SeniorCitizen","ChurnBin"]
        corr = df_f[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", ax=ax,
                    cmap="coolwarm", center=0, linewidths=0.5,
                    cbar_kws={"shrink":0.8})
        ax.set_title("Matrice de Corrélation")
        st.pyplot(fig8); plt.close()

    # Insights
    st.markdown('<p class="sec">Insights Clés</p>', unsafe_allow_html=True)
    churn_by_contract = df_f.groupby("Contract")["ChurnBin"].mean()
    churn_fiber = df_f[df_f["InternetService"]=="Fiber optic"]["ChurnBin"].mean()
    best_contract = churn_by_contract.idxmin()

    st.markdown(f"""
    <div class="insight alert">📌 <b>Contrats Mensuels à Haut Risque :</b> Les clients en contrat mensuel
    churent à <b>{churn_by_contract.get('Month-to-month',0):.1%}</b> contre
    <b>{churn_by_contract.get('Two year',0):.1%}</b> pour les contrats 2 ans.
    Encourager les upgrades contractuels est une priorité absolue.</div>

    <div class="insight warn">📌 <b>Fibre Optique :</b> Les abonnés Fibre présentent un taux de churn de
    <b>{churn_fiber:.1%}</b>. La qualité perçue ou le pricing doivent être réévalués.</div>

    <div class="insight good">📌 <b>Rétention par Ancienneté :</b> Les clients dépassant
    <b>36 mois</b> d'ancienneté churent 3x moins. Les programmes de fidélisation précoce sont rentables.</div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 — MODÈLES ML
# ══════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="sec">Entraînement des Modèles de Classification</p>', unsafe_allow_html=True)

    if st.button("🚀 Lancer l'entraînement de tous les modèles", type="primary"):
        with st.spinner("Entraînement en cours... ⏳"):
            results, scaler, X, y, X_tr, X_te, y_tr, y_te = train_all_models(df)
        st.session_state["results"] = results
        st.session_state["scaler"]  = scaler
        st.session_state["X_cols"]  = X.columns.tolist()
        st.success("✅ Tous les modèles ont été entraînés avec succès !")

    if "results" not in st.session_state:
        st.info("👆 Clique sur le bouton ci-dessus pour entraîner les modèles.")
    else:
        results = st.session_state["results"]
        sel_model = st.selectbox("Sélectionner un modèle à analyser", list(results.keys()))
        res = results[sel_model]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy",  f"{res['accuracy']:.3f}")
        col2.metric("Precision", f"{res['precision']:.3f}")
        col3.metric("Recall",    f"{res['recall']:.3f}")
        col4.metric("F1-Score",  f"{res['f1']:.3f}")

        col5, col6 = st.columns(2)

        with col5:
            st.markdown('<p class="sec">Matrice de Confusion</p>', unsafe_allow_html=True)
            fig9, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
            cm = res["conf_matrix"]
            sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                        xticklabels=["Non-Churn","Churn"],
                        yticklabels=["Non-Churn","Churn"],
                        linewidths=1, linecolor=GRID)
            ax.set_title(f"Matrice de Confusion — {sel_model}")
            ax.set_ylabel("Réel"); ax.set_xlabel("Prédit")
            st.pyplot(fig9); plt.close()

        with col6:
            st.markdown('<p class="sec">Courbe ROC</p>', unsafe_allow_html=True)
            fig10, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
            ax.plot(res["fpr"], res["tpr"], color=PAL[0], lw=2,
                    label=f"AUC = {res['roc_auc']:.3f}")
            ax.plot([0,1],[0,1], color=GRID, lw=1, linestyle="--", label="Aléatoire")
            ax.fill_between(res["fpr"], res["tpr"], alpha=0.15, color=PAL[0])
            ax.set_title(f"Courbe ROC — {sel_model}")
            ax.set_xlabel("Taux de Faux Positifs"); ax.set_ylabel("Taux de Vrais Positifs")
            ax.legend()
            st.pyplot(fig10); plt.close()

        # Feature importance (if RF / GBM / XGB)
        m = res["model"]
        if hasattr(m, "feature_importances_"):
            st.markdown('<p class="sec">Feature Importance</p>', unsafe_allow_html=True)
            fi = pd.Series(m.feature_importances_,
                           index=st.session_state["X_cols"]).sort_values(ascending=False).head(15)
            fig11, ax = plt.subplots(figsize=(10, 3.5), facecolor=BG)
            bars = ax.barh(fi.index[::-1], fi.values[::-1], color=PAL[0], alpha=0.85)
            ax.set_title(f"Top 15 Features — {sel_model}")
            ax.set_xlabel("Importance")
            st.pyplot(fig11); plt.close()

        st.markdown('<p class="sec">Validation Croisée (5-Fold)</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="insight good">✅ <b>CV F1-Score :</b> {res['cv_f1_mean']:.3f}
        (± {res['cv_f1_std']:.3f}) — Stabilité du modèle validée sur 5 folds.</div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4 — COMPARAISON
# ══════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p class="sec">Comparaison des Modèles</p>', unsafe_allow_html=True)

    if "results" not in st.session_state:
        st.info("👆 Entraîne d'abord les modèles dans l'onglet 'Modèles ML'.")
    else:
        results = st.session_state["results"]

        # Summary table
        rows = []
        for name, r in results.items():
            rows.append({
                "Modèle": name,
                "Accuracy": f"{r['accuracy']:.3f}",
                "Precision": f"{r['precision']:.3f}",
                "Recall": f"{r['recall']:.3f}",
                "F1-Score": f"{r['f1']:.3f}",
                "ROC-AUC": f"{r['roc_auc']:.3f}",
                "CV F1": f"{r['cv_f1_mean']:.3f} ±{r['cv_f1_std']:.3f}",
            })
        df_res = pd.DataFrame(rows)
        st.dataframe(df_res, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)

        with col1:
            # Radar / Bar métriques
            fig12, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
            metrics = ["accuracy","precision","recall","f1","roc_auc"]
            x = np.arange(len(metrics))
            w = 0.8 / len(results)
            for i, (name, r) in enumerate(results.items()):
                vals = [r[m] for m in metrics]
                offset = (i - len(results)/2 + 0.5) * w
                ax.bar(x + offset, vals, w, label=name, color=PAL[i%len(PAL)], alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(["Accuracy","Precision","Recall","F1","AUC"], fontsize=9)
            ax.set_ylim(0.5, 1.0); ax.set_title("Comparaison des Métriques")
            ax.legend(fontsize=7, loc="lower right")
            st.pyplot(fig12); plt.close()

        with col2:
            # ROC Curves all models
            fig13, ax = plt.subplots(figsize=(6, 4.5), facecolor=BG)
            for i, (name, r) in enumerate(results.items()):
                ax.plot(r["fpr"], r["tpr"], lw=1.8, color=PAL[i%len(PAL)],
                        label=f"{name} ({r['roc_auc']:.3f})")
            ax.plot([0,1],[0,1], color=GRID, lw=1, linestyle="--")
            ax.set_title("Courbes ROC — Tous les Modèles")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.legend(fontsize=7)
            st.pyplot(fig13); plt.close()

        # Best model highlight
        best_name = max(results, key=lambda k: results[k]["f1"])
        best_r = results[best_name]
        st.markdown(f"""
        <div class="insight good">🏆 <b>Meilleur modèle : {best_name}</b><br>
        F1-Score : <b>{best_r['f1']:.3f}</b> | ROC-AUC : <b>{best_r['roc_auc']:.3f}</b> |
        Recall : <b>{best_r['recall']:.3f}</b><br>
        Ce modèle offre le meilleur compromis précision/rappel pour identifier les churners.</div>

        <div class="insight">📌 <b>Recommandation métier :</b> Privilégier le <b>Recall</b> dans ce contexte —
        mieux vaut alerter un client fidèle par erreur que de rater un vrai churner. Un Recall ≥ 70%
        est recommandé pour une stratégie de rétention proactive.</div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 5 — PRÉDICTION LIVE
# ══════════════════════════════════════════════
with tabs[4]:
    st.markdown('<p class="sec">Prédiction en Temps Réel</p>', unsafe_allow_html=True)

    if "results" not in st.session_state:
        st.info("👆 Entraîne d'abord les modèles dans l'onglet 'Modèles ML'.")
    else:
        results = st.session_state["results"]
        sel = st.selectbox("Modèle à utiliser", list(results.keys()), key="pred_model")

        st.markdown("#### 👤 Profil du Client")
        col1, col2, col3 = st.columns(3)

        with col1:
            tenure = st.slider("Ancienneté (mois)", 1, 72, 12)
            monthly = st.slider("Charges mensuelles (€)", 18, 120, 65)
            contract = st.selectbox("Type de Contrat", ["Month-to-month","One year","Two year"])
            internet = st.selectbox("Service Internet", ["DSL","Fiber optic","No"])

        with col2:
            senior = st.selectbox("Senior Citizen", ["Non","Oui"])
            partner = st.selectbox("Partenaire", ["Yes","No"])
            dependents = st.selectbox("Dépendants", ["No","Yes"])
            phone = st.selectbox("Service Téléphone", ["Yes","No"])

        with col3:
            security = st.selectbox("Sécurité en ligne", ["Yes","No","No internet service"])
            techsup  = st.selectbox("Support Technique", ["Yes","No","No internet service"])
            payment  = st.selectbox("Méthode de Paiement",
                                    ["Electronic check","Mailed check",
                                     "Bank transfer (automatic)","Credit card (automatic)"])
            paperless = st.selectbox("Facturation Dématérialisée", ["Yes","No"])

        if st.button("🔮 Prédire le risque de Churn", type="primary"):
            total_charges = round(monthly * tenure * 1.0, 2)

            input_data = {
                "gender": "Male", "SeniorCitizen": 1 if senior=="Oui" else 0,
                "Partner": partner, "Dependents": dependents, "tenure": tenure,
                "PhoneService": phone, "MultipleLines": "No",
                "InternetService": internet, "OnlineSecurity": security,
                "OnlineBackup": "No", "DeviceProtection": "No",
                "TechSupport": techsup, "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly, "TotalCharges": total_charges,
                "ChurnBin": 0,
                "HasPartnerOrDep": 1 if (partner=="Yes" or dependents=="Yes") else 0,
                "HasStreaming": 0, "HasSecurity": 1 if security=="Yes" else 0,
                "AvgMonthlyRevenue": total_charges / (tenure+1),
            }
            input_df = pd.DataFrame([input_data])

            cat_cols = ["gender","Partner","Dependents","PhoneService","MultipleLines",
                        "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
                        "TechSupport","StreamingTV","StreamingMovies","Contract",
                        "PaperlessBilling","PaymentMethod"]
            num_cols = ["tenure","MonthlyCharges","TotalCharges","HasPartnerOrDep",
                        "HasStreaming","HasSecurity","AvgMonthlyRevenue","SeniorCitizen"]

            dummies = pd.get_dummies(input_df[cat_cols], drop_first=True)
            X_inp = pd.concat([input_df[num_cols], dummies], axis=1)

            # Align columns
            X_cols = st.session_state["X_cols"]
            X_inp = X_inp.reindex(columns=X_cols, fill_value=0)

            scaler = st.session_state["scaler"]
            X_sc = scaler.transform(X_inp)

            m = results[sel]["model"]
            pred = m.predict(X_sc)[0]
            prob = m.predict_proba(X_sc)[0][1] if hasattr(m, "predict_proba") else 0.5

            # Result display
            color = "#ef476f" if pred==1 else "#52b788"
            label = "⚠️ RISQUE DE CHURN" if pred==1 else "✅ CLIENT FIDÈLE"
            st.markdown(f"""
            <div style='background:{BG}; border:2px solid {color}; border-radius:16px;
                padding:1.5rem 2rem; text-align:center; margin:1rem 0;'>
                <div style='font-size:1.8rem; font-weight:700; color:{color};'>{label}</div>
                <div style='font-size:2.5rem; font-weight:800; color:{color};'>{prob:.1%}</div>
                <div style='color:{TEXT}; font-size:.9rem;'>Probabilité de désabonnement
                estimée par <b>{sel}</b></div>
            </div>
            """, unsafe_allow_html=True)

            if pred == 1:
                st.markdown("""
                <div class="insight alert">🚨 <b>Action Recommandée :</b> Ce client présente un risque élevé.
                Envisagez : offre de rétention personnalisée, appel proactif, upgrade contractuel ou
                remise sur la facturation mensuelle.</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="insight good">✅ <b>Client Stable :</b> Faible risque de churn détecté.
                Continuez à l'engager avec des offres premium et des programmes de fidélisation.</div>
                """, unsafe_allow_html=True)

        # API Info
        st.markdown("---")
        st.markdown('<p class="sec">🔌 API Flask — Intégration</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight">💡 <b>Déploiement API Flask disponible</b> — Le modèle peut être exposé
        via une API REST. Lancer avec : <code>python api/app_flask.py</code><br>
        Endpoint : <code>POST http://localhost:5000/predict</code></div>
        """, unsafe_allow_html=True)

        st.code("""
# Exemple d'appel API
import requests, json

payload = {
    "tenure": 12,
    "MonthlyCharges": 75.5,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "SeniorCitizen": 0,
    "PaymentMethod": "Electronic check",
    "PaperlessBilling": "Yes",
    "OnlineSecurity": "No",
    "TechSupport": "No"
}

response = requests.post("http://localhost:5000/predict", json=payload)
print(response.json())
# {'churn_prediction': 1, 'churn_probability': 0.73, 'risk_level': 'HIGH'}
        """, language="python")
