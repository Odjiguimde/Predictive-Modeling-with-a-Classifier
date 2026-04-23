"""
API Flask — Churn Predictor
Projet 5 : Modélisation Prédictive

Endpoints :
  GET  /              → Info API
  GET  /health        → Statut
  POST /train         → Entraîner les modèles
  POST /predict       → Prédire le churn d'un client
  GET  /models        → Métriques de tous les modèles
  GET  /model/<name>  → Métriques d'un modèle spécifique
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)

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

app = Flask(__name__)

# ─── Global State ───────────────────────────────────────────
trained_models = {}
scaler         = None
feature_cols   = []
best_model_name = None
DATA_PATH      = os.path.join(os.path.dirname(__file__), "..", "telco_churn.csv")

CAT_COLS = ["gender","Partner","Dependents","PhoneService","MultipleLines",
            "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
            "TechSupport","StreamingTV","StreamingMovies","Contract",
            "PaperlessBilling","PaymentMethod"]
NUM_COLS = ["tenure","MonthlyCharges","TotalCharges","HasPartnerOrDep",
            "HasStreaming","HasSecurity","AvgMonthlyRevenue","SeniorCitizen"]

# ─── Helper Functions ────────────────────────────────────────
def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Feature Engineering
    df["HasPartnerOrDep"]  = ((df["Partner"]=="Yes") | (df["Dependents"]=="Yes")).astype(int)
    df["HasStreaming"]     = ((df["StreamingTV"]=="Yes") | (df["StreamingMovies"]=="Yes")).astype(int)
    df["HasSecurity"]      = ((df["OnlineSecurity"]=="Yes") | (df["DeviceProtection"]=="Yes")).astype(int)
    df["AvgMonthlyRevenue"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChurnBin"] = (df["Churn"] == "Yes").astype(int)

    dummies = pd.get_dummies(df[CAT_COLS], drop_first=True)
    X = pd.concat([df[NUM_COLS].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    y = df["ChurnBin"]
    return X, y

def preprocess_input(data: dict) -> np.ndarray:
    """Transformer un dictionnaire client en vecteur de features."""
    df = pd.DataFrame([data])

    # Defaults pour les colonnes manquantes
    defaults = {
        "gender": "Male", "SeniorCitizen": 0,
        "Partner": "No", "Dependents": "No",
        "tenure": 12, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "No",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 60.0, "TotalCharges": 720.0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    df["TotalCharges"]     = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(df["MonthlyCharges"] * df["tenure"])
    df["HasPartnerOrDep"]  = ((df["Partner"]=="Yes") | (df["Dependents"]=="Yes")).astype(int)
    df["HasStreaming"]     = ((df.get("StreamingTV","No")=="Yes") | (df.get("StreamingMovies","No")=="Yes")).astype(int)
    df["HasSecurity"]      = ((df.get("OnlineSecurity","No")=="Yes") | (df.get("DeviceProtection","No")=="Yes")).astype(int)
    df["AvgMonthlyRevenue"] = df["TotalCharges"] / (df["tenure"] + 1)

    dummies = pd.get_dummies(df[CAT_COLS], drop_first=True)
    X = pd.concat([df[NUM_COLS].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    X = X.reindex(columns=feature_cols, fill_value=0)
    return scaler.transform(X)

# ─── Routes ─────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name": "Churn Predictor API",
        "version": "1.0.0",
        "description": "API Flask pour la prédiction du churn client Télécoms",
        "endpoints": {
            "GET  /":              "Informations API",
            "GET  /health":        "Statut du serveur et des modèles",
            "POST /train":         "Entraîner tous les modèles",
            "POST /predict":       "Prédire le churn d'un client",
            "GET  /models":        "Métriques de tous les modèles entraînés",
            "GET  /model/<name>":  "Métriques d'un modèle spécifique",
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "models_trained": len(trained_models),
        "best_model":    best_model_name,
        "data_available": os.path.exists(DATA_PATH),
    })


@app.route("/train", methods=["POST"])
def train():
    global trained_models, scaler, feature_cols, best_model_name
    try:
        X, y = load_and_preprocess()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)
        feature_cols = X.columns.tolist()

        models_def = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "knn":                 KNeighborsClassifier(n_neighbors=7),
            "naive_bayes":         GaussianNB(),
            "random_forest":       RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
            "gradient_boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        if HAS_XGB:
            models_def["xgboost"] = XGBClassifier(n_estimators=100, random_state=42,
                                                   use_label_encoder=False, eval_metric="logloss")
        if HAS_LGB:
            models_def["lightgbm"] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)

        metrics_summary = {}
        trained_models = {}

        for name, m in models_def.items():
            m.fit(X_tr_sc, y_tr)
            y_pred = m.predict(X_te_sc)
            y_prob = m.predict_proba(X_te_sc)[:,1] if hasattr(m, "predict_proba") else np.zeros(len(y_te))
            cv = cross_val_score(m, X_tr_sc, y_tr, cv=StratifiedKFold(5), scoring="f1")

            trained_models[name] = m
            metrics_summary[name] = {
                "accuracy":   round(accuracy_score(y_te, y_pred), 4),
                "precision":  round(precision_score(y_te, y_pred, zero_division=0), 4),
                "recall":     round(recall_score(y_te, y_pred, zero_division=0), 4),
                "f1":         round(f1_score(y_te, y_pred, zero_division=0), 4),
                "roc_auc":    round(roc_auc_score(y_te, y_prob), 4),
                "cv_f1_mean": round(cv.mean(), 4),
                "cv_f1_std":  round(cv.std(), 4),
            }

        best_model_name = max(metrics_summary, key=lambda k: metrics_summary[k]["f1"])

        return jsonify({
            "status":      "success",
            "models_trained": list(trained_models.keys()),
            "best_model":  best_model_name,
            "metrics":     metrics_summary,
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    if not trained_models:
        return jsonify({"error": "Modèles non entraînés. Appelle POST /train d'abord."}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON invalide ou vide."}), 400

    model_name = data.pop("model", best_model_name)
    if model_name not in trained_models:
        return jsonify({"error": f"Modèle '{model_name}' non trouvé. Disponibles: {list(trained_models.keys())}"}), 400

    try:
        X_inp = preprocess_input(data)
        m = trained_models[model_name]
        pred = int(m.predict(X_inp)[0])
        prob = float(m.predict_proba(X_inp)[0][1]) if hasattr(m, "predict_proba") else 0.5

        if prob >= 0.7:   risk = "HIGH"
        elif prob >= 0.4: risk = "MEDIUM"
        else:             risk = "LOW"

        return jsonify({
            "model_used":        model_name,
            "churn_prediction":  pred,
            "churn_label":       "Churn" if pred == 1 else "No Churn",
            "churn_probability": round(prob, 4),
            "risk_level":        risk,
            "recommendation": {
                "HIGH":   "Action immédiate : offre de rétention personnalisée recommandée.",
                "MEDIUM": "Surveillance active : planifier un contact proactif.",
                "LOW":    "Client stable : maintenir l'engagement habituel.",
            }[risk],
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/models", methods=["GET"])
def models_list():
    if not trained_models:
        return jsonify({"error": "Aucun modèle entraîné."}), 400
    return jsonify({
        "models": list(trained_models.keys()),
        "best_model": best_model_name,
        "total": len(trained_models),
    })


@app.route("/model/<name>", methods=["GET"])
def model_info(n):
    name = n
    if name not in trained_models:
        return jsonify({"error": f"Modèle '{name}' non trouvé."}), 404
    return jsonify({
        "name":      name,
        "type":      type(trained_models[name]).__name__,
        "is_best":   name == best_model_name,
    })


# ─── Run ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Churn Predictor API démarrée sur http://localhost:5000")
    print("📖 Endpoints : GET / | POST /train | POST /predict | GET /models")
    app.run(debug=True, host="0.0.0.0", port=5000)
