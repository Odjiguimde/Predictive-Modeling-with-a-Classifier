#     Modélisation Prédictive du Churn Client

Ce projet vise à développer un système intelligent capable d’évaluer la solvabilité des clients et de leur attribuer un score de crédit fiable à partir de données financières hétérogènes.
Dans un contexte d’inclusion financière, notamment en Afrique, de nombreux individus ne disposent pas d’historique bancaire classique. Ce système exploite des données alternatives (transactions, mobile money, revenus, dépenses) pour prédire le risque de défaut et faciliter l’accès au crédit.

## Objectif

1.Identifier les clients à risque de churn (désabonnement) avant qu'ils ne partent,
    pour permettre une rétention proactive et réduire les pertes de revenus.

2.Construire un pipeline complet de traitement des données financières.

3.Identifier les facteurs clés influençant la solvabilité.

4.Développer des modèles de machine learning pour la prédiction du risque de crédit
    Fournir un score de crédit interprétable et exploitable.

---

## Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://votre-url.streamlit.app/)

---

## Structure du Projet

```
projet5-churn/
│
├── app.py                          # Dashboard Streamlit interactif
├── telco_churn.csv                 # Dataset Telco Customer Churn
├── Projet5_Churn_Prediction.ipynb  # Notebook — démarche complète
├── requirements.txt                # Dépendances Python
├── Dockerfile                      # Image Docker
├── docker-compose.yml              # Orchestration Streamlit + Flask
├── README.md
│
├── api/
│   └── app_flask.py                # API REST Flask (prédiction)
│
├── plots/                          # Graphiques exportés
│   ├── 01_eda_churn.png
│   ├── 02_correlation.png
│   ├── 03_model_comparison.png
│   ├── 04_confusion_matrices.png
│   └── 05_feature_importance.png
│
└── .streamlit/
    └── config.toml                 # Thème sombre personnalisé
```

---

## Modèles de Classification Implémentés

| Modèle | Type | Points Forts |
|---|---|---|
| **Régression Logistique** | Linéaire | Interprétable, rapide |
| **K-Nearest Neighbors** | Instance-based | Simple, non-paramétrique |
| **Naive Bayes** | Probabiliste | Très rapide, efficace |
| **Random Forest** | Ensemble | Robuste, feature importance |
| **Gradient Boosting** | Ensemble | Précision élevée |
| **XGBoost** | Boosting | Production-ready, rapide |
| **LightGBM** | Boosting | Très rapide sur grands datasets |

---

## Dashboard Streamlit — 5 Onglets

| Onglet | Contenu |
|---|---|
| **📊 Aperçu & EDA** | Dataframe, distributions, variables catégorielles |
| **🔬 Analyse Churn** | Segmentation, heatmap, scatter, insights métier |
| **🤖 Modèles ML** | Entraînement, matrice de confusion, ROC, feature importance |
| **📈 Comparaison** | Tableau récapitulatif, courbes ROC, meilleur modèle |
| **🎯 Prédiction Live** | Formulaire client → prédiction en temps réel |

---

## API Flask — Endpoints

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Informations API |
| `GET` | `/health` | Statut serveur |
| `POST` | `/train` | Entraîner tous les modèles |
| `POST` | `/predict` | Prédire le churn d'un client |
| `GET` | `/models` | Liste des modèles entraînés |

### Exemple d'appel API

```bash
# 1. Entraîner les modèles
curl -X POST http://localhost:5000/train

# 2. Prédire
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 6,
    "MonthlyCharges": 85.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "SeniorCitizen": 0,
    "PaymentMethod": "Electronic check",
    "PaperlessBilling": "Yes",
    "OnlineSecurity": "No",
    "TechSupport": "No"
  }'
```

**Réponse :**
```json
{
  "model_used": "random_forest",
  "churn_prediction": 1,
  "churn_label": "Churn",
  "churn_probability": 0.73,
  "risk_level": "HIGH",
  "recommendation": "Action immédiate : offre de rétention personnalisée recommandée."
}
```

---

## Déploiement Docker

### Option 1 — Streamlit seul
```bash
docker build -t churn-predictor .
docker run -p 8501:8501 churn-predictor
```
→ Dashboard : http://localhost:8501

### Option 2 — Dashboard + API (docker-compose)
```bash
docker-compose up --build
```
→ Dashboard : http://localhost:8501
→ API Flask  : http://localhost:5000

---

## Installation Locale

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-username/projet5-churn.git
cd projet5-churn

# 2. Environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4a. Lancer le Dashboard Streamlit
streamlit run app.py

# 4b. Lancer l'API Flask (autre terminal)
python api/app_flask.py
```

---

## Déploiement Streamlit Cloud (Gratuit)

1. Push sur GitHub (inclure `telco_churn.csv` dans le repo)
2. Aller sur [share.streamlit.io](https://share.streamlit.io)
3. Connecter ton compte GitHub
4. Sélectionner le repo → branche `main` → fichier `app.py`
5. Cliquer **Deploy** → URL générée en ~2 min

---

## Feature Engineering

| Feature | Description |
|---|---|
| `ChurnBin` | Variable cible binaire (0/1) |
| `HasPartnerOrDep` | A un partenaire ou des dépendants |
| `HasStreaming` | Abonné à au moins un service streaming |
| `HasSecurity` | A la sécurité en ligne ou protection device |
| `AvgMonthlyRevenue` | TotalCharges / (tenure + 1) |
| `TenureGroup` | Tranche d'ancienneté (0-1 an, 1-2 ans, …) |

---

## Insights Clés

- **Contrat mensuel** → taux de churn ~3× supérieur aux contrats long terme
- **Fibre optique** → churners disproportionnés (pricing ou qualité)
- **Ancienneté < 12 mois** → période critique d'attrition
- **Absence de services additionnels** → corrélé avec le départ

---

## Technologies

| Outil | Usage |
|---|---|
| `Python 3.11` | Langage principal |
| `Scikit-learn` | Modèles ML, preprocessing, métriques |
| `XGBoost / LightGBM` | Gradient boosting avancé |
| `Streamlit` | Dashboard interactif |
| `Flask` | API REST prédictive |
| `Docker` | Containerisation & déploiement |
| `Pandas / NumPy` | Manipulation des données |
| `Matplotlib / Seaborn` | Visualisations |

---

## Licence

MIT — Libre d'utilisation pour tout projet éducatif ou professionnel.
