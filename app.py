import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import numpy as np

# Configuration de Streamlit
st.set_page_config(page_title="BasketStatIQ", layout="wide")

# Chemin de base
BASE_DIR = r"C:\Users\ibohn\basketstat-iq\basketstat-iq"


# Charger les données
@st.cache_data
def load_data():
    data_path = os.path.join(BASE_DIR, 'data', 'cleaned', 'combined_player_stats.csv')
    return pd.read_csv(data_path)


# Charger le modèle et l'encodeur
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, 'models', 'xgboost_points_model.joblib')
    encoder_path = os.path.join(BASE_DIR, 'models', 'opponent_encoder.joblib')
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder


# Sidebar pour la navigation
st.sidebar.title("BasketStatIQ")
page = st.sidebar.selectbox("Choisir une page", ["Accueil", "Exploration des données", "Prédictions", "À propos"])

# Charger les données
df = load_data()

if page == "Accueil":
    st.title("Bienvenue sur BasketStatIQ")
    st.markdown("""
        BasketStatIQ est une application d'analyse des performances des joueurs NBA (2014-2024).
        Explorez les statistiques, visualisez les tendances, et prédisez les points des joueurs avec notre modèle XGBoost.
    """)
    st.subheader("Résumé des données")
    st.write(f"- Nombre de matchs : {len(df)}")
    st.write(f"- Nombre de joueurs : {df['PLAYER_NAME'].nunique()}")
    st.write(f"- Nombre d'équipes : {df['TEAM'].nunique()}")
    st.write(f"- Saisons couvertes : {df['SEASON'].unique()}")

elif page == "Exploration des données":
    st.title("Exploration des données")

    # Filtres
    st.subheader("Filtres")
    teams = sorted(df['TEAM'].unique())
    selected_team = st.selectbox("Sélectionner une équipe", ["Toutes"] + teams)
    players = sorted(df['PLAYER_NAME'].unique())
    selected_player = st.selectbox("Sélectionner un joueur", ["Tous"] + players)
    seasons = sorted(df['SEASON'].unique())
    selected_season = st.selectbox("Sélectionner une saison", ["Toutes"] + seasons)

    # Filtrer les données
    filtered_df = df.copy()
    if selected_team != "Toutes":
        filtered_df = filtered_df[filtered_df['TEAM'] == selected_team]
    if selected_player != "Tous":
        filtered_df = filtered_df[filtered_df['PLAYER_NAME'] == selected_player]
    if selected_season != "Toutes":
        filtered_df = filtered_df[filtered_df['SEASON'] == selected_season]

    # Afficher le tableau
    st.subheader("Tableau des statistiques")
    st.dataframe(
        filtered_df[['PLAYER_NAME', 'TEAM', 'SEASON', 'GAME_DATE', 'PTS', 'REB', 'AST', 'OFF_EFF', 'SIMPLIFIED_PER']])

    # Visualisations
    st.subheader("Visualisations")

    # Top 10 joueurs par points
    if filtered_df.empty:
        st.warning("Aucune donnée disponible pour les filtres sélectionnés.")
    else:
        top_players = filtered_df.groupby('PLAYER_NAME')['PTS'].mean().reset_index().sort_values('PTS',
                                                                                                 ascending=False).head(
            10)
        st.write("Top 10 joueurs par points (moyenne)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='PTS', y='PLAYER_NAME', data=top_players, ax=ax)
        plt.title('Top 10 joueurs par points marqués (moyenne)')
        plt.xlabel('Points par match')
        plt.ylabel('Joueur')
        st.pyplot(fig)

    # Impact des matchs back-to-back
    b2b_stats = filtered_df.groupby('BACK_TO_BACK')['PTS'].mean().reset_index()
    st.write("Points moyens selon les matchs back-to-back")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='BACK_TO_BACK', y='PTS', data=b2b_stats, ax=ax)
    plt.title('Points moyens selon les matchs back-to-back')
    plt.xlabel('Match Back-to-Back')
    plt.ylabel('Points par match')
    plt.xticks([0, 1], ['Non', 'Oui'])
    st.pyplot(fig)

elif page == "Prédictions":
    st.title("Prédictions de points")
    st.markdown("Entrez les caractéristiques pour prédire les points d'un joueur avec le modèle XGBoost.")

    # Charger le modèle
    try:
        model, le_opponent = load_model()
    except FileNotFoundError:
        st.error("Modèle ou encodeur non trouvé. Veuillez exécuter modeling.ipynb d'abord.")
        st.stop()

    # Formulaire de saisie
    st.subheader("Caractéristiques")
    col1, col2 = st.columns(2)

    with col1:
        back_to_back = st.checkbox("Match back-to-back")
        is_home = st.checkbox("Match à domicile")
        opponent = st.selectbox("Adversaire", sorted(df['OPPONENT'].unique()))
        min_moving_avg = st.slider("Moyenne mobile des minutes jouées", 0.0, 40.0, 30.0)
        points_moving_avg = st.slider("Moyenne mobile des points", 0.0, 40.0, 20.0)
        rebounds_moving_avg = st.slider("Moyenne mobile des rebonds", 0.0, 15.0, 5.0)

    with col2:
        assists_moving_avg = st.slider("Moyenne mobile des passes", 0.0, 15.0, 5.0)
        fg_pct_moving_avg = st.slider("Moyenne mobile du % de tirs", 0.0, 1.0, 0.45)
        plus_minus_moving_avg = st.slider("Moyenne mobile du plus/minus", -20.0, 20.0, 0.0)
        off_eff = st.slider("Efficacité offensive", 0.0, 2.0, 1.0)
        def_reb_pct = st.slider("Pourcentage de rebonds défensifs", 0.0, 1.0, 0.5)

    # Préparer les données pour la prédiction
    opponent_encoded = le_opponent.transform([opponent])[0]
    input_data = pd.DataFrame({
        'BACK_TO_BACK': [int(back_to_back)],
        'IS_HOME': [int(is_home)],
        'OPPONENT_ENCODED': [opponent_encoded],
        'MIN_MOVING_AVG': [min_moving_avg],
        'POINTS_MOVING_AVG': [points_moving_avg],
        'REBOUNDS_MOVING_AVG': [rebounds_moving_avg],
        'ASSISTS_MOVING_AVG': [assists_moving_avg],
        'FG_PCT_MOVING_AVG': [fg_pct_moving_avg],
        'PLUS_MINUS_MOVING_AVG': [plus_minus_moving_avg],
        'OFF_EFF': [off_eff],
        'DEF_REB_PCT': [def_reb_pct]
    })

    # Faire la prédiction
    if st.button("Prédire"):
        prediction = model.predict(input_data)[0]
        st.success(f"Points prédits : **{prediction:.2f}**")

elif page == "À propos":
    st.title("À propos de BasketStatIQ")
    st.markdown("""
        BasketStatIQ est un projet d'analyse des performances NBA (2014-2024) utilisant les données de l'API `nba_api`.
        - **Données** : Statistiques de 90 joueurs majeurs (3 par équipe) sur 10 saisons.
        - **EDA** : Analyse des tendances (back-to-back, domicile/extérieur, corrélations).
        - **Modélisation** : Prédiction des points avec XGBoost.
        - **Web App** : Interface interactive avec Streamlit.

        Développé par [votre nom] pour démontrer des compétences en analyse de données et machine learning.
    """)