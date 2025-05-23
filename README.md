# BasketStatIQ
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5+-green.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-purple.svg)](https://seaborn.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-yellow.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.6+-black.svg)](https://xgboost.readthedocs.io/)
[![nba_api](https://img.shields.io/badge/nba_api-1.1+-blueviolet.svg)](https://github.com/swar/nba_api)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

Unlocking NBA insights through data science

BasketStatIQ is a comprehensive data science project analyzing NBA player performance from the 2014-15 to 2023-24 seasons. It leverages the nba_api to collect data, performs exploratory data analysis (EDA), builds predictive models, and presents insights through an interactive Streamlit web app. The project aims to uncover trends in player performance, such as the impact of back-to-back games or home/away status, and predict points scored using machine learning.

## Features
- Data Collection: Scrapes game logs for 90 major players (3 per NBA team) across 10 seasons using nba_api.
- Exploratory Data Analysis (EDA): Visualizes trends, including top performers, back-to-back game effects, and home/away performance differences.
- Predictive Modeling: Uses an XGBoost model to predict points scored (PTS) based on features like minutes played, offensive efficiency, and opponent.
- Web App: A Streamlit-based interface for exploring data, visualizing insights, and making real-time predictions.
- Key Metrics: Includes points, rebounds, assists, offensive efficiency (OFF_EFF), defensive rebound percentage (DEF_REB_PCT), and more.

## Project Structure
``` plain
BasketStatIQ/
├── data/
│   ├── cleaned/
│   │   └── combined_player_stats.csv
│   └── temp/
├── models/
│   ├── xgboost_points_model.joblib
│   └── opponent_encoder.joblib
├── notebooks/
│   ├── eda.ipynb
│   └── modeling.ipynb
├── scripts/
│   └── data_collection.py
├── visuals/
│   ├── top_players_points.png
│   └── ...
├── app.py
├── requirements.txt
└── README.md
```
- data/: Stores raw and cleaned datasets.
- models/: Contains the trained XGBoost model and label encoder.
- notebooks/: Jupyter Notebooks for EDA and modeling.
- scripts/: Python script for data collection.
- visuals/: Generated plots from EDA and modeling.
- app.py: Streamlit web app.
- requirements.txt: Project dependencies.

## Installation

Prerequisites
- Python 3.8+
- Git
- Virtual environment (recommended)
  
Steps
- Clone the repository:
``` bash
git clone https://github.com/your-username/BasketStatIQ.git
cd BasketStatIQ
```
- Set up a virtual environment:
```bash
python -m venv basketstat_env
source basketstat_env/bin/activate  # On Windows: basketstat_env\Scripts\activate
```
- Install dependencies:
```bash
pip install -r requirements.txt
```
The requirements.txt includes:
```plain
streamlit
pandas
matplotlib
seaborn
joblib
xgboost
scikit-learn
nba_api
jupyter
```


- Create directories (if not already present):
``` plain
mkdir data\cleaned data\temp models visuals notebooks scripts
```

## Usage

### 1. Data Collection

Run the data collection script to fetch NBA player stats:
``` bash
python scripts/data_collection.py
```
- Output: data/cleaned/combined_player_stats.csv and temporary files in data/temp/.
- Note: This may take 20-30 minutes for all 90 players across 10 seasons. For testing, modify seasons or major_players in data_collection.py.

### 2. Exploratory Data Analysis (EDA)

Open the EDA notebook to analyze trends:
``` bash
jupyter notebook notebooks/eda.ipynb
```
- Generates visualizations in visuals/ (e.g., top_players_points.png).
- Explores metrics like back-to-back game effects and team performance.

### 3. Predictive Modeling

Run the modeling notebook to train an XGBoost model:
``` bash
jupyter notebook notebooks/modeling.ipynb
```
- Output: Trained model (models/xgboost_points_model.joblib) and encoder (models/opponent_encoder.joblib).
- Evaluates model performance and plots feature importance.

### 4. Web App
Launch the Streamlit web app:
```bash
streamlit run app.py
```
- Opens at http://localhost:8501.
- Features:
    - Home: Data summary.
    - Data Exploration: Filterable stats table and visualizations.
    - Predictions: Input features to predict points.
    - About: Project overview.

## Example

To predict points for a player in a back-to-back game:
- Run the web app.
- Navigate to the "Predictions" page.
- Select opponent, check "Back-to-Back," and adjust sliders for metrics like MIN_MOVING_AVG.
- Click "Predict" to see the estimated points.

## Deployment

- To share the web app online:
- Push the repository to GitHub.
- Sign up for Streamlit Cloud.
- Create a new app, link your GitHub repository, and specify app.py as the main file.
- Ensure requirements.txt and necessary files (data/, models/) are included.
- Deploy to get a public URL (e.g., https://basketstatiq.streamlit.app).

Note: For large datasets, host combined_player_stats.csv on a cloud service (e.g., Google Drive) and update app.py to fetch it.

## Screenshots

Home page: Data summary.
![image](https://github.com/user-attachments/assets/6ebe03c5-d2f6-4072-8027-f575883700bd)

Data Exploration: Top players bar plot.
![image](https://github.com/user-attachments/assets/ef9a99c1-afb6-48f5-b9d7-27998594ff5b)
![image](https://github.com/user-attachments/assets/84ef8b4e-c42b-497f-ab39-aea5ff0a4bf1)

Predictions: Prediction input form.
![image](https://github.com/user-attachments/assets/11ffb3b9-d346-46cd-8ba8-dfaf8ca32141)
![image](https://github.com/user-attachments/assets/c4a6df8e-4b4e-4e07-8f09-3b695008b809)

## Dependencies
- nba_api: Data collection.
- pandas, numpy: Data manipulation.
- matplotlib, seaborn: Visualizations.
- scikit-learn, xgboost: Modeling.
- streamlit: Web app.
- joblib: Model serialization.
- jupyter: Notebooks.

