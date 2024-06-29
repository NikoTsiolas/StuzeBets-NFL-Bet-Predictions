# StuzeBets
# Niko Tsiolas
# June 24th, 2024

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression

# Load datasets
stadiums = pd.read_csv('nfl_stadiums.csv', encoding='latin1')
teams = pd.read_csv('nfl_teams.csv', encoding='latin1')
scores = pd.read_csv('spreadspoke_scores.csv', encoding='latin1')

# Fill missing values using forward fill method
stadiums.ffill(inplace=True)
teams.ffill(inplace=True)
scores.ffill(inplace=True)

# Verify the data cleaning
print(stadiums.info())
print(scores.info())
print(teams.info())


# Define win columns
scores['home_win'] = (scores['score_home'] > scores['score_away']).astype(int)
scores['away_win'] = (scores['score_away'] > scores['score_home']).astype(int)

# Calculate average points scored by home and away teams
scores['team_points_avg_home'] = scores.groupby('team_home')['score_home'].transform('mean')
scores['team_points_avg_away'] = scores.groupby('team_away')['score_away'].transform('mean')

# Adding rolling average for the last 5 games for home and away teams
scores['home_team_avg_last_5'] = scores.groupby('team_home')['score_home'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
scores['away_team_avg_last_5'] = scores.groupby('team_away')['score_away'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

# Calculate performance metrics based on weather conditions using available columns
# Group by home team and weather conditions
home_performance = scores.groupby(['team_home', 'weather_temperature', 'weather_wind_mph', 'weather_humidity'])['home_win'].mean().reset_index()
home_performance.rename(columns={'team_home': 'team', 'home_win': 'win_rate'}, inplace=True)

# Group by away team and weather conditions
away_performance = scores.groupby(['team_away', 'weather_temperature', 'weather_wind_mph', 'weather_humidity'])['away_win'].mean().reset_index()
away_performance.rename(columns={'team_away': 'team', 'away_win': 'win_rate'}, inplace=True)

# Combine home and away performance data
performance = pd.concat([home_performance, away_performance])
performance = performance.groupby(['team', 'weather_temperature', 'weather_wind_mph', 'weather_humidity'])['win_rate'].mean().reset_index()


# Verify the new features
print(scores[['team_home', 'team_points_avg_home', 'team_away', 'team_points_avg_away']].head())

#func to get weather performance of the team and if value is nil change to 0.5
def get_weather_performance(team, temp, wind, humidity, performance):
    perf = performance[(performance['team'] == team) &
                       (performance['weather_temperature'] == temp) &
                       (performance['weather_wind_mph'] == wind) &
                       (performance['weather_humidity'] == humidity)]
    return perf['win_rate'].values[0] if not perf.empty else 0.5  


# Apply to home and away teams
scores['home_weather_performance'] = scores.apply(lambda row: get_weather_performance(row['team_home'], row['weather_temperature'], row['weather_wind_mph'], row['weather_humidity'], performance), axis=1)
scores['away_weather_performance'] = scores.apply(lambda row: get_weather_performance(row['team_away'], row['weather_temperature'], row['weather_wind_mph'], row['weather_humidity'], performance), axis=1)

# Define the target variable
scores['home_win'] = (scores['score_home'] > scores['score_away']).astype(int)

# Define the features and target variables
features = ['team_points_avg_home', 'team_points_avg_away', 'spread_favorite', 'over_under_line',
            'weather_temperature', 'weather_wind_mph', 'weather_humidity', 'home_weather_performance', 'away_weather_performance']

for feature in features:
    scores[feature] = pd.to_numeric(scores[feature], errors='coerce')

X = scores[features].copy()
y = scores['home_win']

# Handling missing values
X.fillna(0, inplace=True)



# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')


# Train and evaluate the model
model = RandomForestClassifier(n_estimators=100, random_state=45)
model.fit(X_train, y_train)

joblib.dump(model, 'model/stuzebets_model.pkl')

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{classification_rep}')
print(f'AUC-ROC: {roc_auc}')
