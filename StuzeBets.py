#StuzeBets
#Niko Tsiolas
#June 24th, 2024

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split

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

# Calculate average points scored by home and away teams
scores['team_points_avg_home'] = scores.groupby('team_home')['score_home'].transform('mean')
scores['team_points_avg_away'] = scores.groupby('team_away')['score_away'].transform('mean')

# Verify the new features
print(scores[['team_home', 'team_points_avg_home', 'team_away', 'team_points_avg_away']].head())

scores['home_win'] = (scores['score_home'] > scores ['score_away']).astype(int)
#defining the features and target V's
features = ['team_points_avg_home', 'team_points_avg_away', 'spread_favorite', 'over_under_line', 'weather_temperature', 'weather_wind_mph', 'weather_humidity']
X = scores[features]
y = scores['home_win']

#just handling missing variables
X = X.copy()

X.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state=42)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

print ('First few entries of y_train:') 
print(y_train.head())