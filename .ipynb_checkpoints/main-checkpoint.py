from sklearn.model_selection import train_test_split
import fastf1
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

fastf1.Cache.enable_cache('D:\\Programming\\Python\\cache\\fastf1')

# session = fastf1.get_session(2026, "British", 'R')
# session.load()
# print(session.results)
# print(session.results.columns)

events = fastf1.get_events_remaining()
nextRound = events.iloc[0].RoundNumber
# session = fastf1.get_session(2026, nextRound-1, 'R')
# session.load()
# results = session.results.copy()
# results['Round'] = nextRound-1


# session2 = fastf1.get_session(2026, nextRound-2, 'R')
# session2.load()
# results2 = session2.results.copy()
# results2['Round'] = nextRound-2


# data = pd.concat([results, results2], ignore_index=True)
# print(data)

individualRaceData = []

for round in range(nextRound-5, nextRound):
    session = fastf1.get_session(2026, round, 'R')
    session.load(telemetry=False)

    results = session.results.copy()
    results['Round'] = round
    
    individualRaceData.append(results)

# race columns:
# ['DriverNumber', 'BroadcastName', 'Abbreviation', 'DriverId', 'TeamName',
#        'TeamColor', 'TeamId', 'FirstName', 'LastName', 'FullName',
#        'HeadshotUrl', 'CountryCode', 'Position', 'ClassifiedPosition',
#        'GridPosition', 'Q1', 'Q2', 'Q3', 'Time', 'Status', 'Points', 'Laps',
#        'Round']

raceData = pd.concat(individualRaceData, ignore_index=True)
raceData.drop(columns=[
        'BroadcastName', 'Abbreviation', 'DriverId', 'TeamName', 'TeamColor', 
        'FirstName', 'LastName', 'FullName', 'HeadshotUrl', 'CountryCode', 'ClassifiedPosition',
        'Q1', 'Q2', 'Q3', 'Status'
    ], inplace=True)


X = raceData.drop(columns=['Position', 'Time'])
y = raceData['Position']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = pd.get_dummies(X_train, columns=['DriverNumber', 'TeamId', 'Laps', 'Round'])
X_test = pd.get_dummies(X_test, columns=['DriverNumber', 'TeamId', 'Laps', 'Round'])
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(y_test)

mae = mean_absolute_error(y_test, preds)
print(f'Mean Absolute Error: {mae}')    