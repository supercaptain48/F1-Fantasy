import fastf1

fastf1.Cache.enable_cache('cache')

# for round in range(1, 25):
#     session = fastf1.get_session(2025, round, 'R')
#     session.load(telemetry=False)

#     results = session.results.copy()
#     results['Round'] = round

#     results.to_csv(f'data/race_results_2025_{round}.csv', index=False)

for round in range(1, 25):
    session = fastf1.get_session(2025, round, 'Q')
    session.load(telemetry=False)

    results = session.results.copy()
    results['Round'] = round

    results.to_csv(f'data/qualifying_results_2025_{round}.csv', index=False)