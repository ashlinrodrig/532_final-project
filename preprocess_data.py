import pandas as pd
from scipy import stats

# Load datasets
lap_times = pd.read_csv('formula-1-race-data-19502017/lapTimes.csv')
# Convert 'raceId' in lap_times to int64
lap_times['raceId'] = pd.to_numeric(lap_times['raceId'], errors='coerce').astype('Int64')

pit_stops = pd.read_csv('formula-1-race-data-19502017/pitStops.csv')
# Convert duration to numeric, forcing any errors to NaN
pit_stops['duration'] = pd.to_numeric(pit_stops['duration'], errors='coerce')

races = pd.read_csv('formula-1-race-data-19502017/races.csv')

safety_cars = pd.read_csv('formula-1-race-events/safety_cars.csv')
# Convert 'Race' column in safety_cars to int64
safety_cars['Race'] = pd.to_numeric(safety_cars['Race'], errors='coerce').astype('Int64')

results = pd.read_csv('formula-1-race-data-19502017/results.csv')

circuits = pd.read_csv('formula-1-race-data-19502017/circuits.csv', encoding='latin1')

qualifying = pd.read_csv('formula-1-race-data-19502017/qualifying.csv')

driver_standings = pd.read_csv('formula-1-race-data-19502017/driverStandings.csv')

red_flags = pd.read_csv('formula-1-race-events/red_flags.csv')


# Handling missing data and outliers
# Check for nulls in each dataset
datasets = [lap_times, pit_stops, races, safety_cars, results, circuits, qualifying, driver_standings, red_flags]
for i, dataset in enumerate(datasets):
    print(f"Null values in {dataset}:\n", dataset.isnull().sum())

# Fill missing values for lap times and pit stop durations (using median)
lap_times['milliseconds'] = lap_times['milliseconds'].interpolate(method='linear')

# Fill missing values in duration with the median of the column
pit_stops['duration'] = pit_stops['duration'].fillna(pit_stops['duration'].median())


# Remove outliers in pit stop durations ( that are 3 standard deviations away)
pit_stops['z_score_duration'] = stats.zscore(pit_stops['duration'])
pit_stops = pit_stops[abs(pit_stops['z_score_duration']) <= 3]

# Reset the index
pit_stops.reset_index(drop=True, inplace=True)


# Generating desired dataset from the CSVs
# Merge lap times and pit stops on raceId and driverId
lap_data = pd.merge(lap_times, pit_stops, on=['raceId', 'driverId', 'lap'], how='left')

# Add race context
lap_data = pd.merge(lap_data, races[['raceId', 'year', 'round', 'circuitId', 'name']], on='raceId', how='left')

# Add circuit information
lap_data = pd.merge(lap_data, circuits[['circuitId', 'location', 'country', 'lat', 'lng', 'alt']], on='circuitId', how='left')

# Add safety car information
lap_data = pd.merge(lap_data, safety_cars, left_on='raceId', right_on='Race', how='left')


# Capturing relevant information for tire change prediction
# Calculate cumulative lap time per race and driver
lap_data['cumulative_time'] = lap_data.groupby(['raceId', 'driverId'])['milliseconds_x'].cumsum()

# Calculate lap-to-lap degradation (time difference between consecutive laps)
lap_data['lap_degradation'] = lap_data.groupby(['raceId', 'driverId'])['milliseconds_x'].diff()

# Add a binary indicator for pit stops
lap_data['is_pit_stop'] = lap_data['stop'].notnull().astype(int)

# Flag laps with safety cars
lap_data['safety_car_flag'] = lap_data['Cause'].notnull().astype(int)

# Convert categorical data to numerical
lap_data['location'] = lap_data['location'].astype('category').cat.codes
lap_data['country'] = lap_data['country'].astype('category').cat.codes
lap_data['year'] = lap_data['year'].astype('category').cat.codes

# Load the drivers dataset
drivers = pd.read_csv('formula-1-race-data-19502017/drivers.csv', encoding='latin1')

# Check the columns of the drivers dataset
print(drivers.columns)

# Merge the 'results' dataset with the 'drivers' dataset on 'driverId'
driver_info = pd.merge(results, drivers[['driverId', 'forename', 'surname']], on='driverId', how='left')

# Create a full name column by combining 'forename' and 'surname'
driver_info['full_name'] = driver_info['forename'] + ' ' + driver_info['surname']

# Now you can use the 'full_name' column in your descriptions
description = [f"Driver {row['full_name']} completed lap {row['laps']} in {row['time']}" 
                    for index, row in driver_info.iterrows()]


# Output the first few descriptions to verify
print(description[:5])

# # Convert milliseconds to minutes:seconds format for lap times
# lap_data['lap_time'] = lap_data['milliseconds_x'] / 1000  # Convert to seconds
# lap_data['lap_time_formatted'] = lap_data['lap_time'].apply(lambda x: f"{int(x // 60)}:{int(x % 60):02d}.{int((x * 100) % 100):02d}")

# Convert the description list to a pandas DataFrame
description_df = pd.DataFrame(description, columns=['description'])

# Save the DataFrame to a CSV file
description_df.to_csv('lap_descriptions.csv', index=False)

# Removing duplicates
lap_data.drop_duplicates(inplace=True)

# Save preprocessed data to CSV
lap_data.to_csv('preprocessed_lap_data.csv', index=False)

# Removing duplicates
lap_data.drop_duplicates(inplace=True)

# Save preprocessed data to CSV
lap_data.to_csv('preprocessed_lap_data.csv', index=False)
