import pandas as pd
import json

def create_windowed_dataset(W, H, labels_path):

    # Download the dataset and convert the timestamp from a string
    url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv"
    df = pd.read_csv(url)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['is_incident'] = 0

    # Obtain all incident windows from the json file
    with open(labels_path, 'r') as f:
        all_labels = json.load(f)
        nyc_windows = all_labels.get("realKnownCause/nyc_taxi.csv")
    
    # Create a mask on the "is_incident" column to set the target to 1 when timestamp is within the window
    for start, end in nyc_windows:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        df.loc[mask, 'is_incident'] = 1

    # Add addional features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek 
    df['month'] = df['timestamp'].dt.month

    # Compute the target for each timestamp by checking if there is an incident in the next H steps
    df['target'] = df['is_incident'].shift(-H).rolling(window=H, min_periods=1).max().fillna(0).astype(int)

    features, targets, timestamps = [], [], []
    hours, days, months = [], [], []
    value_array = df['value'].values

    # Setup the processed dataset
    for i in range(W, len(df) - H):
        features.append(value_array[i-W:i])
        targets.append(df['target'].iloc[i])
        timestamps.append(df['timestamp'].iloc[i])
        hours.append(df['hour'].iloc[i])
        days.append(df['day_of_week'].iloc[i])
        months.append(df['month'].iloc[i])

    feature_col = [f't-{W-i}' for i in range(W)]
    df_processed = pd.DataFrame(features, columns=feature_col)
    
    df_processed.insert(0, 'timestamp', timestamps)
    df_processed['hour'] = hours
    df_processed['day_of_week'] = days
    df_processed['month'] = months
    df_processed['target'] = targets

    # Export processed dataset to a csv file
    df_processed.to_csv("data.csv", index=False)