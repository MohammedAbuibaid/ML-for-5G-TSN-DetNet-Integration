import pandas as pd
import random
import time

# Path to your CSV file
csv_path = './CSV/ml-FRER-vis.csv'

# Load existing CSV
df = pd.read_csv(csv_path)

# Infinite loop to simulate live data
while True:
    # Randomly choose one row
    random_row = df.sample(n=1)

    # Append it to the CSV (without writing the header)
    random_row.to_csv(csv_path, mode='a', index=False, header=False)

    print("Appended one row to the CSV.")
    time.sleep(1)
