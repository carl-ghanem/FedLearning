import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Directory containing the .dyna files
directory = '/home/noah/Downloads/PeMS_raw_data/PeMS04'

# List all .dyna files in the directory
dyna_files = [f for f in os.listdir(directory) if f.endswith('.dyna')]

if not dyna_files:
    print("No .dyna files found in the directory.")
else:
    for file in dyna_files:
        file_path = os.path.join(directory, file)
        print(f"\nProcessing {file} for PyTorch DataLoader:")

        # Read the CSV
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(['entity_id', 'time'])

        # Handle missing values (fill with 0 for simplicity)
        df['traffic_flow'] = df['traffic_flow'].fillna(0)
        df['traffic_occupancy'] = df['traffic_occupancy'].fillna(0)
        df['traffic_speed'] = df['traffic_speed'].fillna(0)

        unique_sensors = sorted(df['entity_id'].unique())
        N = len(unique_sensors)
        sensor_to_idx = {s: i for i, s in enumerate(unique_sensors)}

        times = sorted(df['time'].unique())
        num_times = len(times)
        time_to_idx = {t: i for i, t in enumerate(times)}

        D = 3  # traffic_flow, traffic_occupancy, traffic_speed
        data = torch.zeros(num_times, N, D)

        for _, row in df.iterrows():
            t_idx = time_to_idx[row['time']]
            s_idx = sensor_to_idx[row['entity_id']]
            data[t_idx, s_idx, 0] = row['traffic_flow']
            data[t_idx, s_idx, 1] = row['traffic_occupancy']
            data[t_idx, s_idx, 2] = row['traffic_speed']

        print(f"Data shape: {data.shape} (num_times, N={N}, D={D})")

        # Define the Dataset class
        class PeMSDataset(Dataset):
            def __init__(self, data, T):
                self.data = data  # (num_times, N, D)
                self.T = T
                self.num_samples = data.shape[0] - T + 1

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                # Return (N, T, D)
                window = self.data[idx:idx + self.T]  # (T, N, D)
                return window.permute(1, 0, 2)  # (N, T, D)

        # Example usage: T=12 historical time steps
        T = 12
        dataset = PeMSDataset(data, T)
        print(f"Dataset length: {len(dataset)} samples")

        # DataLoader with batch_size Bs
        Bs = 32  # Example batch size
        dataloader = DataLoader(dataset, batch_size=Bs, shuffle=True)

        # Example: Get one batch
        for batch in dataloader:
            print(f"Batch shape: {batch.shape} (Bs={Bs}, N={N}, T={T}, D={D})")
            break  # Just show one batch