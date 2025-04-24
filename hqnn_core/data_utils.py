import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import glob

def load_rssi_dataset(
    scenario: str,
    signal: str,
    sc_num: str,
    batch_size: int = 8,
    base_path: str = "data/"
):
    """
    Loads training and test CSVs for a given scenario/signal and returns DataLoaders.

    Returns:
        train_loader, test_loader
    """
    train_path = f"{base_path}/{scenario}/db_sc{sc_num}_{signal}.csv"
    test_path = f"{base_path}/{scenario}/Tests_Scenario{sc_num}_{signal}.csv"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train[["RSSI A", "RSSI B", "RSSI C"]].values
    Y_train = df_train[["x", "y"]].values

    X_test = df_test[["RSSI A", "RSSI B", "RSSI C"]].values
    Y_test = df_test[["x", "y"]].values

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor), batch_size=batch_size)

    return train_loader, test_loader
