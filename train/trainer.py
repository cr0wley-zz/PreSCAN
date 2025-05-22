
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from models.predictor import LearnableSimilarityModel

class SceneSimilarityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def compute_mae(pred, target):
    return torch.mean(torch.abs(pred - target))

def train_model(df, feature_cols, target_col, num_epochs=2000, batch_size=128, lr=1e-3):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=74)

    train_dataset = SceneSimilarityDataset(X_train, y_train)
    test_dataset = SceneSimilarityDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LearnableSimilarityModel(input_dim=len(feature_cols)).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, epoch_mae = 0.0, 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            mae = compute_mae(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_mae += mae.item()

        print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.4f}, MAE = {epoch_mae/len(train_loader):.4f}")

        model.eval()
        test_loss, test_mae = 0.0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                mae = compute_mae(pred, y_batch)
                test_loss += loss.item()
                test_mae += mae.item()

        print(f"Test MSE: {test_loss / len(test_loader):.4f}, Test MAE: {test_mae / len(test_loader):.4f}")

    return model, scaler
