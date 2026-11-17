import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

training_data = np.loadtxt("regression_train.txt")
X = training_data[:, :3] # Keeps x1,x2,x3 (n rows, 3 columns)
y = training_data[:, 3] # (n rows, 1 columns)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

X_train = torch.tensor(X_train_scaled, dtype = torch.float32)
X_val = torch.tensor(X_val_scaled, dtype = torch.float32)

y_train = torch.tensor(y_train, dtype = torch.float32).reshape(-1,1)
y_val = torch.tensor(y_val, dtype = torch.float32).reshape(-1,1)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
#TO MAKE BATCHES
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

batch_size = 32
num_epochs = 10
learning_rate = 0.01



class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        logits = self.nn(x)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)




def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        return running_loss / len(train_loader)




def validate(model, device, test_loader, criterion):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            running_loss += loss.item()
            return running_loss / len(val_loader)



for epoch in range(num_epochs):
    train(model, device, train_loader, criterion, optimizer, epoch)
    validate(model, device, val_loader, criterion)


model.eval()
with torch.no_grad():
    preds = model(X_val.to(device))
    preds = preds.cpu().numpy().flatten()

real_values = y_val.numpy().flatten()
mse = mean_squared_error(real_values, preds)
print("Nn mse: ", mse)