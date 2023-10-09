import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('consumption_temp.csv')

# Filter for 'Oslo'
oslo_data = data[data['location'] == 'oslo']

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(oslo_data[['consumption', 'temperature']])

# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    target = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        target.append(data[i+seq_length])
    return torch.tensor(sequences).float(), torch.tensor(target).float()

seq_length = 24
X, y = create_sequences(scaled_data, seq_length)

# Split into training and test sets
train_size = int(0.8 * len(X))
X_train_seq = X[:train_size]
y_train_seq = y[:train_size]
X_test_seq = X[train_size:]
y_test_seq = y[train_size:]

# Adjusted LSTM model definition
class AdjustedLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(AdjustedLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm1(x, (h0.detach(), c0.detach()))
        out, (hn, cn) = self.lstm2(out)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Initialize the model, criterion, optimizer, and scheduler
input_dim = 2
hidden_dim = 100
num_layers = 2
output_dim = 1
model = AdjustedLSTMModel(input_dim, hidden_dim, num_layers, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

# Training loop
num_epochs = 50
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_seq)
    loss = criterion(outputs, y_train_seq)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    model.eval()
    val_outputs = model(X_test_seq)
    val_loss = criterion(val_outputs, y_test_seq)
    val_losses.append(val_loss.item())
    
    scheduler.step()

# You can print or plot train_losses and val_losses to see the training progression
