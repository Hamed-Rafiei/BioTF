import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
num_series = 100  # Number of time series per class
series_length = 150  # Length of each time series

# Generate training data
def generate_data(num_series, series_length):
    X_sin = np.zeros((num_series, series_length))
    X_cos = np.zeros((num_series, series_length))

    for i in range(num_series):
        t = np.linspace(0, 4*np.pi, series_length) + np.random.randn() * np.pi
        X_sin[i, :] = np.sin(t) + np.random.randn(series_length) * 0.1
        X_cos[i, :] = np.cos(t) + np.random.randn(series_length) * 0.1

    X = np.vstack((X_sin, X_cos))
    Y = np.concatenate([np.zeros(num_series), np.ones(num_series)])
    return X, Y

# BioTF function
def BioTF(data, frame_length, frame_slide):
    Nsample, Nchannel = data.shape

    if frame_length == 1:
        return data

    if frame_length > Nsample:
        raise ValueError('frame_length cannot be larger than the number of samples within a window')

    if frame_slide <= 0 or frame_slide > frame_length:
        raise ValueError('frame_slide must be positive and not larger than frame_length')

    if (Nsample - frame_length) % frame_slide != 0:
        print('Warning: frame_slide does not evenly divide the series length minus frame_length. This may lead to unexpected behavior.')

    num_frames = (Nsample - frame_length) // frame_slide + 1
    out = np.zeros((Nchannel * 4, num_frames))

    for k in range(Nchannel):
        frame_start = 0
        for i in range(num_frames):
            frame_end = frame_start + frame_length
            current_frame = data[frame_start:frame_end, k]

            start_value = current_frame[0]
            end_value = current_frame[-1]
            max_value = np.max(current_frame)
            min_value = np.min(current_frame)

            out_idx = slice(k*4, (k+1)*4)
            out[out_idx, i] = [start_value, max_value, min_value, end_value]

            frame_start += frame_slide

    return out

# LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

# Main script
X, Y = generate_data(num_series, series_length)

# Apply BioTF
frame_length = 10
frame_slide = 3
X_transformed = np.array([BioTF(x.reshape(-1, 1), frame_length, frame_slide).T for x in X])

# Get the correct input size for the LSTM
num_features = X_transformed.shape[2]  # This should be 4 (start, max, min, end values)
sequence_length = X_transformed.shape[1]  # This is the number of frames

print(f"Data shape after BioTF: {X_transformed.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_transformed, Y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model with the correct input size
model = LSTMClassifier(input_size=num_features, hidden_size=100, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).float().mean()
    print(f'Test Accuracy: {accuracy.item()*100:.2f}%')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test.numpy(), predicted.numpy())
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()