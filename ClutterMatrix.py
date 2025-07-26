from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns 
import pandas as pd 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np

# --- 1. Model class definition ---
class SimpleMLP(nn.Module):
    def __init__(self, input_features, hidden_size, output_features):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(input_features, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_features) # خروجی خام (logits)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        return x 

# --- 2. Data preparation ---
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # (N,) به (N, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

print(f"Shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Number of batches in train_loader: {len(train_loader)}")

# --- 3. Initializing the model, defining the loss function and optimizer ---
input_features = 2
hidden_size = 10
output_features = 1

model = SimpleMLP(input_features, hidden_size, output_features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Using BCEWithLogitsLoss for numerical stability
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Model is on: {device}")

# --- 4.  (Training Loop) ---
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)

        predicted_probs = torch.sigmoid(outputs)
        predicted_labels = (predicted_probs >= 0.5).float()
        correct_predictions += (predicted_labels == batch_y).sum().item()
        total_samples += batch_y.size(0)

    epoch_loss = running_loss / len(train_dataset)
    epoch_accuracy = correct_predictions / total_samples * 100

    if (epoch + 1) % 10 == 0: # هر 10 اپوک چاپ شود
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# --- 5. ا(Evaluation) ---
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    test_loss_sum = 0.0

    for batch_X_test, batch_y_test in test_loader:
        batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)

        outputs_test = model(batch_X_test)
        test_loss = criterion(outputs_test, batch_y_test)
        test_loss_sum += test_loss.item() * batch_X_test.size(0)

        predicted_probs_test = torch.sigmoid(outputs_test)
        predicted = (predicted_probs_test >= 0.5).float()

        total += batch_y_test.size(0)
        correct += (predicted == batch_y_test).sum().item()

    avg_test_loss = test_loss_sum / len(test_dataset)
    accuracy = 100 * correct / total
    print(f"\nTest Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

# Convert model outputs and real labels to NumPy arrays on the CPU
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for batch_X_test, batch_y_test in test_loader:
        batch_X_test = batch_X_test.to(device)
        outputs_test = model(batch_X_test)
        predicted_probs_test = torch.sigmoid(outputs_test)
        predicted = (predicted_probs_test >= 0.5).float() # برچسب‌های 0 یا 1

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y_test.cpu().numpy())

all_preds = np.array(all_preds).flatten()
all_labels = np.array(all_labels).flatten()

# --- (Classification Report) ---
print("\n--- Classification Report ---")
# target_names can be the names of your classes, for example ['Class 0', 'Class 1']
print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))

# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
print("\n--- Confusion Matrix ---")
print(cm)

# Visualization of the clutter matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()