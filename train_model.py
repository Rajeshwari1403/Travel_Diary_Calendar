import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["usertravel"]
collection = db["places"]

# Fetch data
data = list(collection.find({}, {"_id": 0}))

if not data:
    raise ValueError("❌ No data found in MongoDB to train the model.")

# Mappings
month_map = {"January": 0, "February": 1, "March": 2, "April": 3, "May": 4, "June": 5,
             "July": 6, "August": 7, "September": 8, "October": 9, "November": 10, "December": 11}
season_map = {"Winter": 0, "Spring": 1, "Summer": 2, "Monsoon": 3, "Autumn": 4}
budget_map = {"Low": 0, "Medium": 1, "High": 2}
activity_map = {"Adventure": 0, "Relaxation": 1, "Culture": 2, "Wildlife": 3, "Pilgrimage": 4}

# Process data
X = []
y = []

for item in data:
    try:
        X.append([
            month_map.get(item["Month"], 0),
            budget_map.get(item["Budget"], 0),
            item.get("Temperature", 25),  # default temp
            season_map.get(item["Season"], 0),
            activity_map.get(item["Activity_Preference"], 0)
        ])
        y.append(item["Suggested_Place"])
    except KeyError as e:
        print(f"Skipping record due to missing field: {e}")

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save place_map
place_map = {label: int(idx) for idx, label in enumerate(le.classes_)}
with open("place_map.json", "w") as f:
    json.dump(place_map, f)
print("✅ Saved place_map.json")

# Convert to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y_encoded, dtype=torch.long)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
class TravelRecommender(nn.Module):
    def __init__(self, num_classes):
        super(TravelRecommender, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

model = TravelRecommender(num_classes=len(place_map))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 100
for epoch in range(epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        acc = (outputs.argmax(1) == y_train).float().mean()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")
    # After training the model
    with torch.no_grad():
      model.eval()
      val_outputs = model(X_test)
      val_loss = criterion(val_outputs, y_test)
      acc = (val_outputs.argmax(1) == y_test).float().mean()
      print(f"Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {acc:.4f}")
      model.train()  # Back to training mode


# Save model
torch.save(model.state_dict(), "travel_model.pth")
print("✅ Model trained and saved as travel_model.pth")
