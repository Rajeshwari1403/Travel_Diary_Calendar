import torch
import torch.nn as nn
import numpy as np
import requests
import json
from collections import Counter
import sys
from dotenv import load_dotenv
from pymongo import MongoClient
import os
import torch.nn.functional as F

# ---------------- MONGODB CONNECTION CHECK ----------------
def test_mongo_connection():
    try:
        # Establish connection to MongoDB
        client = MongoClient(MONGO_URI)
        db = client["usertravel"]
        collection = db["places"]

        # Test by counting the number of documents in the collection
        count = collection.count_documents({})
        print(f"Connected to MongoDB. Found {count} documents in the 'places' collection.")
        return True
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return False

def check_data_format():
    try:
        # Fetch data from MongoDB
        client = MongoClient(MONGO_URI)
        db = client["usertravel"]
        collection = db["places"]

        # Fetch a sample of documents (limit to 5)
        sample_data = list(collection.find().limit(5))
        
        if not sample_data:
            print("No data found in the database.")
            return

        print("Sample Data from MongoDB:")
        for doc in sample_data:
            print(doc)
            print("-" * 40)
        
        # Check if the fields expected by the model exist
        required_fields = ["Month", "Season", "Budget", "Activity_Preference", "Group_Size", "Suggested_Place"]
        for field in required_fields:
            if field not in sample_data[0]:
                print(f"⚠️ Missing field: {field}")

        print("Data format looks correct!")
        return True
    except Exception as e:
        print(f"Error fetching data from MongoDB: {e}")
        return False
# ---------------- MAPPINGS ----------------
month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}
season_map = {"Winter": 1, "Spring": 2, "Summer": 3, "Monsoon": 4, "Autumn": 5}
activity_map = {"Adventure": 1, "Relaxation": 2, "Sightseeing": 3, "Eco Tourism": 4}
budget_map = {"Low": 1, "Medium": 2, "High": 3}
group_size_map = {
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6,
}
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["usertravel"]
collection = db["places"]

# ---------------- MODEL ----------------
class TravelRecommender(nn.Module):
    def __init__(self, num_classes):
        super(TravelRecommender, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 64)
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)


# ---------------- DATA HELPERS ----------------
def balance_dataset(data):
    place_counts = Counter(d.get("Suggested_Place") for d in data if "Suggested_Place" in d)
    min_occurrences = min(place_counts.values(), default=0)
    balanced_data = []
    place_counter = {}

    for d in data:
        place = d.get("Suggested_Place")
        if place and place_counter.get(place, 0) < min_occurrences:
            balanced_data.append(d)
            place_counter[place] = place_counter.get(place, 0) + 1

    return balanced_data

# Fetch user data from MongoDB
def get_real_user_data():
    try:
        response = requests.get("http://127.0.0.1:5000/getUserData", timeout=10)
        response.raise_for_status()
        return response.json().get("data", [])
    except requests.RequestException as e:
        print(f"❌ Error fetching user data: {e}")
        return []


def get_weather_from_api():
    try:
        city = "Chennai"
        api_key = "81ba64561f8aa96605e3eead04b0b1e6"
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

        response = requests.get(url, timeout=10)
        data = response.json()

        if response.status_code != 200 or "main" not in data:
            raise Exception(data.get("message", "Weather data not available."))

        temperature = data["main"]["temp"]
        weather = data["weather"][0]["main"]

        print(f"\n🌡️ Temperature: {temperature}°C")
        print(f"☁️ Weather: {weather}")

        return temperature, weather
    except Exception as e:
        print(f"⚠️ Error fetching live weather: {e}")
        return 25, "Unknown"

# ---------------- TRAINING ----------------
def train_model(user_data_list):
    if not user_data_list:
        print("⚠️ No user data found! Training aborted.")
        return

    unique_places = sorted(set(user.get("Suggested_Place") for user in user_data_list if user.get("Suggested_Place")))
    place_map = {place: idx for idx, place in enumerate(unique_places)}

    with open("place_map.json", "w") as f:
        json.dump(place_map, f)

    model = TravelRecommender(num_classes=len(place_map))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    user_data_list = balance_dataset(user_data_list)

    for user in user_data_list:
        group_size = str(user.get("Group_Size", "1"))
        if int(group_size) >= 5:
            group_size = "5+"

        features = torch.tensor([
            month_map.get(user.get("Month"), 0),
            budget_map.get(user.get("Budget"), 0),
            season_map.get(user.get("Season"), 0),
            activity_map.get(user.get("Activity_Preference"), 0),
            group_size_map.get(group_size, 2)
        ], dtype=torch.float32).unsqueeze(0)

        label = place_map.get(user.get("Suggested_Place"), 0)
        target = torch.tensor([label], dtype=torch.long)

        optimizer.zero_grad()
        loss = criterion(model(features), target)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "travel_model.pth")
    print("✅ Model training complete and saved as travel_model.pth")


# ---------------- PREDICTION ---------------
# Recommend travel places
def get_recommendations(user_data, model):
    try:
        print(f"🔍 Generating recommendations for: {user_data}")

        # Check if user_data is a dictionary (ensure it's not a list)
        if not isinstance(user_data, dict):
            raise ValueError("user_data should be a dictionary")

        # Hybrid logic using past data
        all_data = get_real_user_data()

        # Ensure all_data is a list of dictionaries
        if not isinstance(all_data, list):
            raise ValueError("get_real_user_data() should return a list of dictionaries")

        recent_matches = [entry for entry in all_data if (
            entry.get("Month") == user_data.get("Month") and
            entry.get("Season") == user_data.get("Season") and
            entry.get("Budget") == user_data.get("Budget") and
            entry.get("Activity_Preference") == user_data.get("Activity_Preference")
        )]

        if recent_matches:
            print("💡 Hybrid match found from past users.")
            recommended_places = list(dict.fromkeys(
                [entry["Suggested_Place"] for entry in recent_matches if "Suggested_Place" in entry]
            ))[:3]
            return {"suggested_places": recommended_places}

        print("🔁 No match found. Falling back to model.")

        # Load place map
        with open("place_map.json", "r") as f:
            place_map = json.load(f)

        group_size = str(user_data.get("Group_Size", "1"))
        if int(group_size) >= 5:
            group_size = "5+"

        # Input tensor of 5 features
        features = torch.tensor([ 
            month_map.get(user_data.get("Month"), 0),
            season_map.get(user_data.get("Season"), 0),
            budget_map.get(user_data.get("Budget"), 0),
            activity_map.get(user_data.get("Activity_Preference"), 0),
            group_size_map.get(group_size, 2)
        ], dtype=torch.float32).unsqueeze(0)

        print("📊 Input tensor:", features)

        # Predict
        with torch.no_grad():
            probs = model(features).numpy()[0]

        all_places = list(place_map.keys())
        top_indices = np.argsort(probs)[-3:][::-1]
        recommended_places = [all_places[i] for i in top_indices]

        return {"suggested_places": recommended_places}

    except Exception as e:
        print(f"❌ Error in recommendation: {str(e)}", file=sys.stderr)
        return {"suggested_places": ["Manali", "Rishikesh", "Goa"]}

# ---------------- INPUT ----------------
def get_user_input():
    def get_valid_input(prompt, valid_map):
        while True:
            val = input(prompt).strip().title()
            if val in valid_map:
                return val
            print("❌ Invalid input! Try again.")
    month = get_valid_input("Enter Month (e.g., March): ", month_map)
    season = get_valid_input("Enter Season (Winter, Spring, Summer, Monsoon, Autumn): ", season_map)
    budget = get_valid_input("Enter Budget (Low, Medium, High): ", budget_map)
    activity = get_valid_input("Enter Activity Preference (Adventure, Relaxation, Sightseeing, Eco Tourism): ", activity_map)
    try:
        group_size = int(input("Enter Group Size (e.g., 2, 4, 6, etc.): ").strip())
    except ValueError:
        print("⚠️ Invalid group size! Using default = 2")
        group_size = 2
    return {
        "Month": month,
        "Season": season,
        "Budget": budget,
        "Activity_Preference": activity,
        "Group_Size": group_size
    }

def add_data_to_db(user_data):
    try:
        response = requests.post("http://127.0.0.1:5000/addUserData", json=user_data, timeout=60)
        response.raise_for_status()
        print("✅ User data successfully added to database!")
    except requests.RequestException as e:
        print(f"❌ Error adding user data: {e}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    user_data = get_user_input()
    temp, weather = get_weather_from_api()
    user_data["Temperature"] = temp
    user_data["Weather"] = weather

    all_user_data = get_real_user_data()
    train_model(all_user_data)

    with open("place_map.json", "r") as f:
        place_map = json.load(f)
    model = TravelRecommender(num_classes=len(place_map))
    model.load_state_dict(torch.load("travel_model.pth"))
    model.eval()

    print("\n🔮 Predicting top travel destinations...")
    prediction = get_recommendations(user_data, model)

    print("\n🏖️ Suggested Travel Destinations:")
    for place in prediction["suggested_places"]:
        print(f" → {place}")

    user_data["Suggested_Place"] = prediction["suggested_places"][0]
    add_data_to_db(user_data)