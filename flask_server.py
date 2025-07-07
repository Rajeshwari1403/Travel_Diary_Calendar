from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient
import traceback
import logging
import os
import torch
import json
import time
from threading import Lock  # Thread safety for model retraining

from travel import get_recommendations, TravelRecommender, get_real_user_data, train_model

# Initialize Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s'
)

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("❌ MongoDB URI not found in .env file")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["usertravel"]
user_data_collection = db["users"]
places_collection = db["places"]

# Thread lock for safe retraining
model_lock = Lock()

# Load model and place_map
def load_model_and_map():
    if not os.path.exists("place_map.json") or not os.path.exists("travel_model.pth"):
        raise RuntimeError("❌ Required model files not found. Train the model first.")

    with open("place_map.json", "r") as f:
        place_map = json.load(f)

    model = TravelRecommender(num_classes=len(place_map))
    model.load_state_dict(torch.load("travel_model.pth"), strict=False)
    model.eval()
    logging.info("✅ Model and place_map loaded.")
    return model, place_map

model, place_map = load_model_and_map()

@app.route("/")
def home():
    return jsonify({"message": "Flask Travel Recommendation API is running."})

@app.route("/health", methods=["GET"])
def health_check():
    try:
        db.list_collection_names()
        return jsonify({"status": "ok", "model_loaded": True}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def fetch_user_data_from_db():
    try:
        user_data = list(user_data_collection.find({}))

        if not user_data:
            print("No user data found in the database.")
            return {}

        for user in user_data:
            user["_id"] = str(user["_id"])
        return user_data[0]

    except Exception as e:
        logging.error(f"Error fetching user data from DB: {str(e)}")
        return {"error": "Failed to fetch user data"}

@app.route('/getUserData', methods=['GET'])
def get_user_data():
    start_time = time.time()
    try:
        print("Fetching user data...")
        user_data = fetch_user_data_from_db()
        print(f"User data fetched in {time.time() - start_time:.2f} seconds")
        return jsonify(user_data)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": "Failed to fetch user data"}), 500

@app.route("/addUserData", methods=["POST"])
def add_user_data():
    global model, place_map
    try:
        user_data = request.json
        logging.info(f"📥 Incoming user data: {user_data}")

        # Check for required fields
        required_fields = ["Month", "Season", "Budget", "Activity_Preference", "Group_Size"]
        for field in required_fields:
            if field not in user_data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Normalize string inputs
        user_data = {
            k: v.strip().capitalize() if isinstance(v, str) else v
            for k, v in user_data.items()
        }

        predictions = get_recommendations(user_data, model)

        if not predictions["suggested_places"]:
            return jsonify({"error": "No predictions found."}), 400

        suggested_place = predictions["suggested_places"][0]
        user_data["Suggested_Place"] = suggested_place

        # Save to MongoDB
        places_collection.insert_one(user_data)
        logging.info("✅ User data inserted into MongoDB")

        # Retrain if new place
        if suggested_place not in place_map:
            logging.info("🔄 New place detected. Retraining model...")
            with model_lock:
                all_data = get_real_user_data()
                train_model(all_data)
                model, place_map = load_model_and_map()

        response = {
            "message": "User data added successfully!",
            "suggested_places": predictions["suggested_places"]
        }

        logging.info(f"📤 Response: {response}")
        return jsonify(response), 200

    except Exception as e:
        logging.error("❌ Error in /addUserData", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True, use_reloader=False)
