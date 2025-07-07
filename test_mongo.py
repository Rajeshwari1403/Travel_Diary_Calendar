from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
    client.server_info()  # Trigger connection
    print("MongoDB connected successfully!")
except Exception as e:
    print("Connection failed:", e)
