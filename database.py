from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from utils import generate_random_string
import os

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")

client = MongoClient(MONGODB_URI, server_api=ServerApi("1"))

try:
    client.admin.command("ping")
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


def create_account(username, email, password):
    COLLECTION_NAME = "user_accounts"
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    user = {
        "username": username,
        "email": email,
        "password": password,
    }

    return collection.insert_one(user)


def find_account(email):
    COLLECTION_NAME = "user_accounts"
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    return collection.find_one({"email": email})


def create_session_id(email, password):
    COLLECTION_NAME = "sessions"
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    session_id = generate_random_string()

    session = {
        "email": email,
        "password": password,
        "session_id": session_id,
    }

    collection.insert_one(session)

    return session_id


def check_session_id(session_id):
    COLLECTION_NAME = "sessions"
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    return collection.find_one({"session_id": session_id})


def delete_session_id(session_id):
    COLLECTION_NAME = "sessions"
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    return collection.delete_one({"session_id": session_id})
