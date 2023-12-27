from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")

print(MONGODB_URI)

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
