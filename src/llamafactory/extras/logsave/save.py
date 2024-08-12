import os
import time
from typing import Optional

from pymongo import MongoClient

from .mongodb import default_client


# key of the environment variable that contains the MongoDB URI
db_key = "MONGODB_DB"
# default MongoDB database name
default_db = "training"
# key of the environment variable that contains the MongoDB collection name
collection_key = "MONGODB_COLLECTION"
# default MongoDB collection name
default_collection = "metrics"
ckpt_collection_key = "MONGODB_CKPT_COLLECTION"
default_ckpt_collection = "checkpoints"


class LogSaver:
    mongo_client: Optional[MongoClient] = None

    def __init__(self):
        self.mongo_client = default_client()
        db_name = os.environ.get(db_key, default_db)
        self.db = self.mongo_client[db_name]
        collection_name = os.environ.get(collection_key, default_collection)
        self.collection = self.db[collection_name]
        self.ckpt_collection = self.db[os.environ.get(ckpt_collection_key, default_ckpt_collection)]

    def save(self, log_entries: dict):
        """Save task logs to MongoDB

        Args:
            log_entries (dict): task logs, e.g. {'epoch': 1, 'loss': 0.1}
        """
        # add task_id to log entries
        task_id = os.environ.get("TASK_ID", "unknown")
        timestamp = time.time()
        log_entries["task_id"] = task_id
        log_entries["created_at"] = int(timestamp)
        log_entries["DeleteAt"] = 0
        self.collection.insert_one(log_entries)

    def list_checkpoints(self, output_dir: str):
        try:
            if not os.path.exists(output_dir):
                print(f"Directory {output_dir} does not exist, skip listing checkpoints.")
                return
            task_id = os.environ.get("TASK_ID", None)
            if not task_id:
                print("TASK_ID is not set, skip listing checkpoints.")
                return
            # find all dirs start with "checkpoint-"
            record = self.ckpt_collection.find_one({"task_id": task_id})
            if record and "checkpoints" in record and len(record["checkpoints"]) > 0:
                print(f"Checkpoints already listed for task {task_id}, skip listing.")
                return

            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            self.ckpt_collection.insert_one({"task_id": task_id, "checkpoints": checkpoints, "DeleteAt": 0})
        except Exception as e:
            print(f"Failed to list checkpoints: {e}")


saver: Optional[LogSaver] = None


def save_logs(log_entries: dict):
    """Save logs to MongoDB

    Args:
        log_entries (dict): training logs, e.g. {'epoch': 1, 'loss': 0.1}
    """
    global saver
    task_id = os.environ.get("TASK_ID")
    if not task_id:
        return
    try:
        if saver is None:
            saver = LogSaver()
        saver.save(log_entries)
    except Exception as e:
        print(f"Failed to save logs: {e}")


def list_checkpoints(output_dir: str):
    global saver
    try:
        if saver is None:
            saver = LogSaver()
        saver.list_checkpoints(output_dir)
    except Exception as e:
        print(f"Failed to list checkpoints: {e}")
