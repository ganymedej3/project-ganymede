import json
import os
import time
from dotenv import load_dotenv
from functools import wraps

# ----------------------------
# JSONL File Operations
# ----------------------------
def load_jsonl(file_path):
    """Load a JSONL file and return a list of JSON objects."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def append_to_jsonl(file_path, record):
    """Append a record to a JSONL file."""
    with open(file_path, "a") as f:
        f.write(json.dumps(record) + "\n")

def get_processed_files(file_path):
    """Retrieve processed file names from a text file."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

# ----------------------------
# API Request Handling with Retry Logic
# ----------------------------
def retry_operation(func, *args, retries=3, delay=1, backoff=2, exceptions=(Exception,), **kwargs):
    """Execute a function with retry logic using exponential backoff."""
    attempt = 0
    current_delay = delay
    while attempt < retries:
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            attempt += 1
            if attempt >= retries:
                raise
            time.sleep(current_delay)
            current_delay *= backoff

def retry_decorator(retries=3, delay=1, backoff=2, exceptions=(Exception,)):
    """Decorator for retrying a function with exponential backoff."""
    def decorator_retry(func):
        @wraps(func)
        def wrapper_retry(*args, **kwargs):
            attempt = 0
            current_delay = delay
            while attempt < retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= retries:
                        raise
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper_retry
    return decorator_retry

# ----------------------------
# Configuration Loading
# ----------------------------
def load_config():
    """Load environment variables from a .env file."""
    load_dotenv()
    return os.environ