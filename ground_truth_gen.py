import os
import json
import requests
import re
import time
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Get Azure Blob Storage credentials from environment variables
AZURE_BLOB_URL = os.getenv("AZURE_BLOB_URL")
AZURE_SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN")
CONTAINER_NAME = "ganymedej"
BLOB_PREFIX = "findocs-testdata/"

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Use JSONL format for more efficient handling of large datasets
GROUND_TRUTH_FILE = "ground_truth_dataset.jsonl"
BATCH_SIZE = 10  # Number of entries to process before writing to disk

# API request configuration
REQUEST_TIMEOUT = 120  # seconds
MAX_RETRIES = 3
RETRY_DELAY_BASE = 5  # seconds

# Updated prompt with the new JSON format instructions.
MISTRAL_PROMPT = (
    "You are an AI trained to classify financial documents into four categories: Invoice, Receipt, Bank Statement, W-2 Form. "
    "First, extract the text from the document using OCR. Then, classify the document based on the extracted text, double-checking the classification to make sure it is correct. "
    "If you are unsure, set the docType.category as 'None of the above'."
    "Include only the relevant portions of the text that support the classification. Respond with ONLY the following JSON format: "
    "{\n"
    '    "file_name": "<file_name>",\n'
    '    "classification": {\n'
    '        "docType": "category",\n'
    '        "confidence": confidence_score\n'
    "    },\n"
    '    "relevant_text": "Relevant text that aided in classification"\n'
    "}"
)

# this is redundant & unnecessary - needs further testing to confirm & remove
def get_blob_sas_url(blob_client):
    """
    Generate a direct blob SAS URL for a specific blob.
    This creates a URL that Mistral API can access directly.
    """
    # Get the base URL without any SAS token
    base_url = blob_client.url
    
    # Construct the full URL with SAS token 
    direct_url = f"{base_url}"
    
    # Print the URL for manual review
    #print(f"File: {os.path.basename(blob_client.blob_name)}")
    print(f"SAS URL: {direct_url}")
    print("-" * 50)
    
    return direct_url


def call_mistral_api(file_name, file_type, sas_url):
    """
    Call the Mistral API for OCR extraction and document classification.
    Implements retry logic with exponential backoff for handling long response times.
    """
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Use the constant prompt and insert file name dynamically
    prompt_text = MISTRAL_PROMPT.replace("<file_name>", file_name)
    
    # Create the second content object based on file type
    if file_type == "image":
        second_content = {
            "type": "image_url",
            "image_url": sas_url
        }
    elif file_type == "pdf":
        second_content = {
            "type": "document_url",
            "document_url": sas_url
        }
    else:
        print(f"Unsupported file type: {file_name}")
        return None
    
    # Construct the payload following the exact structure
    payload = {
        "model": "mistral-small-latest",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    second_content
                ]
            }
        ]
    }
    
    # Implement retry logic with exponential backoff
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Calling Mistral API for {file_name} (attempt {attempt+1}/{MAX_RETRIES})...")
            
            # Use a longer timeout for large files
            response = requests.post(
                MISTRAL_API_URL, 
                headers=headers, 
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                print(f"Successfully received response for {file_name}")
                return response.json()
            elif response.status_code == 429:  # Rate limit exceeded
                retry_after = int(response.headers.get('Retry-After', RETRY_DELAY_BASE * (2 ** attempt)))
                print(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
            else:
                print(f"Mistral API call failed for file {file_name} with status code {response.status_code}")
                print("Response:", response.text)
                
                # Exponential backoff before retry
                if attempt < MAX_RETRIES - 1:  # Don't sleep after the last attempt
                    retry_delay = RETRY_DELAY_BASE * (2 ** attempt)
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                
        except requests.exceptions.Timeout:
            print(f"Request timed out for {file_name} after {REQUEST_TIMEOUT} seconds")
            if attempt < MAX_RETRIES - 1:
                retry_delay = RETRY_DELAY_BASE * (2 ** attempt)
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        except Exception as e:
            print(f"Error calling Mistral API for file {file_name}: {e}")
            if attempt < MAX_RETRIES - 1:
                retry_delay = RETRY_DELAY_BASE * (2 ** attempt)
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    print(f"All retry attempts failed for {file_name}")
    return None


def parse_mistral_response(response, local_file_name):
    """
    Extract the JSON object from the Mistral response and override the file_name 
    with the local file name tracked for each request.
    """
    try:
        content = response["choices"][0]["message"]["content"]
        
        # Try to parse JSON directly first
        try:
            result_json = json.loads(content)
            transformed = {
                "file_name": local_file_name,
                "ground_truth": result_json.get("classification", {}),
                "ocr_extracted_text": result_json.get("relevant_text", "")
            }
            return transformed
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from markdown code block
            match = re.search(r"```json\s*(\{.*\})\s*```", content, re.DOTALL)
            if match:
                json_str = match.group(1)
                result_json = json.loads(json_str)
                transformed = {
                    "file_name": local_file_name,
                    "ground_truth": result_json.get("classification", {}),
                    "ocr_extracted_text": result_json.get("relevant_text", "")
                }
                return transformed
            else:
                print(f"No valid JSON found in response for {local_file_name}")
                print(f"Response content: {content[:200]}...")  # Print first 200 chars
                return None
    except Exception as e:
        print(f"Error parsing Mistral response for {local_file_name}: {e}")
        return None


def append_to_jsonl(entry, file_path):
    """
    Append a single JSON entry to a JSONL file.
    Each line in the file is a complete JSON object.
    """
    try:
        with open(file_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        return True
    except Exception as e:
        print(f"Error appending to JSONL file: {e}")
        return False


def get_processed_files():
    """
    Read the JSONL file and extract the list of already processed file names.
    """
    processed_files = set()
    if os.path.exists(GROUND_TRUTH_FILE):
        with open(GROUND_TRUTH_FILE, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    processed_files.add(entry.get('file_name', ''))
                except json.JSONDecodeError:
                    continue
    return processed_files


def main():
    # Connect to Azure Blob Storage using SAS token
    blob_service_client = BlobServiceClient(account_url=AZURE_BLOB_URL, credential=AZURE_SAS_TOKEN)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    
    # List all blobs under the specified prefix.
    blobs = list(container_client.list_blobs(name_starts_with=BLOB_PREFIX))
    
    if not blobs:
        print(f"No files found with prefix '{BLOB_PREFIX}' in container '{CONTAINER_NAME}'")
        return
    
    # Get list of already processed files to avoid reprocessing
    processed_files = get_processed_files()
    
    # Filter out already processed files
    blobs_to_process = [blob for blob in blobs if os.path.basename(blob.name) not in processed_files]
    
    if not blobs_to_process:
        print(f"All {len(blobs)} files have already been processed.")
        return
    
    print(f"Found {len(blobs_to_process)} new files to process out of {len(blobs)} total files")
    
    # Create the ground truth file if it doesn't exist
    if not os.path.exists(GROUND_TRUTH_FILE):
        open(GROUND_TRUTH_FILE, 'w').close()
    
    # Process files with progress bar
    batch_entries = []
    successful_count = 0
    
    for blob in tqdm(blobs_to_process, desc="Processing files"):
        file_name = os.path.basename(blob.name)
        print(f"\nProcessing file: {file_name}")
        blob_client = container_client.get_blob_client(blob)
        sas_url = get_blob_sas_url(blob_client)
        
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            file_type = "image"
            file_data = sas_url
        elif file_name.lower().endswith(".pdf"):
            file_type = "pdf"
            file_data = sas_url
        else:
            print(f"Skipping unsupported file type: {file_name}")
            continue
        
        # Process the file
        mistral_response = call_mistral_api(file_name, file_type, file_data)
        if not mistral_response:
            continue
        
        ground_truth_entry = parse_mistral_response(mistral_response, file_name)
        if ground_truth_entry:
            # Add to batch
            batch_entries.append(ground_truth_entry)
            successful_count += 1
            print(f"Successfully processed: {file_name}")
            
            # Write to file if batch size reached or last item
            if len(batch_entries) >= BATCH_SIZE or blob == blobs_to_process[-1]:
                for entry in batch_entries:
                    append_to_jsonl(entry, GROUND_TRUTH_FILE)
                print(f"Wrote batch of {len(batch_entries)} entries to {GROUND_TRUTH_FILE}")
                batch_entries = []  # Clear the batch
        else:
            print(f"Failed to parse response for file {file_name}")
    
    print(f"\nProcessing complete. {successful_count} files successfully processed.")
    print(f"Results saved to {GROUND_TRUTH_FILE} in JSON Lines format.")


if __name__ == "__main__":
    main()
