import os
import json
import time
import requests
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm
from dotenv import load_dotenv
import argparse

# Load environment variables from .env file
load_dotenv()

# --- Azure Blob Storage Configuration ---
AZURE_BLOB_URL = os.getenv("AZURE_BLOB_URL")
AZURE_SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "ganymedej")
BLOB_PREFIX = os.getenv("BLOB_PREFIX", "findocs-testdata/")

# --- Document Intelligence Configuration ---
# Example: https://ganymedej.cognitiveservices.azure.com
DOC_INTELLIGENCE_ENDPOINT = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
DOC_INTELLIGENCE_KEY = os.getenv("DOC_INTELLIGENCE_KEY")
MODEL_ID = os.getenv("DOC_INTELLIGENCE_MODEL_ID")  # e.g., md-doc-classifier

# --- Other Configurations ---
PREDICTIONS_FILE = "mut_classification_predictions_dataset.jsonl"
TEST_PREDICTIONS_FILE = "test_classification_result.jsonl"
BATCH_SIZE = 10
POLL_INTERVAL = 3         # seconds between GET calls
MAX_GET_RETRIES = 3       # maximum number of retries for each GET call
MAX_POST_RETRIES = 3      # maximum number of retries for POST submissions
POST_RETRY_DELAY_BASE = 5 # seconds base delay for POST retries
API_VERSION = "2024-11-30" # Azure Document Intelligence API version
REQUEST_TIMEOUT = 30      # seconds timeout for requests

def get_blob_sas_url(blob_client):
    """
    Generate a SAS URL for the given blob client.
    This URL will be used by the Document Intelligence service to access the blob.
    """
    sas_url = blob_client.url
    
    # Safely extract the blob name
    try:
        if hasattr(blob_client, 'blob_name'):
            blob_name = blob_client.blob_name
        else:
            # Extract from URL if blob_name attribute is not available
            blob_url = str(blob_client.url).split('?')[0]
            blob_name = os.path.basename(blob_url)
            
        print(f"SAS URL for {blob_name}: {sas_url}")
    except Exception as e:
        print(f"Warning: Could not extract blob name: {e}")
        print(f"Using full SAS URL: {sas_url}")
        
    print("-" * 50)
    return sas_url

def submit_document_intelligence(file_name, file_type, sas_url):
    """
    Submit a POST request to the custom classifier endpoint using the SAS URL.
    Returns the Operation-Location URL if successful.
    """
    analyze_url = f"{DOC_INTELLIGENCE_ENDPOINT}/documentintelligence/documentClassifiers/{MODEL_ID}:analyze?api-version={API_VERSION}"
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": DOC_INTELLIGENCE_KEY
    }
    payload = {"urlSource": sas_url}
    
    attempt = 0
    while attempt < MAX_POST_RETRIES:
        try:
            print(f"Submitting analysis request for {file_name} (attempt {attempt+1}/{MAX_POST_RETRIES})...")
            post_response = requests.post(analyze_url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            if post_response.status_code != 202:
                print(f"Error: Analysis request for {file_name} failed with status code {post_response.status_code}. Response: {post_response.text}")
                attempt += 1
                time.sleep(POST_RETRY_DELAY_BASE * (2 ** attempt))
                continue
            operation_url = post_response.headers.get("Operation-Location")
            if not operation_url:
                print(f"Error: Operation-Location header not found in response for {file_name}")
                attempt += 1
                time.sleep(POST_RETRY_DELAY_BASE * (2 ** attempt))
                continue
            print(f"Received Operation-Location for {file_name}: {operation_url}")
            return operation_url
        except requests.exceptions.Timeout:
            print(f"Timeout submitting analysis request for {file_name}. Retrying...")
            attempt += 1
            time.sleep(POST_RETRY_DELAY_BASE * (2 ** attempt))
        except Exception as e:
            print(f"Exception submitting analysis request for {file_name}: {e}")
            attempt += 1
            time.sleep(POST_RETRY_DELAY_BASE * (2 ** attempt))
    print(f"Exceeded maximum POST retries for {file_name}.")
    return None

def poll_get_result(operation_url, file_name, headers, max_retries=MAX_GET_RETRIES, poll_interval=POLL_INTERVAL):
    """
    Poll the Operation-Location GET URL using a FIFO approach.
    Retries up to max_retries times if the result is not yet 'succeeded'.
    Returns the JSON analysis result if succeeded, else None.
    """
    for attempt in range(1, max_retries + 1):
        print(f"Polling GET URL for {file_name} (attempt {attempt}/{max_retries})...")
        try:
            get_response = requests.get(operation_url, headers=headers, timeout=REQUEST_TIMEOUT)
            if get_response.status_code != 200:
                print(f"Error: GET request for {file_name} returned status code {get_response.status_code}")
                time.sleep(poll_interval)
                continue

            result_json = get_response.json()
            status = result_json.get("status", "").lower()
            if status == "succeeded":
                print(f"Analysis succeeded for {file_name}")
                return result_json
            elif status == "failed":
                print(f"Analysis failed for {file_name}")
                return None
            else:
                print(f"Status for {file_name}: {status}. Retrying after {poll_interval} seconds...")
                time.sleep(poll_interval)
        except requests.exceptions.Timeout:
            print(f"Timeout polling GET result for {file_name}. Retrying...")
            time.sleep(poll_interval)
        except Exception as e:
            print(f"Exception polling GET result for {file_name}: {e}")
            time.sleep(poll_interval)
    print(f"Exceeded maximum GET retries for {file_name} with no successful result.")
    return None

def parse_di_response(result, file_name):
    """
    Parse the analysis result to extract the predicted docType and confidence.
    Constructs an entry with the file name and classification result.
    """
    try:
        analyze_result = result.get("analyzeResult", {})
        documents = analyze_result.get("documents", [])
        if documents and len(documents) > 0:
            document = documents[0]
            doc_type = document.get("docType", "unknown")
            confidence = document.get("confidence", 0.0)
        else:
            print(f"No documents found in analysis result for {file_name}")
            return None
        
        prediction_entry = {
            "file_name": file_name,
            "ground_truth": {
                "docType": doc_type,
                "confidence": confidence
            },
            "ocr_extracted_text": ""  # Document Intelligence does not provide OCR text in this response
        }
        return prediction_entry
    except Exception as e:
        print(f"Error parsing analysis result for {file_name}: {e}")
        return None

def append_to_jsonl(entry, file_path):
    """
    Append a JSON entry to a JSONL file.
    """
    try:
        with open(file_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")
        return True
    except Exception as e:
        print(f"Error appending to {file_path}: {e}")
        return False

def get_processed_files(file_path):
    """
    Load already processed file names from the JSONL predictions file.
    """
    processed_files = set()
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    processed_files.add(entry.get("file_name", ""))
                except json.JSONDecodeError:
                    continue
    return processed_files

def prepare_output_file(file_path, test_mode=False):
    """
    Prepare the output file by creating a new file or clearing the existing one.
    """
    with open(file_path, 'w') as f:
        pass
    if test_mode:
        print(f"Test mode: Cleared existing predictions file: {file_path}")
    else:
        print(f"Created new predictions file: {file_path}")

def main():
    """
    Main function to process documents using Azure Document Intelligence.
    Connects to Azure Blob Storage, generates SAS URLs for each blob,
    sends POST requests to get the Operation-Location URLs, and then polls
    the GET endpoints in FIFO order to obtain classification results.
    The classification results are then saved to a JSONL file.
    """
    global PREDICTIONS_FILE
    
    parser = argparse.ArgumentParser(description='Azure Document Intelligence Financial Document Classifier')
    parser.add_argument('--test-file', type=str, help='Process only a single file (specify blob name)')
    args = parser.parse_args()
    
    print("Starting Azure Document Intelligence classification process...")
    
    # Validate environment variables
    if not all([AZURE_BLOB_URL, AZURE_SAS_TOKEN, DOC_INTELLIGENCE_ENDPOINT, DOC_INTELLIGENCE_KEY, MODEL_ID]):
        print("Error: Missing required environment variables. Please check your .env file.")
        print("Required variables: AZURE_BLOB_URL, AZURE_SAS_TOKEN, DOC_INTELLIGENCE_ENDPOINT, DOC_INTELLIGENCE_KEY, DOC_INTELLIGENCE_MODEL_ID")
        return
    
    try:
        # Connect to Azure Blob Storage
        print(f"Connecting to Azure Blob Storage: {AZURE_BLOB_URL}")
        blob_service_client = BlobServiceClient(account_url=AZURE_BLOB_URL, credential=AZURE_SAS_TOKEN)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        print(f"Successfully connected to container: {CONTAINER_NAME}")
        
        if args.test_file:
            test_blob_name = args.test_file
            if not test_blob_name.startswith(BLOB_PREFIX) and BLOB_PREFIX:
                test_blob_name = BLOB_PREFIX + test_blob_name
            print(f"Test mode: Processing only file '{test_blob_name}'")
            
            # Create a proper blob client for the test file
            blob_client = container_client.get_blob_client(test_blob_name)
            # Verify the blob exists
            blob_client.get_blob_properties()  # Raises exception if not found
            
            # Create a dummy blob object with a name attribute and the blob client
            class DummyBlob:
                def __init__(self, name, client):
                    self.name = name
                    self.client = client
            
            blobs = [DummyBlob(test_blob_name, blob_client)]
        else:
            blobs = list(container_client.list_blobs(name_starts_with=BLOB_PREFIX))
            if not blobs:
                print(f"No files found with prefix '{BLOB_PREFIX}' in container '{CONTAINER_NAME}'")
                return
    except Exception as e:
        print(f"Error connecting to Azure Blob Storage: {e}")
        return

    # Determine files to process (skip already processed in non-test mode)
    if not args.test_file:
        processed_files = get_processed_files(PREDICTIONS_FILE)
        blobs_to_process = [blob for blob in blobs if os.path.basename(blob.name) not in processed_files]
        if not blobs_to_process:
            print("No new files to process.")
            return
        print(f"Found {len(blobs_to_process)} new files to process out of {len(blobs)} total files.")
    else:
        blobs_to_process = blobs
        print("Test mode: Skipping check for previously processed files")
        PREDICTIONS_FILE = TEST_PREDICTIONS_FILE
        print(f"Test mode: Results will be saved to {PREDICTIONS_FILE}")

    prepare_output_file(PREDICTIONS_FILE, test_mode=args.test_file is not None)

    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": DOC_INTELLIGENCE_KEY
    }

    # ---------------------------
    # STEP 1: Submit POST requests for all files
    # ---------------------------
    print(f"\n{'='*20} STEP 1: Submitting POST requests {'='*20}")
    print(f"Submitting {len(blobs_to_process)} documents for classification...")
    file_to_operation = {}  # Dictionary to map file names to GET operation URLs

    for blob in tqdm(blobs_to_process, desc="Submitting POST requests"):
        file_name = os.path.basename(blob.name)
        print(f"\nProcessing file: {file_name}")
        
        try:
            # Get the blob client - either from the dummy blob or create a new one
            if hasattr(blob, 'client'):
                blob_client = blob.client
                print(f"Using existing blob client for {file_name}")
            else:
                print(f"Creating new blob client for {file_name}")
                blob_client = container_client.get_blob_client(blob.name)
                
            sas_url = get_blob_sas_url(blob_client)
            
            # Determine file type based on extension
            if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif")):
                file_type = "image"
            elif file_name.lower().endswith((".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls")):
                file_type = "document"
            else:
                print(f"Skipping unsupported file type: {file_name}")
                continue
            
            print(f"Submitting {file_type} file: {file_name} to Azure Document Intelligence...")
            operation_url = submit_document_intelligence(file_name, file_type, sas_url)
            if operation_url:
                file_to_operation[file_name] = operation_url
            else:
                print(f"Submission failed for {file_name}")
        except Exception as e:
            print(f"Error processing {file_name} during POST submission: {e}")

    if not file_to_operation:
        print("No POST submissions succeeded. Exiting.")
        return

    # ---------------------------
    # STEP 2: Poll GET endpoints in FIFO order
    # ---------------------------
    print(f"\n{'='*20} STEP 2: Polling GET endpoints {'='*20}")
    print(f"Successfully submitted {len(file_to_operation)} documents. Starting to poll for results...")
    batch_entries = []
    successful_count = 0
    error_count = 0

    for file_name, operation_url in file_to_operation.items():
        print(f"\nPolling results for {file_name} with URL: {operation_url}")
        result_json = poll_get_result(operation_url, file_name, headers)
        if result_json:
            print(f"Successfully received classification results for {file_name}")
            prediction_entry = parse_di_response(result_json, file_name)
            if prediction_entry:
                batch_entries.append(prediction_entry)
                successful_count += 1
                print(f"Successfully processed: {file_name}")
                print(f"Classification: {prediction_entry['ground_truth']['docType']} with confidence: {prediction_entry['ground_truth']['confidence']:.3f}")
            else:
                error_count += 1
                print(f"Failed to parse response for: {file_name}")
        else:
            error_count += 1
            print(f"No analyzed results for: {file_name}")

        if len(batch_entries) >= BATCH_SIZE:
            for entry in batch_entries:
                append_to_jsonl(entry, PREDICTIONS_FILE)
            print(f"Wrote batch of {len(batch_entries)} entries to {PREDICTIONS_FILE}")
            batch_entries = []

    # Write any remaining entries
    if batch_entries:
        for entry in batch_entries:
            append_to_jsonl(entry, PREDICTIONS_FILE)
        print(f"Wrote final batch of {len(batch_entries)} entries to {PREDICTIONS_FILE}")

    print(f"\n{'='*20} SUMMARY {'='*20}")
    print(f"Processing complete. {successful_count} files successfully processed, {error_count} errors.")
    print(f"Results saved to {PREDICTIONS_FILE}")

if __name__ == "__main__":
    main()
