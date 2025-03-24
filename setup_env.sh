#!/bin/bash

# Function to prompt for input
get_input() {
    local prompt="$1"
    local var_name="$2"
    local current_value="${!var_name}"
    
    if [ -n "$current_value" ]; then
        read -p "$prompt (current: $current_value): " value
    else
        read -p "$prompt: " value
    fi
    
    if [ -n "$value" ]; then
        echo "$value"
    else
        echo "$current_value"
    fi
}

# Create or load existing .env file
touch .env
source .env

# Get Azure Form Recognizer credentials
echo "Please enter your Azure Document Intelligence credentials:"
endpoint=$(get_input "Enter your Azure Document Intelligence endpoint" DOC_INTELLIGENCE_ENDPOINT)
key=$(get_input "Enter your Azure Document Intelligence key" DOC_INTELLIGENCE_KEY)
model_id=$(get_input "Enter your Azure Document Intelligence Model ID" DOC_INTELLIGENCE_MODEL_ID)

# Get additional required credentials
mistral_key=$(get_input "Enter your Mistral API key" MISTRAL_API_KEY)

# Get Azure Blob Storage credentials
azure_blob_url=$(get_input "Enter your Azure Blob Storage URL (e.g., https://accountname.blob.core.windows.net)" AZURE_BLOB_URL)
azure_sas_token=$(get_input "Enter your Azure SAS token (without the URL)" AZURE_SAS_TOKEN)

# Write to .env file
cat > .env << EOL
DOC_INTELLIGENCE_ENDPOINT="$endpoint"
DOC_INTELLIGENCE_KEY="$key"
DOC_INTELLIGENCE_MODEL_ID="$model_id"
MISTRAL_API_KEY="$mistral_key"
AZURE_BLOB_URL="$azure_blob_url"
AZURE_SAS_TOKEN="$azure_sas_token"
EOL

echo "Environment variables have been updated in .env file"
