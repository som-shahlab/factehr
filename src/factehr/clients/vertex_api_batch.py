import argparse
import os
import json
import time
from pathlib import Path
from google.cloud import bigquery
from datetime import datetime
import glob
import requests
from google.auth.transport.requests import Request
from google.oauth2 import id_token
import google.auth
from factehr.utils.estimate_llm_api_cost import estimate_request_limits 
import math


# ----------- Utility Functions -----------

def prepare_input_data(merged_jsonl_file, max_samples=None):
    """
    Processes the merged JSONL file and extracts relevant fields for batch prediction.
    
    Args:
        merged_jsonl_file (str): Path to the merged JSONL file.
        max_samples (int): Maximum number of samples to process (for testing).
    
    Returns:
        list: List of dictionaries containing 'custom_id' and 'content' for each prompt.
    """
    input_data = []
    with open(merged_jsonl_file, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                record = json.loads(line)
                metadata = record["metadata"]
                messages = record.get("messages", [])
                prompt = messages[0]["content"] if messages else "No prompt provided"
                
                input_data.append({
                    "metadata": metadata,
                    "content": prompt
                })
            except Exception as e:
                print(f"Error processing record: {str(e)}")
    return input_data


def store_in_bigquery(project_id, dataset_id, instances):
    """
    Stores the input data in a BigQuery table and submits a batch prediction job via Vertex AI.
    
    Args:
        project_id (str): Google Cloud Project ID.
        dataset_id (str): BigQuery Dataset ID.
        instances (list): List of dictionaries with 'custom_id' and 'content'.
        
    Returns: table_id (str)
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_id = f"batch_storage_{current_time}"

    # Prepare data for BigQuery insertion
    json_data = []
    for instance in instances:
        request = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": instance["content"]}]
                }
            ]
        }
        
        metadata = instance["metadata"]
        
        # Combine request and metadata into a single dictionary
        json_data.append({
            "request": request,
            "metadata": metadata
        })

    # Initialize BigQuery client
    client = bigquery.Client()

    print("client initialized")
    # Define schema for both 'request' and 'metadata' columns
    schema = [
        bigquery.SchemaField("request", "JSON"),
        bigquery.SchemaField("metadata", "JSON")
    ]
    
    # Create the table with the appropriate schema
    table = bigquery.Table(f"{project_id}.{dataset_id}.{table_id}", schema=schema)
    table = client.create_table(table)  # Create the table in BigQuery

    print("table created")
    print(table_id)
    
    chunk_size = 200 # empirical testing showed that you get stuck when setting this too high (eg. >=1000)
    
    # Insert rows into BigQuery
    total_rows = len(json_data)
    num_chunks = math.ceil(total_rows / chunk_size) # chunks of 2000
    
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_rows)
        chunk = json_data[start:end]
        # Prepare rows to insert
        rows_to_insert = [{"request": json.dumps(item["request"]), "metadata": json.dumps(item["metadata"])} for item in chunk]
        # Insert the chunk of rows into BigQuery
        errors = client.insert_rows_json(table, rows_to_insert)
        if errors:
            print(f"Errors occurred while inserting rows {start} to {end}: {errors}")
        else:
            print(f"Successfully inserted rows {start} to {end}.")

    if errors:
        print("Errors while inserting rows: ", errors)
    else:
        print(f"Data successfully inserted into BigQuery table: {table_id}")

    return table_id



def check_batch_job_status(project_id, batch_id):
    """
    Checks the status of a Vertex AI batch prediction job using the batch ID.
    
    Args:
        project_id (str): Google Cloud Project ID.
        batch_id (str): Batch prediction job ID.
    
    Returns:
        str: The status of the job (e.g., "JOB_STATE_SUCCEEDED").
    """
    # Get the Google Cloud auth token
    auth_req = Request()
    credentials, _ = google.auth.default()
    credentials.refresh(auth_req)
    token = credentials.token
    # API URL
    url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/batchPredictionJobs/{batch_id}"
    # Headers
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    # Make the GET request
    response = requests.get(url, headers=headers)
    # Check response
    
    print(response.json())
    
    if response.status_code == 200:
        response_json = response.json()
        job_status = response_json.get("state", "UNKNOWN")
        print(f"Batch Job Status: {job_status}")
        return job_status
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None
    
def parse_and_write_jsonl(rows, output_file, model_name):
    """
    Parses rows from BigQuery results and writes them to a JSONL file in the desired format.
    
    Args:
        rows (list): List of BigQuery result rows.
        output_file (str): Path to the output .jsonl file.
        model_name (str): Model name used for the response generation.
    """
    with open(output_file, 'w') as f:
        for row in rows:
            try:
                # Assuming row is a dictionary with the following keys: status, time, request, metadata, response
                
                # Extract user prompt (from 'request' -> 'contents' -> 'role': 'user')
                user_message = row['request']['contents'][0]['parts'][0]['text']
                
                # Extract assistant response (from 'response' -> 'candidates' -> 'content')
                assistant_message = row['response']['candidates'][0]['content']['parts'][0]['text']
                
                # Extract model metadata (model version, safety ratings, token counts)
                model_metadata = {
                    "model_version": row['response'].get('modelVersion', ''),
                    "safety_ratings": row['response']['candidates'][0].get('safetyRatings', []),
                    "token_counts": row['response'].get('usageMetadata', {})
                }
                
                # Extract original metadata
                original_metadata = json.loads(row['metadata'])
                
                # Prepare the completion structure
                completion = [
                    {
                        "messages": [{"role": "user", "content": user_message}],
                        "model": model_name
                    },
                    {
                        "choices": [{"message": {"content": assistant_message, "role": "assistant"}}]
                    },
                    {
                        "metadata": {**original_metadata, **model_metadata}
                    }
                ]
                
                # Write each parsed completion as a line in the JSONL file
                f.write(json.dumps(completion) + '\n')
            except Exception as e:
                print(f"Error processing row: {e}")
    
    print(f"Results written to {output_file}")
    
def read_prediction_table(project_id, dataset_id, prediction_table_id):
    """
    Reads the prediction table from BigQuery once the job is complete.
    
    Args:
        project_id (str): Google Cloud Project ID.
        dataset_id (str): BigQuery Dataset ID.
        prediction_table_id (str): The BigQuery table ID of the prediction results.
    
    Returns:
        bigquery.Table: BigQuery Table containing the prediction results.
    """
    # Initialize BigQuery client
    client = bigquery.Client()

    # Construct full table ID
    table_id = f"{project_id}.{dataset_id}.{prediction_table_id}"
    
    # Fetch data from BigQuery table
    query = f"SELECT * FROM `{table_id}`"
    query_job = client.query(query)  # Run the query

    # Process the results
    results = query_job.result()  # Wait for the job to complete
    rows = list(results)
    
    print(f"Number of predictions: {len(rows)}")
       
    return rows

def submit_vertex_batch_prediction(project_id, dataset_id, table_id, model_name, generation_params, max_output_tokens):
    """
    Submits a batch prediction job to Vertex AI using the input data from BigQuery with specific model parameters.
    
    Args:
        project_id (str): Google Cloud Project ID.
        dataset_id (str): BigQuery Dataset ID.
        table_id (str): ID of the BigQuery table containing the batch data.
        model_name (str): Vertex AI model name for batch prediction.
        generation_params (dict): dictionary with keys for temperature and top_p
        max_output_tokens (int): Maximum number of output tokens for the model.
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_job_name = f"{project_id}.{dataset_id}.{table_id}_prediction"

    post_json_body = {
        "displayName": prediction_job_name,
        "model": f"publishers/google/models/{model_name}",
        "inputConfig": {
            "instancesFormat": "bigquery",
            "bigquerySource": {
                "inputUri": f"bq://{project_id}.{dataset_id}.{table_id}"
            }
        },
        "modelParameters": {
            "temperature": generation_params['generation']['temperature'],
            "topP": generation_params['generation']['top_p'],
            "maxOutputTokens": max_output_tokens
        },
        "outputConfig": {
            "predictionsFormat": "bigquery",
            "bigqueryDestination": {
                "outputUri": f"bq://{project_id}.{dataset_id}.{table_id}_prediction"
            }
        }
    }

    # Pass the JSON content directly to the batch function
    batch_id = make_batch_prediction_call(project_id, post_json_body)

    return batch_id



def make_batch_prediction_call(project_id, post_json_body):
    """
    Submits a batch prediction job to Vertex AI using the input data from BigQuery.
    Returns the batch id.
    
    Args:
        project_id (str): Google Cloud Project ID.
        post_json_body (dict): The JSON body containing the batch prediction request.
        
    Returns:
        batch_id (str)
    """
    # Get the Google Cloud auth token
    auth_req = Request()
    credentials, project_id = google.auth.default()
    credentials.refresh(auth_req)
    token = credentials.token

    # API URL
    url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/batchPredictionJobs"
    
    # Headers
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }

    # Make the POST request
    response = requests.post(url, headers=headers, json=post_json_body)

    # Check response
    if response.status_code == 200:
        response_json = response.json()
        batch_id = response_json.get('name', '').split('/')[-1]
        print("Batch prediction job submitted successfully.")
        print("Response:", response_json)
    else:
        print(f"Error {response.status_code}: {response.text}")
        batch_id = None
        
    return batch_id



# ----------- Main Program -----------

def main(args):
    start_time = time.time()
    
    generation_params = json.load(open(args.generation_config, "r"))
    
    if not args.prediction_table_id:
        if not args.table_id:
            # Merge JSONL files into one
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Prepare input data
            print("Preparing input data...")
            input_data = prepare_input_data(args.input_jsonl, args.max_samples)
            print(f"Processed {len(input_data)} samples")

            # Extract the prompts (content) from input_data and estimate the cost
            # Rough estimation concatenating all input content as one string
            prompts = [" ".join(item["content"]) for item in input_data]   
            
            try:
                estimate_request_limits(
                    user_model_name=args.model_name,
                    prompts=prompts,
                    tokens_per_minute=80000,  # Example value for Vertex AI models
                    max_tokens=args.max_new_tokens,  # Set according to max_new_tokens
                    max_cost_threshold=args.max_cost_threshold  # Set an appropriate threshold
                )
            except ValueError as e:
                print(f"Cost estimation error: {e}")
                exit(1)  # Exit if the cost exceeds the threshold

            if args.max_samples:
                print(f"Limited to {args.max_samples} samples for testing")

            # Store the input data in BigQuery
            table_id = store_in_bigquery(args.project_id, args.dataset_id, input_data)
        else:
            print("using existing table_id")
            table_id = args.table_id

        # Submit the batch prediction job
        print("Submitting batch prediction job...")
        batch_id = submit_vertex_batch_prediction(args.project_id, args.dataset_id, table_id, args.model_name,
                                                generation_params=generation_params,
                                                max_output_tokens=args.max_new_tokens)

        if not batch_id:
            print("Failed to submit the batch prediction job.")
            return

        # Check the status of the batch prediction job
        print(f"Checking the status of batch prediction job {batch_id}...")
        job_status = None
        retry_count = 0
        max_retries = 2  # Set maximum number of retries

        while job_status != "JOB_STATE_SUCCEEDED":
            job_status = check_batch_job_status(args.project_id, batch_id)
            
            if job_status == "JOB_STATE_FAILED":
                print("Batch job failed. Exiting...")
                return
            elif job_status == "JOB_STATE_CANCELLED":
                print("Batch job was cancelled. Exiting...")
                return
            elif job_status == "JOB_STATE_SUCCEEDED":
                print("Job completed successfully.")
                break
            else:
                print("Waiting for the job to complete...")
                time.sleep(30)  # Poll every 30 seconds
                
                # Increment the retry counter
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Max retries exceeded. Please check on job in console using batch id {batch_id} and table id {table_id}.")
                    return

        print("Batch job completed successfully.")

        # Read the prediction table from BigQuery
        prediction_table_id = f"{table_id}_prediction"
    else:
        prediction_table_id = args.prediction_table_id
    
    print(f"Reading predictions from BigQuery table {prediction_table_id}...")
    rows = read_prediction_table(args.project_id, args.dataset_id, prediction_table_id)

    if not rows:
        print("No predictions found in the table.")
        return

    # Write predictions to a JSONL file
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_jsonl_file = Path(args.output_folder) / f"{args.model_name}_{current_time}.jsonl"
    parse_and_write_jsonl(rows, output_jsonl_file, args.model_name)

    print(f"Total time taken: {time.time() - start_time:.2f} seconds")
    print(f"Predictions written to {output_jsonl_file}")


# ----------- Argument Parser -----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction on entailment JSONL files.")
    
    parser.add_argument("--project_id", type=str, required=True, help="Google Cloud project ID")
    parser.add_argument("--dataset_id", type=str, required=True, help="BigQuery Dataset ID")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder containing JSONL files")
    parser.add_argument("--model_name", type=str, required=True, help="Vertex AI model name")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (optional)")
    parser.add_argument("--generation_config", type=str, default=None, help="path to generation configs file")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="max new tokens")
    parser.add_argument("--table_id", type=str, default=None, help="table_id if already exists")
    parser.add_argument("--max_cost_threshold", type=int, help="max cost per batch in $", default=100)
    parser.add_argument("--prediction_table_id", type=str, help="prediction table_id if already exists", default=None)
        
    args = parser.parse_args()

    main(args)
