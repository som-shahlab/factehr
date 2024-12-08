import argparse
import os
import json
import time
from pathlib import Path
from google.cloud import bigquery, storage
from datetime import datetime
import subprocess
import glob


# ----------- Utility Functions -----------

def merge_jsonl_files(input_folder, output_file):
    """
    Merges all JSONL files from a given folder into a single file.
    
    Args:
        input_folder (str): Directory containing JSONL files.
        output_file (str): Path to save the merged JSONL file.
    """
    with open(output_file, 'w') as outfile:
        for filename in glob.glob(f"{input_folder}/*.jsonl"):
            with open(filename, 'r') as infile:
                outfile.write(infile.read())
    print(f"Merged JSONL files into {output_file}")


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
                custom_id = record["metadata"].get("custom_id", "unknown")
                messages = record.get("messages", [])
                prompt = messages[0]["content"] if messages else "No prompt provided"
                
                input_data.append({
                    "custom_id": custom_id,
                    "content": prompt
                })
            except Exception as e:
                print(f"Error processing record: {str(e)}")
    return input_data


def store_in_bigquery(project_id, dataset_id, instances, model_name):
    """
    Stores the input data in a BigQuery table and submits a batch prediction job via Vertex AI.
    
    Args:
        project_id (str): Google Cloud Project ID.
        dataset_id (str): BigQuery Dataset ID.
        instances (list): List of dictionaries with 'custom_id' and 'content'.
        model_name (str): Vertex AI model name.
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
            ],
            "system_instruction": {
                "parts": [
                    {
                        "text": "You are tasked with determining whether the premise entails the hypothesis."
                    }
                ]
            }
        }
        json_data.append(request)

    # Initialize BigQuery client
    client = bigquery.Client()

    schema = [bigquery.SchemaField("request", "JSON")]
    table = bigquery.Table(f"{project_id}.{dataset_id}.{table_id}", schema=schema)
    table = client.create_table(table)  # Create the table in BigQuery

    rows_to_insert = [{"request": json.dumps(item)} for item in json_data]
    errors = client.insert_rows_json(table, rows_to_insert)  # Insert rows into BigQuery
    print(f"Data inserted into BigQuery table: {table_id}")

    # Create and submit a batch prediction job
    submit_vertex_batch_prediction(project_id, dataset_id, table_id, model_name)

    return errors


def submit_vertex_batch_prediction(project_id, dataset_id, table_id, model_name):
    """
    Submits a batch prediction job to Vertex AI using the input data from BigQuery.
    
    Args:
        project_id (str): Google Cloud Project ID.
        dataset_id (str): BigQuery Dataset ID.
        table_id (str): ID of the BigQuery table containing the batch data.
        model_name (str): Vertex AI model name for batch prediction.
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
        "outputConfig": {
            "predictionsFormat": "bigquery",
            "bigqueryDestination": {
                "outputUri": f"bq://{project_id}.{dataset_id}.{table_id}_prediction"
            }
        }
    }
    temp_json_file = f"temp_{current_time}.json"
    with open(temp_json_file, 'w') as f:
        json.dump(post_json_body, f, indent=2)
    # Submit the batch prediction job via curl
    curl_command = f"""curl -X POST \
        -H "Authorization: Bearer $(gcloud auth print-access-token)" \
        -H "Content-Type: application/json; charset=utf-8" \
        -d @{temp_json_file} \
        "https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/batchPredictionJobs"
    """
    try:
        result = subprocess.run(curl_command, shell=True, check=True, capture_output=True, text=True)
        print("Batch prediction job submitted successfully.")
        print("Response:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error submitting batch prediction job:")
        print("Exit code:", e.returncode)
        print("Error output:", e.stderr)


# ----------- Main Program -----------

def main(args):
    start_time = time.time()
    
    # Merge JSONL files into one
    merged_jsonl_file = Path(args.input_folder) / "merged_timelines.jsonl"
    merge_jsonl_files(args.input_folder, merged_jsonl_file)
    
    print("Preparing input data...")
    input_data = prepare_input_data(merged_jsonl_file, args.max_samples)
    print(f"Processed {len(input_data)} samples")

    if args.max_samples:
        print(f"Limited to {args.max_samples} samples for testing")

    # Store the input data in BigQuery and submit for batch prediction
    store_in_bigquery(args.project_id, args.dataset_id, input_data, args.model_name)
    
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")


# ----------- Argument Parser -----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction on entailment JSONL files.")
    
    parser.add_argument("--project_id", type=str, required=True, help="Google Cloud project ID")
    parser.add_argument("--dataset_id", type=str, required=True, help="BigQuery Dataset ID")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing JSONL files")
    parser.add_argument("--model_name", type=str, required=True, help="Vertex AI model name")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (optional)")
    
    args = parser.parse_args()

    main(args)
