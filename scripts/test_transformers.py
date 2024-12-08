import subprocess
import json
import os

# Create a small dataset for testing
# sample_data = [
#     {
#         "metadata": {"uid": "test_1"},
#         "messages": [{"role": "user", "content": "Explain what AI is."}]
#     },
#     {
#         "metadata": {"uid": "test_2"},
#         "messages": [{"role": "user", "content": "Tell me a joke about computers."}]
#     }
# ]

# # Write the sample data to a JSONL file
sample_data_file = "/local-scratch/shahlab/akshaysw/just-the-facts/data/datasets/prompted_sampled/entailment_test.jsonl"
# with open(sample_data_file, "w") as file:
#     for item in sample_data:
#         file.write(json.dumps(item) + "\n")

# Define output file and generation parameters
output_file = "/local-scratch/shahlab/akshaysw/just-the-facts/data/datasets/prompted_sampled/entailment_test_results.jsonl"
generation_params = {
    "generation": {
        "temperature": 1,
        "top_p": 1
    }
}

# Write the generation parameters to a JSON file
generation_params_file = "/local-scratch/shahlab/akshaysw/just-the-facts/src/factehr/clients/generation_params.json"
with open(generation_params_file, "w") as file:
    json.dump(generation_params, file)

# Command to run the transformers_api.py script with the small dataset
command = [
    "python", "src/factehr/clients/transformers_api.py",
    "--path_to_prompted_dataset", sample_data_file,
    "--path_to_output_file", output_file,
    "--model_name_or_path", "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "--generation_config", generation_params_file,
    "--dynamic_batching", "40000"
]

# Run the command and print the output in real-time
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Read stdout and stderr line by line and print
for stdout_line in iter(process.stdout.readline, ""):
    print(stdout_line, end="")  # print stdout in real-time

for stderr_line in iter(process.stderr.readline, ""):
    print(stderr_line, end="")  # print stderr in real-time

# Wait for the process to finish and get the exit status
process.stdout.close()
process.stderr.close()
return_code = process.wait()

if return_code != 0:
    print(f"Process failed with return code: {return_code}")
else:
    print(f"Process completed successfully with return code: {return_code}")

# Check the output file
if os.path.exists(output_file):
    print(f"Output written to {output_file}")
    with open(output_file, "r") as file:
        for line in file:
            print(json.loads(line))
else:
    print("No output file generated.")
