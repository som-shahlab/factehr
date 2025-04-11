import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import vertexai.preview.generative_models as generative_models
import time
import argparse
import json
from factehr.utils.estimate_llm_api_cost import estimate_request_limits 

# Define safety settings to control harmful content
safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

def generate(prompt, model_name, generation_config, max_tokens, max_retries=2, retry_delay=5):
    """
    Generates content using a specified model and generation configuration.
    
    Args:
        prompt (str): The input prompt for the model.
        model_name (str): The name of the model to use.
        generation_config (dict): The generation configuration for the model (e.g., max_output_tokens, temperature).
        max_retries (int): The maximum number of retries if an error occurs.
        retry_delay (int): The time to wait before retrying in case of failure.

    Returns:
        list: A list of generated response texts or an error message.
    """
    vertexai.init(project="...", location="us-central1")
    model = GenerativeModel(model_name)
    
    for attempt in range(max_retries):
        try:
            responses = model.generate_content(
                contents=prompt,
                generation_config=GenerationConfig(temperature=generation_config["generation"]["temperature"],
                                                   top_p=generation_config["generation"]["top_p"],
                                                   max_output_tokens=max_tokens),
                safety_settings=safety_settings,
                stream=True,
            )

            response_texts = []
            for response in responses:
                try:
                    response_texts.append(response.text)
                except ValueError as ve:
                    print(f"ValueError encountered: {ve}")
                    response_texts.append("SAFETY_HAZARD")
                except AttributeError as ae:
                    print(f"AttributeError encountered: {ae}")
                    continue

            if response_texts:
                return response_texts
            else:
                print(f"No valid responses generated on attempt {attempt + 1}")
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error occurred: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to generate content after {max_retries} attempts")
                return ["GENERATION_FAILED"]

    return ["GENERATION_FAILED"]


def process_jsonl(input_jsonl, output_jsonl, model_name, generation_config, max_tokens, max_retries=2, retry_delay=5):
    """
    Processes the input JSONL file, generates content, and writes the results to an output JSONL file.

    Args:
        input_jsonl (str): Path to the input JSONL file.
        output_jsonl (str): Path to the output JSONL file.
        model_name (str): The name of the model to use for generation.
        generation_config (dict): Configuration for generation (e.g., max tokens, temperature).
        max_retries (int): The maximum number of retries in case of failure.
        retry_delay (int): Delay between retries in case of failure.
    """
    with open(input_jsonl, "r") as infile, open(output_jsonl, "w") as outfile:
        for line in infile:
            record = json.loads(line)
            prompt = record["messages"][0]["content"]  # Extracting the user prompt from the input

            # Call the generation function
            responses = generate(
                prompt=prompt,
                model_name=model_name,
                generation_config=generation_config,
                max_retries=max_retries,
                retry_delay=retry_delay,
                max_tokens=max_tokens
            )
            
            # Concatenate all responses into a single string
            concatenated_responses = "".join(responses)

            # Create the completion structure to match the expected format
            completion = [
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "model": model_name
                },
                {
                    "choices": [{"message": {"content": concatenated_responses, "role": "assistant"}}]
                },
                {
                    "metadata": record.get("metadata", {})
                }
            ]
            # Write the completion result to the output JSONL file
            outfile.write(json.dumps(completion) + "\n")


def main():
    """
    Main function to handle command-line arguments and execute the content generation process.
    
    This script reads from a JSONL input file, generates content using a Vertex AI model, 
    and writes the generated content back to an output JSONL file.
    """
    # Initialize argparse
    parser = argparse.ArgumentParser(description="Generate content using Vertex AI Generative Model.")
    
    # Add command-line arguments
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to use.")
    parser.add_argument("--generation_config", type=str, required=True, help="Path to the JSON file containing generation configuration (e.g., max tokens, temperature).")
    parser.add_argument("--max_retries", type=int, default=2, help="Maximum number of retries in case of failure.")
    parser.add_argument("--retry_delay", type=int, default=5, help="Delay between retries in seconds.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="max new tokens")
    parser.add_argument("--max_cost_threshold", type=int, help="max cost per batch in $", default=100)
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Load generation config from the provided JSON file
    try:
        with open(args.generation_config, "r") as config_file:
            generation_config = json.load(config_file)
    except Exception as e:
        print(f"Error loading generation config: {e}")
        return
    
    with open(args.input_jsonl) as f:
        prompts = []
        for line in f:
            request = json.loads(line)
            # Join all the content from the "messages" into one string
            concatenated_prompt = " ".join([message["content"] for message in request["messages"]])
            prompts.append(concatenated_prompt)

    try:
        estimate_request_limits(
            user_model_name=args.model_name,
            prompts=prompts,
            max_tokens=args.max_new_tokens,
            max_cost_threshold=args.max_cost_threshold  
        )
    except ValueError as e:
        print(f"Cost estimation error: {e}")
        exit(1) 

    # Call the function to process the JSONL file
    process_jsonl(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        model_name=args.model_name,
        generation_config=generation_config,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        max_tokens=args.max_new_tokens
    )

if __name__ == "__main__":
    main()
 
