import torch
from ipdb import set_trace
import os
os.environ['TRANSFORMERS_CACHE'] = '/share/pi/nigam/'
import transformers
access_token = os.getenv('HF_TOKEN')

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=0,
)


def generate(note, prompt):
    # Define the stop token IDs
    terminators = [
        pipeline.tokenizer.eos_token_id,
        #pipeline.tokenizer.convert_tokens_to_ids(""),
    ]

    messages = [{"role": "user", "content": prompt + note}]
    
    # Generate the text
    outputs = pipeline(
        messages,
        max_new_tokens=8192,
        eos_token_id=terminators,  # Use the first stop token as eos_token_id
        do_sample=True,
        temperature=0.01,
        top_p=0.9,
    )


    # Extract and return the generated text
    result = outputs[0]["generated_text"]   
    try:
        generated_text = result[1]['content']
    except:
        generated_text = "ERROR"

    # Optionally, handle any manual stopping based on terminators
    # for terminator in terminators:
    #     stop_token = pipeline.tokenizer.decode([terminator])
    #     generated_text = generated_text.split(stop_token)[0]

    return generated_text



    

