import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import time

load_dotenv() # reads from .env in ./rag-the-facts
    
def query_gpt4(note, prompt, deployment='shc-gpt-4o'):
    # Initialize the Azure OpenAI client
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    try:
        # Make the API call to get the response from GPT-4
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "user", "content": prompt + note}
            ]
        )
    except Exception as e:
        retry_time = 30  # Default retry time
        if hasattr(e, 'response') and hasattr(e.response, 'headers'):
            retry_time = int(e.response.headers.get('Retry-After', retry_time))
        print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return query_gpt4(note, prompt)
    # Return the content of the response
    return response.choices[0].message.content

