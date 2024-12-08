import base64
import vertexai
import time
from ipdb import set_trace
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models


from google.api_core import exceptions
from google.api_core.retry import Retry

_MY_RETRIABLE_TYPES = [
   exceptions.TooManyRequests,  # 429
   exceptions.InternalServerError,  # 500
   exceptions.BadGateway,  # 502
   exceptions.ServiceUnavailable,  # 503
]

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

# safety_settings = {
#     generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
#     generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
#     generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
#     generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
# }

def medlm(note, prompt):

    client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}
    client = aiplatform.gapic.PredictionServiceClient(
        client_options=client_options
    )
    instance_dict = {
        "content": prompt + note
    }
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]
    parameters_dict = {
        "candidateCount": 1,
        "maxOutputTokens": 8192,
        "temperature": 0.2,
        "topP": 0.8,
        "topK": 40
    }
    parameters = json_format.ParseDict(parameters_dict, Value())

    try: 
        response = client.predict(
            endpoint="projects/som-nero-phi-nigam-starr/locations/us-central1/publishers/google/models/medlm-medium", instances=instances, parameters=parameters
        )

    except Exception as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 30
        print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return medlm(note, prompt)
    
    predictions = response.predictions
    #metadatas = response.metadata
    
    for prediction in predictions:
        return prediction['content']


def gemini(note, prompt):

    try: 
        vertexai.init(project="som-nero-nigam-starr", location="us-central1")
        model = GenerativeModel(
        "gemini-1.5-pro-001",
        )

        

        responses = model.generate_content(
            [prompt + note],
            generation_config=generation_config,
            #safety_settings=safety_settings,
            stream=True,
        )

        part_response = []
        for response in responses:
            #print(response.text, end="")
            try:
                part_response.append(response.text)
            except ValueError:
                part_response = "SAFETY HAZARD"

        return part_response


    except Exception as e:
            retry_time = e.retry_after if hasattr(e, "retry_after") else 30
            print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            return gemini(note, prompt)
    

