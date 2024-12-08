import json
import ast
from ipdb import set_trace

file_path = "/share/pi/nigam/rag-the-facts/datasets/sentences/gemini/mimiciii/progress_note/PROMPT1_sentence_recall.json"
output_path = "/share/pi/nigam/rag-the-facts/datasets/sentences/GPT4/mimiciii/progress_note/PROMPT1_sentence_recall.json"
data = []
try: 
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

except:
    print("Deleted line : ", line)
    pass

set_trace()

for each_id in data:
    idd = each_id['ID']
    card = len(each_id['text'])
    print(f"Number of IDS: {card}")

# with open(output_path, "a") as f:
#     json.dump(data, f)