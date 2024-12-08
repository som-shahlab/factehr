"""
FactEHR: Create Prompt Templates

We use Langchain to store and manage prompt templates, which supports adding 
metadata to each template. We define the following prompt categories:

- Fact Decomposition Prompt Templates (Zero-shot and 2-shot ICL)
- Fact Decomposition LLM-as-a-Judge / Model Evaluator Prompt Templates

TODO Get `langchain` serialization of PromptTemplate objects working correctly.


python scripts/build_prompt_templates.py \
--path_to_output data/prompt_templates

"""

import os
import argparse
from pathlib import Path
from langchain_core.prompts import PromptTemplate

########## Argparse Setup ##########

parser = argparse.ArgumentParser(description="Generate prompt templates")
parser.add_argument(
    "-o",
    "--path_to_output",
    type=str,
    help="Path to the output directory",
    default="data/prompt_templates/",
)

########## Fact Generation Prompt Templates ##########

prompt1 = """Please breakdown the following text into independent facts as a numbered list (Do not have any other text, or say "Here is the list..." ):
{text}"""

prompt2 = """You are a meticulous physician who is reading this medical note from the electronic health record. 
Your goal is to: 
Write out as a numbered list (and no sublists) of key pieces of clinical information from the note as separate, independent facts. 
At each step, ensure to separate out information with multiple modifiers including concepts like laterality, size, into simpler, more granular facts.

(Do not have any other text, or say "Here is the list..." )
Note: 
{text}"""

prompt1_icl = """Please breakdown the following text into independent facts as a numbered list (and no sublists):

Example 1 : 
Note: "There is a dense consolidation in the left lower lobe."

Atomic facts:
1. There is a consolidation.
2. The consolidation is dense.
3. The consolidation is on the left.
4. The consolidation is in a lobe.
5. The consolidation is in the lower portion of the left lobe.


Example 2: 

Note: "The patient has been having intermittent shortness of breath for the last two years."

Atomic facts:
1. The patient has been having shortness of breath.
2. The shortness of breath is intermittent.
3. The shortness of breath has been present for the last two years.

(Do not have any other text, or say "Here is the list..." )

Note: 
{text}"""

prompt2_icl = """You are a meticulous physician who is reading this medical note from the electronic health record. 
Your goal is to: 
Write out as a numbered list (and no sublists) of key pieces of clinical information from the note as separate, independent facts. 
At each step, ensure to separate out information with multiple modifiers including concepts like laterality, size, into simpler, more granular facts.

Example 1: 
    
Note: "There is a dense consolidation in the left lower lobe."

Atomic facts:
1. There is a consolidation.
2. The consolidation is dense.
3. The consolidation is on the left.
4. The consolidation is in a lobe.
5. The consolidation is in the lower portion of the left lobe.


Example 2: 

Note: "The patient has been having intermittent shortness of breath for the last two years."

Atomic facts:
1. The patient has been having shortness of breath.
2. The shortness of breath is intermittent.
3. The shortness of breath has been present for the last two years.

(Do not have any other text, or say "Here is the list..." )

Note: 
{text}"""

########## Fact Entailmen Prompt Templates ##########

fact_entailment_prompt_tmpl = """You are an expert on natural language entailment. Your task is to deduce whether premise statements entail hypotheses. Only return a '1' if the hypothesis can be fully entailed by the premise. Return '0' if the hypothesis contains information that cannot be entailed by the premise. Also generate an explanation for your answer. Generate the answer in JSON format with the following keys: 'explanation': the reason why the entailment prediction is made, 'entailment_prediction': 1 or 0, whether the claim can be entailed. Only return the JSON-formatted answer and nothing else.
Premise: {premise}\nHypothesis: {hypothesis}"""


def main():

    args = parser.parse_args()

    output_file_path = Path(args.path_to_output) / "fact_decomposition"
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    for name, tmpl in zip(
        ["prompt1", "prompt2", "prompt1_icl", "prompt2_icl"],
        [prompt1, prompt2, prompt1_icl, prompt2_icl],
    ):
        with open(output_file_path / f"{name}.tmpl", "w") as f:
            f.write(tmpl)

        # I cannot get serialization to work correctly here. Dumb langchain
        # prompt_template = PromptTemplate.from_template(
        #     template=tmpl,
        #     name=name,
        #     ags=["fact_decomposition"],
        # )
        # prompt_template.save(file_path=f"{args.path_to_output}/{name}.yaml")

    output_file_path = Path(args.path_to_output) / "fact_entailment"
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    with open(output_file_path / "fact_entailment.tmpl", "w") as f:
        f.write(fact_entailment_prompt_tmpl)


if __name__ == "__main__":
    main()
