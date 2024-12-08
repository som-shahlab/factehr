from enum import Enum

_prompt1 = (
    """
    Please breakdown the following text into independent facts as a numbered list (Do not have any other text, or say "Here is the list..." ):
    """
)

_prompt2 = ( 
    """
    You are a meticulous physician who is reading this medical note from the electronic health record. 
    Your goal is to: 
    Write out as a numbered list (and no sublists) of key pieces of clinical information from the note as separate, independent facts. 
    At each step, ensure to separate out information with multiple modifiers including concepts like laterality, size, into simpler, more granular facts.

    (Do not have any other text, or say "Here is the list..." )
    Note: 
    """   
)

_prompt1_icl = (
    """
    Please breakdown the following text into independent facts as a numbered list (and no sublists):

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
    """
)

_prompt2_icl = (
    """
    You are a meticulous physician who is reading this medical note from the electronic health record. 
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
    """
)


class Prompts(Enum):
    PROMPT1 = _prompt1
    PROMPT1_ICL = _prompt1_icl
    PROMPT2 = _prompt2
    PROMPT2_ICL = _prompt2_icl
