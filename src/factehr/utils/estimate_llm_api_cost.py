# Imports
from typing import List
import tiktoken


def estimate_request_limits(
    prompts: List[str],
    user_model_name: str,
    tokens_per_minute: int = 80000,
    max_tokens: int = 16,
    max_cost_threshold: float = 100  # Added max_cost_threshold
):
    """
    Estimates the request limits and costs based on provided prompts and limits.
    Number of tokens is estimated as the number of characters in the prompt divided by 4.
    
    Cost Estimation: 
    https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
    https://cloud.google.com/vertex-ai/generative-ai/pricing

    Args:
        prompts (List[str]): List of prompt strings representing user queries.
        tokens_per_minute (int): Max number of tokens that can be processed per minute.
        max_tokens (int, optional): Max number of tokens per completion. Default is 16.
        max_cost_threshold (float, optional): Maximum allowable cost. If exceeded, a ValueError is raised.

    Returns:
        est_req_rate (float): Estimated rate of requests that can be processed per minute.

    Raises:
        ValueError: If the estimated total cost exceeds the `max_cost_threshold`.
    """
    
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Calculate token usage based on number of characters divided by 4 (rough estimation)
    lens = [len(encoding.encode(prompt)) for prompt in prompts]
    total_tokens = sum(lens)
    worst_case_output_tokens = len(prompts) * max_tokens
    
    # Rate estimations
    idealized_token_rate = total_tokens / tokens_per_minute
    est_completion_tokens = max_tokens + (sum(lens) / len(lens)) + (2 * (max(lens) - min(lens))) / 4  # Rough mean and std
    
    est_req_rate = tokens_per_minute / est_completion_tokens

    # Log details
    print("~" * 50)
    print(f"Total Prompts: {len(prompts)}")
    print(f"Total Input Tokens (estimated): {total_tokens}")
    print(f"Worst Case Output Tokens (max_tokens={max_tokens}): {worst_case_output_tokens}")
    print(f"Per completion tokens (estimated): {est_completion_tokens}")
    print(f"Idealized Wall Time (Total Tokens/TPM): {idealized_token_rate:.1f} minutes")
    print(f"Est. Request Rate: {est_req_rate}/per minute )")
    print("~" * 50)

    # Pricing for input and output tokens (per 1000 tokens)
    pricing_inputs = {"GPT-4": 0.03, 
                      "GPT4-32k": 0.06, 
                      "shc-gpt-4o": 0.005, 
                      "gemini-1.5-flash-002": 0.000075,
                      "medlm-medium": 0.000075,
                      "gemini-1.5-pro-002": 0.00125}
    pricing_outputs = {"GPT-4": 0.06, 
                       "GPT4-32k": 0.12, 
                       "shc-gpt-4o": 0.015, 
                       "gemini-1.5-flash-002": 0.0003,
                       "medlm-medium": 0.0003,
                       "gemini-1.5-pro-002": 0.005}
    
    total_input_costs = {}
    total_output_costs = {}
    total_costs = {}

    if user_model_name not in pricing_inputs:
        raise ValueError(
            f"Pricing estimates for {user_model_name} not found. "
        )
    
    # Input cost estimation
    for model_name in pricing_inputs:
        input_cost = total_tokens / 1000 * pricing_inputs[model_name]
        total_input_costs[model_name] = input_cost
        print(f"~Input Cost: {model_name:<15}${input_cost:.2f} USD")
    
    # Output cost estimation
    for model_name in pricing_outputs:
        output_cost = worst_case_output_tokens / 1000 * pricing_outputs[model_name]
        total_output_costs[model_name] = output_cost
        print(f"Worst-Case Output Cost: {model_name:<15}${output_cost:.2f} USD")
    
    # Total cost estimation and threshold check
    for model_name in pricing_inputs:
        total_cost = total_input_costs[model_name] + total_output_costs[model_name]
        total_costs[model_name] = total_cost
        print(f"Total Estimated Cost: {model_name:<15}${total_cost:.2f} USD")
        
    # Check if total cost exceeds the max cost threshold
    if max_cost_threshold is not None and int(total_costs[user_model_name]) > max_cost_threshold:
        raise ValueError(
            f"Estimated cost for {user_model_name} exceeds the max cost threshold of ${max_cost_threshold:.2f}. "
            f"Estimated cost: ${total_costs[user_model_name]:.2f} USD"
        )
    
    return est_req_rate
