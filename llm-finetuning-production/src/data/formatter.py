"""
Chat Template & Dataset Formatting Utilities
Prepares raw dataset records into tokenizer-compatible chat sequences.
"""
from typing import List, Dict, Any, Optional

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert entity extraction system. Extract structured order information "
    "from the user's text into valid JSON according to the schema: "
    '{"customer": string, "quantity": integer, "product": string, "amount": float, "delivery_day": string}.'
)

def format_chat_prompt(
    text_input: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> List[Dict[str, str]]:
    """Constructs a standard OpenAI-style conversational message sequence."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text_input}
    ]

def apply_template_to_messages(
    messages: List[Dict[str, str]],
    tokenizer: Any,
    add_generation_prompt: bool = True
) -> str:
    """Applies the model's tokenizer chat template if available, else formats standard markup."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
    
    # Fallback to standard chat markup if tokenizer lacks template
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    if add_generation_prompt:
        formatted += "<|im_start|>assistant\n"
    return formatted

def create_conversational_dataset(raw_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validates and extracts 'messages' list from dataset entries for TRL SFTTrainer."""
    formatted_data = []
    for item in raw_dataset:
        if "messages" in item:
            formatted_data.append({"messages": item["messages"]})
        else:
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": item.get("input", "")},
                {"role": "assistant", "content": str(item.get("output", ""))}
            ]
            formatted_data.append({"messages": messages})
    return formatted_data
