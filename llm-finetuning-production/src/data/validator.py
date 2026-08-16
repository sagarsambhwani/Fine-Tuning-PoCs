"""
JSON Schema & Data Integrity Validator
Parses raw LLM text outputs, strips markdown codeblocks, and validates schema conformance.
"""
import re
import json
from typing import Dict, Any, Tuple, Optional

REQUIRED_SCHEMA_KEYS = {"customer", "quantity", "product", "amount", "delivery_day"}

def extract_json_substring(raw_text: str) -> Optional[str]:
    """Extracts JSON substring from LLM generation (handling ```json fences or raw brackets)."""
    if not raw_text:
        return None

    # Try extracting inside markdown ```json ... ``` fences
    markdown_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if markdown_match:
        return markdown_match.group(1).strip()

    # Try extracting outermost curly braces
    brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw_text, re.DOTALL)
    if brace_match:
        return brace_match.group(0).strip()

    return raw_text.strip()

def parse_and_validate_extraction(raw_text: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Parses LLM output into JSON and validates against schema.
    Returns: (is_valid, parsed_dict, error_message)
    """
    cleaned_text = extract_json_substring(raw_text)
    if not cleaned_text:
        return False, None, "Empty text or no JSON structure found"

    try:
        data = json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        return False, None, f"JSONDecodeError: {str(e)}"

    if not isinstance(data, dict):
        return False, None, f"Parsed JSON is not an object (type: {type(data).__name__})"

    # Check required fields
    missing_keys = REQUIRED_SCHEMA_KEYS - set(data.keys())
    if missing_keys:
        return False, data, f"Missing required schema keys: {missing_keys}"

    # Type validation & casting
    try:
        customer = str(data["customer"]).strip()
        quantity = int(data["quantity"])
        product = str(data["product"]).strip()
        amount = float(data["amount"])
        delivery_day = str(data["delivery_day"]).strip()
    except (ValueError, TypeError) as e:
        return False, data, f"Type validation error: {str(e)}"

    validated_dict = {
        "customer": customer,
        "quantity": quantity,
        "product": product,
        "amount": amount,
        "delivery_day": delivery_day
    }
    return True, validated_dict, None

def validate_json_schema(data: Dict[str, Any]) -> bool:
    """Returns True if dictionary strictly contains all required fields with appropriate types."""
    if not isinstance(data, dict):
        return False
    if not REQUIRED_SCHEMA_KEYS.issubset(data.keys()):
        return False
    return (
        isinstance(data.get("customer"), str)
        and isinstance(data.get("quantity"), (int, float))
        and isinstance(data.get("product"), str)
        and isinstance(data.get("amount"), (int, float))
        and isinstance(data.get("delivery_day"), str)
    )
