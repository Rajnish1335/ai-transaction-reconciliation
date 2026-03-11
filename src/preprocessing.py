"""
Data cleaning functions.

Makes text consistent and transaction types standardized.
"""

import re


def clean_description(text):
    """
    Clean transaction descriptions.
    
    - Convert to lowercase
    - Remove special characters
    - Remove extra whitespace
    
    Example: "TRADER JOES #123" → "trader joes 123"
    """
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)  # Remove special chars
    text = re.sub(r"\s+", " ", text)       # Remove extra spaces
    return text.strip()


def normalize_type(value):
    """
    Convert transaction types to standard format.
    
    Both "DEBIT" and "DR" → "DR"
    Both "CREDIT" and "CR" → "CR"
    """
    value = str(value).lower()
    
    if value in ["debit", "dr"]:
        return "DR"
    elif value in ["credit", "cr"]:
        return "CR"
    else:
        return value.upper()