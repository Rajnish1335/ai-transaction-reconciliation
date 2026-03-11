"""
Simple matching rule.

Matches transactions when amount and type are identical at the same position.
This is fast and accurate for exact matches.
"""

import pandas as pd


def unique_amount_matching(bank, register):
    """
    Find transactions that match exactly by amount and type.
    
    For each position, check if bank[i] and register[i] have:
    - Same amount
    - Same transaction type (DR or CR)
    
    If yes, count it as a match.
    
    Parameters:
    -----------
    bank : DataFrame - bank transactions (must have: amount, type)
    register : DataFrame - register transactions (must have: amount, type)
    
    Returns:
    --------
    matches : DataFrame with columns: bank_id, register_id, confidence
    used_register : set of matched register IDs
    """
    matches = []
    used_register = set()
    
    # Only check up to the shorter dataset
    max_rows = min(len(bank), len(register))
    
    for i in range(max_rows):
        bank_amount = bank.loc[i, "amount"]
        register_amount = register.loc[i, "amount"]
        
        bank_type = bank.loc[i, "type"]
        register_type = register.loc[i, "type"]
        
        # If amount and type both match, it's a definite match
        if bank_amount == register_amount and bank_type == register_type:
            matches.append({
                "bank_id": i,
                "register_id": i,
                "confidence": 1.0  # 100% confidence for exact matches
            })
            used_register.add(i)
    
    return pd.DataFrame(matches), used_register