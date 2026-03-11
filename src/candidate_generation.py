"""
Candidate pair generation.

For each bank transaction, find similar register transactions to score later.
Use amount proximity and date window to reduce search space.
"""

import numpy as np


def generate_candidates(bank, register, k=15):
    """
    Generate candidate transaction pairs for matching.
    
    Strategy: For each bank transaction, find the 15 register transactions
    with closest amounts, then filter to those within 7 days.
    
    This ensures correct matches are candidates while keeping the list small.
    
    Parameters:
    -----------
    bank : DataFrame - bank transactions
    register : DataFrame - register transactions
    k : int - how many closest amounts to consider (default 15)
    
    Returns:
    --------
    list of (bank_index, register_index) tuples
    """
    candidates = []
    
    # Pre-compute all register amounts for fast comparison
    reg_amounts = register["amount"].values
    
    # For each bank transaction
    for b_idx, bank_row in bank.iterrows():
        
        # Find register transactions with closest amounts
        amount_differences = np.abs(reg_amounts - bank_row["amount"])
        k_closest_positions = np.argsort(amount_differences)[:k]
        
        # Check each of the k closest
        for position in k_closest_positions:
            reg_idx = register.index[position]
            reg_row = register.iloc[position]
            
            # Only include if within 7-day date window
            days_apart = abs((bank_row["date"] - reg_row["date"]).days)
            if days_apart <= 7:
                candidates.append((b_idx, reg_idx))
    
    return candidates