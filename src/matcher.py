"""
Match selection logic.

Scores candidate pairs and selects best one-to-one matches.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.scoring import compute_score


def match_transactions(
    bank,
    register,
    bank_embeddings,
    register_embeddings,
    candidates,
    bank_index_map,
    register_index_map
):
    """
    Score candidate pairs and select the best matches.
    
    Each bank transaction matches with at most 1 register transaction (one-to-one).
    Uses greedy selection: pick highest score, then next highest, etc.
    
    Parameters:
    -----------
    bank : DataFrame - bank transactions
    register : DataFrame - register transactions
    bank_embeddings : list of vectors - text embeddings for bank descriptions
    register_embeddings : list of vectors - text embeddings for register descriptions
    candidates : list of (bank_idx, register_idx) - pairs to consider
    bank_index_map : dict - maps dataframe index to embedding position
    register_index_map : dict - maps dataframe index to embedding position
    
    Returns:
    --------
    DataFrame with columns: bank_id, register_id, confidence
    """
    
    # Step 1: Score all candidate pairs
    scored_pairs = []
    
    for bank_idx, register_idx in candidates:
        
        # Get embedding positions
        bank_pos = bank_index_map[bank_idx]
        register_pos = register_index_map[register_idx]
        
        # Compute text similarity using AI embeddings
        text_sim = cosine_similarity(
            [bank_embeddings[bank_pos]],
            [register_embeddings[register_pos]]
        )[0][0]
        
        # Compute overall match score using multiple factors
        score = compute_score(
            bank.loc[bank_idx],
            register.loc[register_idx],
            text_sim
        )
        
        scored_pairs.append({
            "bank_id": bank_idx,
            "register_id": register_idx,
            "confidence": score
        })
    
    # Convert to dataframe and sort by score (highest first)
    scored_pairs = pd.DataFrame(scored_pairs)
    scored_pairs = scored_pairs.sort_values("confidence", ascending=False)
    
    # Step 2: Greedily select matches (one-to-one constraint)
    matches = []
    used_bank = set()
    used_register = set()
    
    for _, row in scored_pairs.iterrows():
        bank_id = row["bank_id"]
        register_id = row["register_id"]
        
        # Skip if either is already matched
        if bank_id in used_bank or register_id in used_register:
            continue
        
        # Accept this match
        matches.append(row)
        used_bank.add(bank_id)
        used_register.add(register_id)
    
    return pd.DataFrame(matches)