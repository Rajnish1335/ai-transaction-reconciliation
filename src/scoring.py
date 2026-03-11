"""
Match scoring function.

Combines multiple signals to score how well two transactions match.
"""


def compute_score(bank_row, reg_row, text_similarity):
    """
    Score how well two transactions match (0-1 scale).
    
    Combines:
    - Text similarity (60%) - description match using AI embeddings
    - Amount similarity (25%) - how close the transaction amounts are
    - Date similarity (10%) - how close the dates are
    - Type match (5%) - whether both are DEBIT or both are CREDIT
    
    Parameters:
    -----------
    bank_row : dict - bank transaction (has: amount, date, type)
    reg_row : dict - register transaction (has: amount, date, type)
    text_similarity : float - AI model similarity score (0-1)
    
    Returns:
    --------
    score : float between 0 and 1
    """
    
    # Amount: deduct points for each $5 difference
    amount_diff = abs(bank_row["amount"] - reg_row["amount"])
    amount_score = max(0, 1 - amount_diff / 5)
    
    # Date: deduct points for each 7 days difference
    date_diff = abs((bank_row["date"] - reg_row["date"]).days)
    date_score = max(0, 1 - date_diff / 7)
    
    # Type: perfect match gets 1, mismatch gets 0
    type_score = 1 if bank_row["type"] == reg_row["type"] else 0
    
    # Weighted combination
    final_score = (
        0.60 * text_similarity +   # Most important
        0.25 * amount_score +      # Very important
        0.10 * date_score +        # Somewhat important
        0.05 * type_score          # Least important
    )
    
    return final_score