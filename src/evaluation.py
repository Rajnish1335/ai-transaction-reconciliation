"""
Evaluation metrics.

Calculates precision, recall, and F1 score for match predictions.
"""


def evaluate(matches, ground_truth):
    """
    Compare predictions to ground truth.
    
    Parameters:
    -----------
    matches : DataFrame - predicted matches (columns: bank_id, register_id, confidence)
    ground_truth : DataFrame - correct matches (columns: bank_id, register_id)
    
    Returns:
    --------
    precision : float - % of predictions that were correct
    recall : float - % of ground truth we found
    f1 : float - harmonic mean of precision and recall
    """
    # Convert to sets of ID pairs
    predicted_pairs = set(zip(matches.bank_id, matches.register_id))
    correct_pairs = set(zip(ground_truth.bank_id, ground_truth.register_id))
    
    # Count correct matches
    correct = len(predicted_pairs & correct_pairs)  # Intersection
    
    # Calculate metrics
    precision = correct / len(predicted_pairs) if predicted_pairs else 0
    recall = correct / len(correct_pairs) if correct_pairs else 0
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1