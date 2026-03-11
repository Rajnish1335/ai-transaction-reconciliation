"""
Feedback loop utilities.
"""

import pandas as pd


def load_feedback(path="outputs/feedback.csv"):

    try:
        return pd.read_csv(path)

    except FileNotFoundError:
        return pd.DataFrame(columns=["bank_id", "register_id", "label"])


def save_feedback(feedback, path="outputs/feedback.csv"):
    feedback.to_csv(path, index=False)