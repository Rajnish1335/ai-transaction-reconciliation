"""
Data loading utilities.

Reads CSV files and prepares data for matching.
"""

import pandas as pd


def load_datasets(bank_path, register_path):
    """
    Load bank and register CSV files.
    
    Converts date columns to proper datetime format.
    
    Parameters:
    -----------
    bank_path : str - path to bank CSV file
    register_path : str - path to register CSV file
    
    Returns:
    --------
    bank : DataFrame - bank transactions
    register : DataFrame - register transactions
    """
    # Load CSV files
    bank = pd.read_csv(bank_path)
    register = pd.read_csv(register_path)
    
    # Convert date columns to datetime for proper date math
    bank["date"] = pd.to_datetime(bank["date"])
    register["date"] = pd.to_datetime(register["date"])
    
    return bank, register