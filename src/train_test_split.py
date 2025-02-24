import pandas as pd
from sklearn.model_selection import train_test_split
def stratified_train_test_split(
    df : pd.DataFrame, 
    test_size : float,
    num_bins : int = 10,
    random_state : int = 42
):
    num_bins = 10  # Adjust based on your distribution
    df['num_tokens_bin'] = pd.qcut(df['num_tokens'], q=num_bins, labels=False, duplicates='drop')

    # Step 2: Perform stratified train-test split
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['num_tokens_bin'], random_state=random_state)

    # Step 3: Drop the bin column (optional)
    train_df = train_df.drop(columns=['num_tokens_bin'])
    test_df = test_df.drop(columns=['num_tokens_bin'])

    return train_df, test_df