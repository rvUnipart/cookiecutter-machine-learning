import pandas as pd
import config

def load_raw_as_dataframe():
    """
    Load content from the specified dataset file as a Pandas DataFrame.

    :return: A Pandas DataFrame representing the content of the specified file.
    """

    return pd.read_csv(config.RAW_DATA_FILE)
