import pandas as pd

def cleanse_data(df, live_auction=False):
    """Conducts Elementary Data Cleansing

    Args
        df: A pandas DataFrame of Webscraped Auction Results from BringATrailer
        live_auction: (bool) Indicates whether the Data is from Live Auctions of
    Returns
        A pandas DataFrame.
    """

    # Removes rows with missing values
    if live_auction == True:
        df = df.drop('Date', axis=1)
        df = df.drop('Price', axis=1)
    else:
        df["Date"] = pd.to_datetime(df["Date"])

    df = df.dropna()
    df["Running Condition"] = df["Running Condition"].astype(bool)
    df["Year"] = df["Year"].astype(int)
    df["Kilometers"] = df["Kilometers"].astype(int)

    # Removes cars that are not in running condition.
    df = df.loc[df["Running Condition"] == True]

    return df


def remove_outliers(df):
    """
    Removes outliers from the data.

    Args:
        df: A pandas Dataframe
    Returns
        A pandas Dataframe.
    """

    # Remove Oldtimers that cost more than $250,000
    df = df.loc[df["Price"] < 250000]

    # Remove Oldtimers that have driven more than 450,000 kilometers.
    df = df.loc[df["Kilometers"] < 450000]

    df = df.loc[df['Year'] > 1928]

    return df
