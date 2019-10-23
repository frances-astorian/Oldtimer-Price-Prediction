import os
import pickle
import datetime
import pandas as pd


def engineer_basic_features(df, live_auction=False):
    """
    Creates features for machine learning.

    Args:
        df: A pandas dataframe
        live_auction: (bool)
    Returns
        A Pandas dataframe.
    """
    # Creates Year Sold, Month Sold, and Decade Built columns
    df["Decade Built"] = df["Year"].apply(lambda x: (x - x % 10))
    if live_auction == False:
        df["Year Sold"] = df["Date"].apply(lambda x: x.year)
    else:
        df["Year Sold"] = datetime.datetime.now().year

    if not os.exists('categorical_dtypes'):
        os.mkdir('categorical_dtypes')

    if live_auction == False:
        model_dtype = pd.api.types.CategoricalDtype(categories=df["Model"].unique(), ordered=True)
        year_sold_dtype = pd.api.types.CategoricalDtype(categories=df["Year Sold"].unique(), ordered=True)
        decade_built_dtype = pd.api.types.CategoricalDtype(categories=df["Decade Built"].unique(), ordered=True)

        pickle.dump(model_dtype, open('./categorical_dtypes/model.pickle', 'wb'))
        pickle.dump(year_sold_dtype, open('./categorical_dtypes/year_sold.pickle', 'wb'))
        pickle.dump(decade_built_dtype, open('./categorical_dtypes/decade_built.pickle', 'wb'))

    else:
        model_dtype = pickle.load(open('./categorical_dtypes/model.pickle', 'rb'))
        year_sold_dtype = pickle.load(open('./categorical_dtypes/year_sold.pickle', 'rb'))
        decade_built_dtype = pickle.load(open('./categorical_dtypes/decade_built.pickle', 'rb'))

    df['Model'] = df['Model'].astype(model_dtype)
    df['Year Sold'] = df['Year Sold'].astype(year_sold_dtype)
    df['Decade Built'] = df['Decade Built'].astype(decade_built_dtype)

    make = pd.get_dummies(df["Model"], drop_first=True)
    year_sold = pd.get_dummies(df["Year Sold"], drop_first=True)
    decade_built = pd.get_dummies(df["Decade Built"], drop_first=True)

    df["Intercept"] = 1
    if live_auction == False:
        df = pd.concat((df[["Kilometers", "Engine", "Price", "Intercept", "Gearbox", "Year"]], make,
                        decade_built, year_sold), axis=1)
    else:
        df = pd.concat((df[["Kilometers", "Engine", "Intercept", "Gearbox", "Year"]], make,
                        decade_built, year_sold), axis=1)

    return df


def engineer_engine_features(df):
    """
    Converts the Engine column into categorical features using regex.

    Args:
        df: A Pandas Dataframe.
    Returns:
        A Pandas DataFrame.
    """
    # create column for engines that were rebuilt.
    df["Engine Rebuilt"] = df["Engine"].str.contains("Rebuilt").astype(int)

    # create columns for turbocharged, and twin turbocharged engines.
    df["Twin Turbocharged"] = df["Engine"].str.contains(r"Twin-Turbocharged | Twin-Turbo |Twin-turbocharged").astype(
        int)
    df["Engine Turbocharged"] = df["Engine"].str.contains(r"Turbocharged | Turbo").astype(int)

    # create column for engines that are still original.
    df["Original Engine"] = df["Engine"].str.contains(r"Original|Numbers-Matching|Numbers Matching").astype(int)

    # create columns for engine displacement
    for x in range(1, 5):
        for x2 in range(0, 10):
            engine_displacement = str(x) + "." + str(x2) + "L"
            rows_with_engine_displacement = df["Engine"].str.contains(engine_displacement)
            if rows_with_engine_displacement.any():
                df[engine_displacement] = rows_with_engine_displacement.astype(int)

    # Create columns depending on Engine type
    df["Flat Six"] = df["Engine"].str.contains(r"Flat-Six|Flat Six").astype(int)
    df["Flat Four"] = df["Engine"].str.contains("Flat-Four").astype(int)
    df["Inline Four"] = df["Engine"].str.contains("Inline-Four").astype(int)
    df["V8"] = df["Engine"].str.contains("V8").astype(int)

    df = df.drop("Engine", axis=1)

    return df


def engineer_gearbox_features(df):
    """
    Converts the Gearbox Column into categorical features using regex.

    Args:
        df: A Pandas DataFrame
    Returns:
        A Pandas Dataframe.
    """

    df["Manual"] = df["Gearbox"].str.contains(r"Manual|manual").astype(int)
    df["ZF Gearbox"] = df["Gearbox"].str.contains(r"ZF").astype(int)
    df["Sequential Gearbox"] = df["Gearbox"].str.contains(r"Sequential").astype(int)
    df["4-Speed"] = df["Gearbox"].str.contains(r"4-Speed").astype(int)
    df["5-Speed"] = df["Gearbox"].str.contains(r"5-Speed").astype(int)
    df["6-Speed"] = df["Gearbox"].str.contains(r"6-Speed").astype(int)

    return df
