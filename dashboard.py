import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ta
import yfinance as yf
from datetime import datetime
import json
from scipy.stats import zscore

# import pygwalker as pyg


# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon=":ninja:",
    layout="wide",
    menu_items={"About": "# This is an *extremely* cool app!"},
)


option = st.sidebar.selectbox(
    "Which Dashboard?",
    (
        "chart",
        "Options",
        "EDA for financial datasets",
        "Get Ticker Info",
        "list all stocks",
        "Fundamental Score",
    ),
    5,
)


###############################################################################
#
###############################################################################
@st.cache_data
def load_data():
    components = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S" "%26P_500_companies"
    )[0]
    #    return components.drop('SEC filings', axis=1).set_index('Symbol')
    return components.set_index("Symbol")


###############################################################################
#
###############################################################################
@st.cache_data
def load_quotes(asset):
    return yf.download(asset)


###############################################################################
#
###############################################################################
st.header(option)


###############################################################################
#
###############################################################################
if option == "chart":
    symbol = st.sidebar.text_input(
        "Symbol", value="TSLA", max_chars=None, key=None, type="default"
    )

    st.subheader(symbol.upper())
    st.image(f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=1&p=d&s=l")
    # Read the CSV file
    data = pd.read_csv(f"datasets/daily/{symbol}.csv")

    # Convert the 'Date' column to datetime
    data["Date"] = pd.to_datetime(data["Date"])

    # Get the date 10 months ago from today
    six_months_ago = datetime.now() - pd.DateOffset(months=6)

    # Filter the dataframe to include only the last 6 months
    data = data[data["Date"] >= six_months_ago]

    # Continue with your plotting
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data["Date"],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name=symbol,
            )
        ]
    )

    fig.update_xaxes(type="category")
    fig.update_layout(height=800)

    st.plotly_chart(fig, use_container_width=True)

    st.write(data)

    # pyg.walk(data, env="Streamlit", dark="dark")


###############################################################################
#
###############################################################################
if option == "list all stocks":
    components = load_data()
    st.dataframe(
        components[
            [
                "Security",
                "GICS Sector",
                "GICS Sub-Industry",
                "Headquarters Location",
                "Date added",
                "CIK",
                "Founded",
            ]
        ]
    )


###############################################################################
#
###############################################################################
def more_quotes():
    components = load_data()
    title = st.empty()
    st.sidebar.title("Options")

    def label(symbol):
        a = components.loc[symbol]
        return symbol + " - " + a.Security

    if st.sidebar.checkbox("View companies list"):
        st.dataframe(
            components[
                [
                    "Security",
                    "GICS Sector",
                    "GICS Sub-Industry",
                    "Headquarters Location",
                    "Date added",
                    "CIK",
                    "Founded",
                ]
            ]
        )

    st.sidebar.subheader("Select asset")
    asset = st.sidebar.selectbox(
        "Click below to select a new asset",
        components.index.sort_values(),
        index=3,
        format_func=label,
    )
    title.title(components.loc[asset].Security)
    if st.sidebar.checkbox("View company info", True):
        st.table(components.loc[asset])
    data0 = load_quotes(asset)
    data = data0.copy().dropna()
    data.index.name = None

    section = st.sidebar.slider(
        "Number of quotes",
        min_value=30,
        max_value=min([2000, data.shape[0]]),
        value=500,
        step=10,
    )

    data2 = data[-section:]["Adj Close"].to_frame("Adj Close")

    sma = st.sidebar.checkbox("SMA")
    if sma:
        period = st.sidebar.slider(
            "SMA period", min_value=5, max_value=500, value=20, step=1
        )
        data[f"SMA {period}"] = data["Adj Close"].rolling(period).mean()
        data2[f"SMA {period}"] = data[f"SMA {period}"].reindex(data2.index)

    sma2 = st.sidebar.checkbox("SMA2")
    if sma2:
        period2 = st.sidebar.slider(
            "SMA2 period", min_value=5, max_value=500, value=100, step=1
        )
        data[f"SMA2 {period2}"] = data["Adj Close"].rolling(period2).mean()
        data2[f"SMA2 {period2}"] = data[f"SMA2 {period2}"].reindex(data2.index)

    st.subheader("Chart")
    st.line_chart(data2)

    if st.sidebar.checkbox("View stadistic"):
        st.subheader("Stadistic")
        st.table(data2.describe())

    if st.sidebar.checkbox("View quotes"):
        st.subheader(f"{asset} historical data")
        st.write(data2)


###############################################################################
#
###############################################################################
if option == "Options":
    more_quotes()


###############################################################################
# Data preparation
###############################################################################
@st.cache_data()
def EDA_load_data(symbol):
    df = pd.read_csv(f"datasets/daily/{symbol}.csv")

    # Convert the 'Date' column to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Clean NaN values
    df = ta.utils.dropna(df)

    # Apply feature engineering (technical analysis)
    # df = ta.add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume", fillna=True)
    df = ta.add_volatility_ta(
        df, "High", "Low", "Close", fillna=False, colprefix=ta_col_prefix
    )
    df = ta.add_momentum_ta(
        df,
        "High",
        "Low",
        "Close",
        "Volume",
        fillna=False,
        colprefix=ta_col_prefix,
    )

    return df


###############################################################################
#
###############################################################################
if option == "EDA for financial datasets":
    # Settings
    width_px = 1000
    ta_col_prefix = "ta_"

    # Need to add ticker select here

    symbol = st.sidebar.text_input(
        "Symbol", value="TSLA", max_chars=None, key=None, type="default"
    )

    symbol = symbol.upper()

    df = EDA_load_data(symbol)

    # Sidebar
    return_value = st.sidebar.selectbox(
        "How many periods to calculate the return price?", [1, 2, 3, 5, 7, 14, 31]
    )

    # Prepare target: X Periods Return
    df["y"] = (df["Close"] / df["Close"].shift(return_value) - 1) * 100

    # Clean NaN values
    df = df.dropna()

    # Body
    st.title(f"EDA for financial dataset ({symbol})")

    # dataset fields names   --- Date,Open,High,Low,Close,Adj Close,Volume
    a = df["Date"].iloc[0]
    b = df["Date"].iloc[-1]

    # a = datetime.utcfromtimestamp(df["Timestamp"].iloc[0]).strftime("%Y-%m-%d %H:%M:%S")
    # b = datetime.utcfromtimestamp(df["Timestamp"].iloc[-1]).strftime("%Y-%m-%d %H:%M:%S")
    st.write(
        f"We try to explore a small financial time series dataset (2000 rows) with BTC/USD prices from {a} to {b}"
    )
    st.write(
        "We start with a financial dataset, and we get some technical analysis features from the original "
        "dataset using [ta package](https://github.com/bukosabino/ta). Then, we define the target value as the X "
        "period return value (the user can set it). Finally, we explore these features and the target column "
        "graphically."
    )

    st.subheader(f"Dataframe ({symbol})")
    st.write(df)

    st.subheader(f"Describe dataframe ({symbol})")
    st.write(df.describe())

    st.write("Number of rows: {}, Number of columns: {}".format(*df.shape))

    st.subheader(f"Price ({symbol})")
    st.line_chart(df["Close"], width=width_px)

    st.subheader(f"Return {return_value} periods ({symbol})")
    st.area_chart(df["y"], width=width_px)

    st.subheader(f"Histogram ({symbol})")
    bins = list(np.arange(-10, 10, 0.5))
    hist_values, hist_indexes = np.histogram(df["y"], bins=bins)
    st.bar_chart(
        pd.DataFrame(data=hist_values, index=hist_indexes[0:-1]), width=width_px
    )
    st.write(
        "Target value min: {0:.2f}%; max: {1:.2f}%; mean: {2:.2f}%; std: {3:.2f}".format(
            np.min(df["y"]), np.max(df["y"]), np.mean(df["y"]), np.std(df["y"])
        )
    )

    # Univariate Analysis
    st.subheader(f"Correlation coefficient TA features  ({symbol})")

    x_cols = [
        col
        for col in df.columns
        if col not in ["Date", "y"] and col.startswith(ta_col_prefix)
        # if col not in ["Timestamp", "y"] and col.startswith(ta_col_prefix)
    ]
    labels = [col for col in x_cols]
    values = [np.corrcoef(df[col], df["y"])[0, 1] for col in x_cols]

    st.bar_chart(data=pd.DataFrame(data=values, index=labels), width=width_px)


fundamentals_tables = {
    "col1_a": {
        "trailingPE",
        "forwardPE",
        "pegRatio",
        "priceToSalesTrailing12Months",
        "priceToBook",
        "trailingEps",
        "forwardEps",
        "quickRatio",
        "currentRatio",
    },
    "col1_b": [
        "auditRisk",
        "boardRisk",
        "compensationRisk",
        "shareHolderRightsRisk",
        "overallRisk",
    ],
    "col2": [
        "totalRevenue",
        "totalCash",
        "totalDebt",
        "52WeekChange",
        "SandP52WeekChange",
        "twoHundredDayAverage",
        "fiftyDayAverage",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
        "earningsQuarterlyGrowth",
    ],
    "col3": [
        "currentPrice",
        "previousClose",
        "targetHighPrice",
        "targetLowPrice",
        "targetMeanPrice",
        "targetMedianPrice",
        "recommendationMean",
        "recommendationKey",
        "numberOfAnalystOpinions",
    ],
    "col4": [
        "floatShares",
        "sharesOutstanding",
        "sharesShort",
        "sharesShortPriorMonth",
        "sharesShortPreviousMonthDate",
        "dateShortInterest",
        "shortRatio",
        "shortPercentOfFloat",
        "beta",
        "heldPercentInsiders",
        "heldPercentInstitutions",
    ],
    "header": [
        "longName",
        # "symbol",
        "address1",
        "city",
        "state",
        # "zip",
        "website",
        "industry",
        "sector",
        # "messageBoardId",
        "quoteType",
        "exchange",
    ],
    "revenue": [
        "totalRevenue",
        "debtToEquity",
        "revenuePerShare",
        "returnOnAssets",
        "returnOnEquity",
        "grossProfits",
        "freeCashflow",
        "operatingCashflow",
        "revenueGrowth",
        "grossMargins",
        "ebitdaMargins",
        "operatingMargins",
        "financialCurrency",
        "trailingPegRatio",
        "earningsQuarterlyGrowth",
    ],
}


###############################################################################
#
###############################################################################
def get_col_df(table, stats, head_1, head_2):
    # all fields are not always avalable.  add them in missing and set to NaN
    for key in table:
        stats.setdefault(key, "NaN")

    # create a subset dict for each table to display from stats dict
    mylist = {}
    for field in table:
        mylist[field] = stats[field]

    # now create a dataframe so we can use the DF display for a nice display format
    df = pd.DataFrame(list(mylist.items()), columns=[head_1, head_2])

    # send it home
    return df


###############################################################################
#
###############################################################################
def score_stocks(stocks, weights):
    # Standardize each ratio using z-score
    for key in weights.keys():
        values = [stock[key] for stock in stocks]
        print(values)

        # Handling edge cases
        if len(set(values)) == 1:
            print(f"All values for {key} are the same.")
            standardized_values = [
                0 for _ in values
            ]  # all values are the same, z-scores will be 0
        elif any(np.isnan(val) for val in values):
            print(f"There are NaN values in data for {key}.")
            # Decide what to do with NaN values. Here I'm assuming replacing them with 0
            standardized_values = [
                0 if np.isnan(val) else val for val in zscore(values)
            ]
        else:
            standardized_values = zscore(values)

        print(standardized_values)

        for i, stock in enumerate(stocks):
            stock[key] = standardized_values[i]

    # Calculate weighted score for each stock
    for stock in stocks:
        score = sum(stock[key] * weights[key] for key in weights.keys())

        # Normalize score to a 0-100 scale
        stock["score"] = 50 * (score + 1)

    # Sort stocks by score in descending order
    stocks.sort(key=lambda x: x["score"], reverse=True)

    return stocks


###############################################################################
#
###############################################################################
def score_single_stock(stock, weights):
    # Calculate weighted score for the stock
    score = sum(stock[key] * weights[key] for key in weights.keys())

    # Normalize score to a 0-100 scale if necessary
    # This step assumes that the maximum possible score is known and equal to the sum of the weights
    max_score = sum(weights.values())
    normalized_score = 100 * (score / max_score)

    return normalized_score


###############################################################################
#
###############################################################################
if option == "Get Ticker Info":
    # placeholder for header.  its an st thing
    title = st.empty()

    # get the symbol to pull fundamentals
    symbol = st.sidebar.text_input(
        "Symbol", value="TSLA", max_chars=None, key=None, type="default"
    )
    # normalize it a little
    symbol = symbol.upper()
    # pull the fundamentals from yahoo
    myticker = yf.Ticker(symbol)

    # remove these from the data.  not really fundamentals and we need a flat dict for this to work
    stats = myticker.info
    if "companyOfficers" in stats:
        del stats["companyOfficers"]
    if "longBusinessSummary" in stats:
        summary = stats["longBusinessSummary"]
        del stats["longBusinessSummary"]

    title.title(stats["shortName"])
    df = get_col_df(fundamentals_tables["header"], stats, "Detail:", "Value")
    st.table({symbol: df.set_index("Detail:")["Value"].to_dict()})

    st.image(f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=1&p=d&s=l")

    # st.write(myticker.info)

    stats["priceToBook"] = stats["currentPrice"] / stats["bookValue"]

    st.header(f"Fundamentals ({symbol})")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        stats["priceToBook"] = stats["currentPrice"] / stats["bookValue"]

        df = get_col_df(fundamentals_tables["col1_a"], stats, "Ratios:", "Value")
        st.dataframe(df, hide_index=True)

        df = get_col_df(fundamentals_tables["col1_b"], stats, "Risk:", "Value")
        st.dataframe(df, hide_index=True)

    with col2:
        df = get_col_df(fundamentals_tables["col2"], stats, "Books:", "Value")
        st.dataframe(df, hide_index=True)

        df = get_col_df(fundamentals_tables["revenue"], stats, "Revenue:", "Value")
        st.dataframe(df, hide_index=True)

    with col3:
        df = get_col_df(fundamentals_tables["col3"], stats, "Price:", "Value")
        st.dataframe(df, hide_index=True)

    with col4:
        df = get_col_df(fundamentals_tables["col4"], stats, "Short:", "Value")
        st.dataframe(df, hide_index=True)

    # df = pd.DataFrame([stats])
    # df_t = df.T

    # Assume we have a list of dictionaries, where each dictionary contains financial ratios for a stock
    # stocks = [
    #     {'symbol': 'AAPL', 'P/E': 15.1, 'P/B': 5.6, 'Dividend Yield': 0.7, 'ROE': 0.4, 'D/E': 1.5, 'EPS': 3.1, 'Current Ratio': 1.3, 'PEG': 1.5},
    #     {'symbol': 'GOOG', 'P/E': 20.4, 'P/B': 4.3, 'Dividend Yield': 0, 'ROE': 0.2, 'D/E': 0.2, 'EPS': 2.0, 'Current Ratio': 3.6, 'PEG': 2.0},
    #     # Add more stocks...
    # ]
    # Define weights for each financial ratio
    # weights = {
    #     'P/E': 0.15, #trailingPE
    #     'P/B': 0.15, # priceToBook
    #     'Dividend Yield': 0.10, # trailingAnnualDividendYield
    #     'ROE': 0.20, # returnOnEquity
    #     'D/E': 0.10, # debtToEquity
    #     'EPS': 0.15, # trailingEps
    #     'Current Ratio': 0.10, # currentRatio
    #     'PEG': 0.15 # pegRatio
    # }

    # Define weights for each financial ratio
    weights = {
        "trailingPE": 0.15,  # trailingPE
        "priceToBook": 0.15,  # priceToBook
        "trailingAnnualDividendYield": 0.10,  # trailingAnnualDividendYield
        "returnOnEquity": 0.20,  # returnOnEquity
        "debtToEquity": 0.10,  # debtToEquity
        "trailingEps": 0.15,  # trailingEps
        "currentRatio": 0.10,  # currentRatio
        "pegRatio": 0.15,  # pegRatio
    }
    # beta???

    # Use the function
    scored_stocks = score_single_stock(stats, weights)

    print(scored_stocks)

    # Now 'scored_stocks' is a list of stock dictionaries, each with a 'score' key and sorted by score
    #for stock in scored_stocks:
    st.subheader(f"Fundamental score ({symbol})")
    st.write(f"Symbol: {symbol}, Score: {scored_stocks:.2f}")

    st.text_area(
        f"Business Summary ({symbol} - {stats['longName']})",
        summary,
        height=250,
        disabled=True,
    )

    if st.checkbox("Show/Hide raw fundamentals data"):
        st.table(stats)
