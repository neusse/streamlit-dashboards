import streamlit as st
import pandas as pd
import yfinance as yf


# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon=":ninja:",
    layout="wide",
    menu_items={"About": "# This is an *extremely* cool app!"},
)

st.header("Fundamentals")

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

st.text_area(
    f"Business Summary ({symbol} - {stats['longName']})",
    summary,
    height=250,
    disabled=True,
)

if st.checkbox("Show/Hide raw fundamentals data"):
    st.table(stats)
