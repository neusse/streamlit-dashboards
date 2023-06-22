import streamlit as st
import yfinance as yf
import time
from patterns import candlestick_patterns
import plotly.graph_objects as go
import os, csv
import pandas as pd
import numpy as np
import talib

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
        "symbol vs all patterns",
        "pattern vs all symbols",
    ),
    1,
)


###############################################################################
#
###############################################################################
st.header(option)


###############################################################################
#
###############################################################################
if option == "pattern vs all symbols":
    cs_lookup = {}
    for cs in candlestick_patterns.keys():
        cs_lookup[candlestick_patterns[cs]] = cs

    pattern = st.sidebar.selectbox("Select", cs_lookup)

    st.write(pattern)
    # st.write(cs_lookup[pattern])

    stocks = {}

    with open("datasets/symbols.csv") as f:
        for row in csv.reader(f):
            stocks[row[0]] = row[1]

    if pattern:
        bullish = 0
        bearish = 0

        for filename in os.listdir("datasets/daily"):
            df = pd.read_csv("datasets/daily/{}".format(filename))
            pattern_function = getattr(talib, cs_lookup[pattern])
            symbol = filename.split(".")[0]

            try:
                results = pattern_function(
                    df["Open"], df["High"], df["Low"], df["Close"]
                )
                last = results.tail(1).values[0]
                date = df["Date"].iloc[-1]

                if last > 0:
                    bullish += 1
                    st.subheader(f"{stocks[symbol]} - Bullish")
                    st.write(date)
                    st.image(
                        f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=1&p=d&s=l"
                    )
                    st.write("-----------------------------------------------------")
                elif last < 0:
                    stocks[symbol][pattern] = "bearish"
                    bearish += 1
                    st.write(date)
                    st.image(
                        f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=1&p=d&s=l"
                    )
                    st.write("-----------------------------------------------------")
                else:
                    pass
                    # stocks[symbol][pattern] = None
            except Exception as e:
                pass
                # print(f'failed on filename: {filename} - {e}')

        st.sidebar.write(f" {bullish} - Bullish  \t {bearish} - Bearish")

    st.sidebar.write("--DONE--")
    st.write("--DONE--")


###############################################################################
#
###############################################################################
if option == "symbol vs all patterns":
    symbol = st.sidebar.text_input(
        "Symbol", value="MSFT", max_chars=None, key=None, type="default"
    )

    symbol = symbol.upper()

    cs_lookup = {}
    for cs in candlestick_patterns.keys():
        cs_lookup[candlestick_patterns[cs]] = cs

    stocks = {}

    with open("datasets/symbols.csv") as f:
        for row in csv.reader(f):
            #            stocks[row[0]] = {"company": row[1]}
            stocks[row[0]] = row[1]
            # st.write(stocks[row[0]])

    st.write(f" {symbol} - {stocks[symbol]}")

    st.image(f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=1&p=d&s=l")
    st.write("-----------------------------------------------------")

    bullish = 0
    bearish = 0

    for pattern in cs_lookup:
        pattern_function = getattr(talib, cs_lookup[pattern])
        df = pd.read_csv(f"datasets/daily/{symbol}.csv")

        try:
            results = pattern_function(df["Open"], df["High"], df["Low"], df["Close"])
            last = results.tail(1).values[0]
            date = df["Date"].iloc[-1]

            if last > 0:
                bullish += 1
                st.subheader(f"{pattern} - {date} - Bullish")
                st.write("-----------------------------------------------------")
            elif last < 0:
                bearish += 1
                st.subheader(f"{pattern} - {date} - Bearish")
                st.write("-----------------------------------------------------")
            # else:
            #     pass
            # stocks[symbol][pattern] = None
        except Exception as e:
            pass
            # print(f'failed on filename: {filename} - {e}')

        # st.dataframe(stocks)
    st.sidebar.write(f" {bullish} - Bullish  \t {bearish} - Bearish")
    st.sidebar.write("--DONE--")
    st.write("--DONE--")
