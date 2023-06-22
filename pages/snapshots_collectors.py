import streamlit as st
import yfinance as yf
import time
import pandas as pd
import datetime as datetime


def get_current_date():
    return datetime.datetime.now().strftime("%Y-%m-%d")


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
        "snapshot",
        "Snapshot Fundamental",
        "Snapshot all Fundamentals to csv",
    ),
    2,
)


###############################################################################
#
###############################################################################
st.header(option)


###############################################################################
#
###############################################################################
if option == "snapshot":
    if st.button("Start Snapshot Process"):
        st.write("Button is pushed. Snapshot collection is running.")
        with open("datasets/symbols.csv") as f:
            time.sleep(2)
            for line in f:
                if "," not in line:
                    continue
                symbol = line.split(",")[0]
                data = yf.download(symbol, start="2022-06-01", end=get_current_date())
                mydata = "datasets/daily/{}.csv".format(symbol)
                data.to_csv(mydata)
                st.write(mydata)

        st.sidebar.write("--DONE--")
        st.write("--DONE--")
    else:
        st.write("Waiting for the button to be pushed.")


if option == "Snapshot Fundamental":
    if st.button("Start Snapshot Fundamentals Process"):
        st.write("Button is pushed. Snapshot collection is running.")
        with open("datasets/symbols.csv") as f:
            time.sleep(2)
            for line in f:
                if "," not in line:
                    continue
                symbol = line.split(",")[0]
                data = yf.Ticker(symbol)
                data = data.info
                if "companyOfficers" in data:
                    del data["companyOfficers"]
                if "longBusinessSummary" in data:
                    del data["longBusinessSummary"]
                # print(data)
                x = pd.DataFrame([data])
                mydata = "datasets/fundamentals/{}.csv".format(symbol)
                x.to_csv(mydata)
                st.write(mydata)

        st.sidebar.write("--DONE--")
        st.write("--DONE--")
    else:
        st.write("Waiting for the button to be pushed.")

if option == "Snapshot all Fundamentals to csv":
    # Create an empty list to store each DataFrame
    dataframes = []
    if st.button("Start Snapshot Fundamentals Process"):
        st.write("Button is pushed. Snapshot collection is running.")

        with open("datasets/symbols.csv") as f:
            time.sleep(2)
            for line in f:
                if "," not in line:
                    continue
                symbol = line.split(",")[0]
                data = yf.Ticker(symbol).info
                if "companyOfficers" in data:
                    del data["companyOfficers"]
                if "longBusinessSummary" in data:
                    del data["longBusinessSummary"]
                data["symbol"] = symbol
                st.write(symbol)
                # Append DataFrame to list
                dataframes.append(pd.DataFrame([data]))

        # Concatenate all DataFrames
        df = pd.concat(dataframes)

        # Write to CSV
        df.to_csv("all_fundamentals.csv", index=False)

        st.sidebar.write("--DONE--")
        st.write("--DONE--")
    else:
        st.write("Waiting for the button to be pushed.")
