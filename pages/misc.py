import streamlit as st
import requests

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
        "Link Catalogue",
        "stocktwits",
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
if option == "stocktwits":
    symbol = st.sidebar.text_input("Symbol", value="AAPL", max_chars=5)

    r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json")

    if r.status_code == 200:
        data = r.json()
        for message in data["messages"]:
            st.image(message["user"]["avatar_url"])
            st.write(f"User: {message['user']['username']}")
            st.write(f"Created: {message['created_at']}")
            st.write(message["body"])
    else:
        st.write("BAD RETURN CODE!")


if option == "Link Catalogue":
    st.write(" dicussion")
    st.write("community.backtrader.com [link](https://community.backtrader.com/)")

    st.write("Technical Analysis:")

    # ta - Technical Analysis Library in Python based on pandas
    st.write(
        "Technical Analysis Library in Python based on pandas [link](https://technical-analysis-library-in-python.readthedocs.io)"
    )
    # TA-Lib - Python wrapper for TA-Lib
    st.write(
        "TA-Lib - Python wrapper for TA-Lib [link](https://github.com/TA-Lib/ta-lib-python)"
    )
    # bta-lib - backtrader ta-lib
    st.write("bta-lib - backtrader ta-lib [link](https://btalib.backtrader.com/)")
    # pandas-ta - Pandas Technical Analysis (Pandas TA) is an easy to use library that leverages the Pandas library with more than 120 Indicators and Utility functions
    st.write(
        "pandas-ta - Pandas Technical Analysis (Pandas TA) is an easy to use library that leverages the Pandas library with more than 120 Indicators and Utility functions [link](https://github.com/twopirllc/pandas-ta)"
    )
    # tulipy - Python bindings for Tulip Indicators
    st.write(
        "tulipy - Python bindings for Tulip Indicators [link](https://github.com/cirla/tulipy)"
    )

    st.write("Misc:")
    st.write("TradeKit [link](https://github.com/hackingthemarkets/tradekit)")

    st.write("Hacking the Markets [link](https://github.com/hackingthemarkets)")
    st.write("Bext of streamlit [link](https://github.com/jrieke/best-of-streamlit)")
    st.write(
        "streamlit graph viz [link](https://github.com/ChrisDelClea/streamlit-agraph)"
    )
    st.write(
        "personal-website-streamlit - uses containers [link]https://github.com/Sven-Bo/personal-website-streamlit"
    )
    st.write(
        "PyGWalker streamlit julyter [link](https://github.com/Sven-Bo/PyGWalker-Guide-for-Streamlit-and-Jupyter)"
    )

    st.write(
        "streamlit option menu [link](https://github.com/victoryhb/streamlit-option-menu)"
    )

    st.write("stream lit extras demo [link](https://extras.streamlit.app/)")

    st.write(
        "FINVIZ Screener [link](https://finviz.com/screener.ashx?v=111&f=cap_small,sec_utilities)"
    )


# # ---- PROJECTS ----
# with st.container():
#     st.write("---")
#     st.header("My Projects")
#     st.write("##")
#     image_column, text_column = st.columns((1, 2))
#     with image_column:
#         st.write("img_lottie_animation")
#     with text_column:
#         st.subheader("Integrate Lottie Animations Inside Your Streamlit App")
#         st.write(
#             """
#             Learn how to use Lottie Files in Streamlit!
#             Animations make our web app more engaging and fun, and Lottie Files are the easiest way to do it!
#             In this tutorial, I'll show you exactly how to do it
#             """
#         )
#         st.markdown("[Watch Video...](https://youtu.be/TXSOitGoINE)")
