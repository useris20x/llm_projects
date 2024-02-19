import requests
import time
import openai
import json
import xml.etree.ElementTree as ET
from tavily import TavilyClient
from tradingview_ta import TA_Handler, Interval, Exchange
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys

model_name = "accounts/fireworks/models/fw-function-call-34b-v0"
#"accounts/fireworks/models/firefunction-v1"

fireworks_api = "FIREWORKS-API"
tavily_api = "TAVILY-API"

client = openai.OpenAI(
    base_url = "https://api.fireworks.ai/inference/v1",
    api_key = fireworks_api
)

def tavily_search(query):
    tavily_client = TavilyClient(api_key=tavily_api)
    search_result = tavily_client.get_search_context(query, search_depth="advanced", max_tokens=8000)
    return search_result

def get_recomendation(symbol):
    tesla = TA_Handler(
        symbol=symbol,
        screener="america",
        exchange="NASDAQ",
        interval=Interval.INTERVAL_1_DAY,
    )
    return tesla.get_analysis().summary

def calculate_indicators(symbol):
    # Download historical data for the current date
    now = datetime.today()
    end_date =  now.strftime('%Y-%m-%d')
    start_date = (now - timedelta(days = 30)).strftime('%Y-%m-%d')
    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)

    # Check if there are enough data points to calculate the indicators
    if len(data) < 14:
        return None

    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    rsi_value = RSI.iloc[-1]

    # Calculate Bollinger Bands
    window = 20
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma.iloc[-1] + 2 * std.iloc[-1]
    lower_band = sma.iloc[-1] - 2 * std.iloc[-1]

    # Calculate Aroon
    period = 14
    high_highest = data['High'].rolling(window=period).max().iloc[-1]
    low_lowest = data['Low'].rolling(window=period).min().iloc[-1]
    aroon_up = ((period - (data['High'].iloc[-1] - high_highest)) / period) * 100
    aroon_down = ((period - (data['Low'].iloc[-1] - low_lowest)) / period) * 100

    # Calculate Chaikin Money Flow (CMF)
    mf_multiplier = ((data['Close'].iloc[-1] - data['Low'].iloc[-1]) - (data['High'].iloc[-1] - data['Close'].iloc[-1])) / (data['High'].iloc[-1] - data['Low'].iloc[-1])
    mf_volume = mf_multiplier * data['Volume'].iloc[-1]
    cmf = mf_volume / data['Volume'].iloc[-1]

    return {
        'RSI': round(rsi_value, 2),
        'BBL': {'UpperBand': round(upper_band, 2), 'LowerBand': round(lower_band, 2)},
        'Aroon': {'AroonUp': round(aroon_up, 2), 'AroonDown': round(aroon_down, 2)},
        'CMF': round(cmf, 2)
    }

def get_yahoo_trends():
    """
    Retrieves recent news from Yahoo based on the specified news_type.
    Returns:
        str: A JSON-formatted string containing a list of dictionaries, each representing a news item
             with 'title', 'link', and 'pubDate' as key-value pairs.
    """
        
    url = "https://finance.yahoo.com//rss/"
    response = requests.get(url)
    xml_data = response.text
    root = ET.fromstring(xml_data)

    news_list = []
    for item in root.findall('.//item'):
        title = item.find('title').text
        link = item.find('link').text
        pub_date = item.find('pubDate').text

        news_item = {
            'title': title,
            'link': link,
            'pubDate': pub_date
        }

        news_list.append(news_item)

    return json.dumps(news_list, indent=2)

tools = [ {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": "Get information on recent events from the web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query to use. For example: 'Latest news on Nvidia stock performance'"},
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recomendation",
            "description": "Get current Technical analysis for indices (index) from TradingView.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Input the symbol.'"},
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_indicators",
            "description": "Calculates current trading indicators of the stock: RSI, BBL, Aroon and CMF.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Input the symbol.'"},
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_yahoo_trends",
            "description": "Get the list of recent news from Yahoo finance along with title, link, description, pubDate",
            "parameters": {
                "type": "object",
                "properties": {
                },
            },
        }
    },
]

ticker = "AAPL"

yahoo_trends = "You should evaluate the latest headlines on Yahoo Finance and use that information to make a decision on whether to buy or not buy a " + ticker + " stock. Do not write headlines."
calculated_indicators = "Should I buy " + ticker + " stocks now based on calculated technical indicators ?"
Tradeview_indicators = "Should I buy " + ticker + " stocks now based on Technical analysis from TradingView ?"
web_search = "Should I buy " + ticker + " stocks now based on the recent web information ?"

questions = [yahoo_trends, calculated_indicators, Tradeview_indicators, web_search]

for question in questions:
    
    messages = [
        {"role": "system", "content": f"You are a helpful assistant with access to functions. Use them if required."},
        {"role": "user", "content": question}
    ]

    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        temperature=0.1
    )

    #print(chat_completion.choices[0].message)

    function_call = chat_completion.choices[0].message.tool_calls[0].function
    tool_response = locals()[function_call.name](**json.loads(function_call.arguments))

    agent_response = chat_completion.choices[0].message

    messages.append(
        {
            "role": agent_response.role, 
            "content": "",
            "tool_calls": [
                tool_call.model_dump()
                for tool_call in chat_completion.choices[0].message.tool_calls
            ]
        }
    )

    messages.append(
        {
            "role": "function",
            "content": json.dumps(tool_response)
        }
    )


    next_chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        temperature=0.1

    )

    text = next_chat_completion.choices[0].message.content

    print(question)
    print(text)
    print()

