import requests
import xml.etree.ElementTree as ET
import json
import openai

client = openai.OpenAI(
    base_url = "https://api.fireworks.ai/inference/v1",
    api_key = "API-KEY"
)

def get_yahoo_trends(news_type: str):
    """
    Retrieves recent news from Yahoo based on the specified news_type.
    Args:
        news_type (str): The category of news, such as "news," "finance," or "sports."
    Returns:
        str: A JSON-formatted string containing a list of dictionaries, each representing a news item
             with 'title', 'link', and 'pubDate' as key-value pairs.
    """
        
    url = "https://" + news_type + ".yahoo.com/rss/"
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

"""
A list of messages representing a conversation between the user and the assistant.

- System message: Describes the assistant's role and capabilities, encouraging the use of available functions.
- User message: Initiates a query about the main trends in recent news on Yahoo News.
"""
messages = [
    {"role": "system", "content": f"You are a helpful assistant with access to functions." 
     															"Use them if required."},
    {"role": "user", "content": "What is the main trends in the recent news on yahoo finance? "}
]


tools = [ {
        "type": "function",
        "function": {
            "name": "get_yahoo_trends",
            "description": "Get list of recent news from yahoo along with title, link, description, pubDate",
            "parameters": {
                "type": "object",
                "properties": {
                    "news_type": {
                        "type": "string",
                        "enum": ["news", "finance", "sports"],
                        "description": "Categories of the news.",
                    },
                },
                "required": ["news_type"],
            },
        }
    },
]

chat_completion = client.chat.completions.create(
    model="accounts/fireworks/models/fw-function-call-34b-v0",
    messages=messages,
    tools=tools,
    temperature=0.1
)

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
    model="accounts/fireworks/models/fw-function-call-34b-v0",
    messages=messages,
    tools=tools,
    temperature=0.1

)

print(next_chat_completion.choices[0].message.content)
