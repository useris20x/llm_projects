import requests
import time
import openai
import http.client
import json
from tavily import TavilyClient

bot_token = "bot-token"
chat_id = "chat_id"
fireworks_api = "fireworks-API"
tavily_api = "tavily-API"

last_msg_id = []


def echo(chat_id, message_text):
    def tavily_search(query):
        tavily_client = TavilyClient(api_key=tavily_api)
        search_result = tavily_client.get_search_context(query, search_depth="advanced", max_tokens=8000)
        return search_result
    
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
        }
    ]

    client = openai.OpenAI(
        base_url = "https://api.fireworks.ai/inference/v1",
        api_key = fireworks_api
    )
   
    messages = [
        {"role": "system", "content": f"You are a helpful assistant with access to functions. Use them if required."},
        {"role": "user", "content": message_text}
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

    message_text = next_chat_completion.choices[0].message.content

    endpoint = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    params = {
        "chat_id": chat_id,
        "text": f'{message_text}'
    }
    response = requests.post(endpoint, params=params)
    return response.json()

# Define the read_messages function
def read_messages(chat_id):
    global last_msg_id  # Global variable to keep track of the last message ID
    endpoint = f"https://api.telegram.org/bot{bot_token}/getUpdates"  # Telegram API endpoint
    params = {"chat_id": chat_id}  # Parameters for the API request
    response = requests.get(endpoint, params=params)  # Send GET request to the Telegram API
    updates = response.json().get("result", [])  # Extract updates from the response
    
    if updates:
        last_message = updates[-1].get("message", {})  # Get the last message
        if not last_message.get('message_id') in last_msg_id:  # Check if the message ID is not in the list of last message IDs
            last_msg_id.append(last_message.get('message_id'))  # Add the message ID to the list
            return last_message.get('text')  # Return the text of the last message
        else:
            return None  # Return None if the message ID is already in the list
    else:
        return None  # Return None if there are no updates

# Set up the Telegram bot
def main():
    while True:
        last_message = read_messages(chat_id)
        if last_message:
            echo(chat_id, last_message)
        time.sleep(5)

if __name__ == '__main__':
    main()