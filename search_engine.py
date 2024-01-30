import requests
import http.client
import json
import openai
import sys

fireworks_api = "API-KEY"
serp_api = "API-KEY"

client = openai.OpenAI(
    base_url = "https://api.fireworks.ai/inference/v1",
    api_key = fireworks_api
)

def google_search(search_query: str):
    """
    Perform a Google search query.

    Args:
        search_query (str): The search query to be executed.

    Returns:
        str: The result of the Google search.
    """

    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({ "q": search_query  })
    headers = {
        'X-API-KEY': serp_api,
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    result = data.decode("utf-8")

    return result


messages = [
    {"role": "system", "content": f"You are a helpful assistant with access to functions. Use them if required."},
    {"role": "user", "content": "What is the Apple Inc? "}
]


tools = [ {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "Retrieve information through a Google search",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Perform a search query..",
                    },
                },
                "required": ["search_query"],
            },
        },
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
