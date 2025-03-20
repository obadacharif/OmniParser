import asyncio
import os
from dotenv import load_dotenv
import requests
import base64
import pyautogui
from dataclasses import dataclass
import uuid
import time

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OMNIPARSER_URL = os.getenv("OMNIPARSER_URL")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

if not OMNIPARSER_URL:
    raise ValueError("OMNIPARSER_URL not found in environment variables")

# Set the API key for OpenAI
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OMNIPARSER_URL"] = OMNIPARSER_URL

from agents import Agent, Runner, function_tool, RunContextWrapper, trace, TResponseInputItem


@dataclass
class ParsedScreenshot:  
    parsed_content: str



def format_icons(data):
    result = []
    for key, value in data.items():
        attributes = {k: v for k, v in value.items() if k != "bbox"}
        attr_str = ", ".join(f"{k}: {v}" for k, v in attributes.items())
        result.append(f"{key} | {attr_str}")
    return "\n".join(result)


@function_tool
def parse_user_screen(wrapper: RunContextWrapper[ParsedScreenshot]) -> str:
    # Omniparser API URL
    url = OMNIPARSER_URL

    # Request payload
    data = {
        "box_threshold": "0.05",
        "iou_threshold": "0.1",
        "use_paddleocr": "true",
        "imgsz": "640"
    }
    
    
    print (" ----- parse_user_screen tool ----")
    if(wrapper.context.parsed_content):
        print("I've a context !!")
    else:
        print("no context yet !")
    # Take a screenshot using pyautogui
    screenshot = pyautogui.screenshot()
    
    # Save the screenshot temporarily
    temp_image_path = "temp_screenshot.png"
    screenshot.save(temp_image_path)
    
    # Prepare the file for the request
    files = {"image": open(temp_image_path, "rb")}
    
    # Sending the request
    response = requests.post(url, headers={"Accept": "application/json"}, files=files, data=data)

    # Check if the request was successful
    if response.status_code == 200:
        response_data = response.json()
        
        # Save parsed content (optional)
        parsed_content = response_data.get("parsed_content", "")
        
        formatted_content = format_icons(parsed_content)
        
        print("Parsed Content:", formatted_content)
        
        # Add the latest parsed data to the context
        wrapper.context.parsed_content = parsed_content;
        # Decode and save the image
        if "image_base64" in response_data:
            with open("latest.png", "wb") as f:
                f.write(base64.b64decode(response_data["image_base64"]))
            print("Image saved as latest.png")
        else:
            print("No image data found in response.")
        
        return formatted_content
    else:
        print(f"Error: {response.status_code}, {response.text}")
        
        return ""






async def main():
    parsed_screenshot = ParsedScreenshot(parsed_content="")
    
    agent = Agent[ParsedScreenshot](
        name="UI Agent",
        instructions="""
        You are a helpful UI parsing agent, parse the screen using parse_user_screen tool and guide the user to take the next action, 
        you should always specify the targeted element from the parsed screen
        
        Output Format: 
        Action:
        Element:
        """,
        # instructions="you're a helpful assistant"
        tools=[parse_user_screen]
    )
    input_items: list[TResponseInputItem] = []
    conversation_id = uuid.uuid4().hex[:16]
    
    print (conversation_id)
    while True:
        user_input = input("Enter your message: ")
        
        # Sleep for one second to avoid overwhelming the API
        time.sleep(1)
        with trace("UI Agent workflow", group_id=conversation_id):  
            input_items.append({"content": user_input, "role": "user"})
            #result = await Runner.run(agent, input=input_items, context=parsed_screenshot)
            result = await Runner.run(agent, input=user_input, context=parsed_screenshot)
            
            
            print(result.final_output)
            input_items = result.to_input_list()
    # The weather in Tokyo is sunny.


if __name__ == "__main__":
    asyncio.run(main())