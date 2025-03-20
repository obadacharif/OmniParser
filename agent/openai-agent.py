import asyncio
import os
from dotenv import load_dotenv
import requests
import base64
import pyautogui
from dataclasses import dataclass
import uuid

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
        
        print("Parsed Content:", parsed_content)
        
        # Add the latest parsed data to the context
        wrapper.context.parsed_content = parsed_content;
        # Decode and save the image
        if "image_base64" in response_data:
            with open("latest.png", "wb") as f:
                f.write(base64.b64decode(response_data["image_base64"]))
            print("Image saved as latest.png")
        else:
            print("No image data found in response.")
        
        #TODO remove bboxes
        return parsed_content
    else:
        print(f"Error: {response.status_code}, {response.text}")
        
        return ""






async def main():
    parsed_screenshot = ParsedScreenshot(parsed_content="")
    
    agent = Agent[ParsedScreenshot](
        name="UI Agent",
        instructions="""
        You are a helpful UI parsing agent, parse the screen and guide the user to take the next action, 
        you should always specify the targeted element
        """,
        # instructions="you're a helpful assistant"
        tools=[parse_user_screen]
    )
    input_items: list[TResponseInputItem] = []
    conversation_id = uuid.uuid4().hex[:16]
    
    print (conversation_id)
    while True:
        user_input = input("Enter your message: ")
        
        with trace("UI Agent workflow", group_id=conversation_id):  
            input_items.append({"content": user_input, "role": "user"})
            result = await Runner.run(agent, input=input_items, context=parsed_screenshot)
            
            print(result.final_output)
            input_items = result.to_input_list()
    # The weather in Tokyo is sunny.


if __name__ == "__main__":
    asyncio.run(main())