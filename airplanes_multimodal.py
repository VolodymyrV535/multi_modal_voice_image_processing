# imports
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import gradio as gr

# Some imports for handling images
import base64
from io import BytesIO
from PIL import Image

# text to audio 
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import subprocess
import time

# audio to text
import speech_recognition as sr


# Initialization
load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
MODEL = "gpt-4o-mini"
openai = OpenAI()

# program
# system message for openai api
system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."

# additional model for translation
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
claude = anthropic.Anthropic()
system_message_translator = "You are a helpful assistant for translation from English to Ukraininan."
system_message_translator += "Give precise translation for a given input text."


# Let's start by making a useful function
ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

# function to be used as a tool1 for openai
def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")


# There's a particular dictionary structure that's required to describe our function:
price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}


# # tool2 for booking flights
booked_flights = []

def book_flight(destination_city, amount, name):
    print(f"Tool book_flight called for {destination_city}, {amount}, {name}")
    city = destination_city.lower()
    one_ticket_price = get_ticket_price(city)
    if one_ticket_price == "Unknown":
        return "No flights for that city"
    else:
        total_price = int(one_ticket_price.strip('$')) * int(amount)
        booked_flights.append({
            "buyer": {name},
            "amount": {amount},
            "destination_city": {destination_city}
        })
        return f"{amount} ticket(s) to {destination_city} booked for {name}"
    
# There's a particular dictionary structure that's required to describe our function:
book_function = {
    "name": "book_flight",
    "description": "Books ticket(s) to the destination city if flight exists. Call this whenever you need to book ticket for the peron, for example when a customer asks 'book a ticket or tickets to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
            "name": {
                "type": "string",
                "description": "The name of the customer",
             },
            "amount":{
                "type": "string",
                "description": "amount of tickets to buy",
            }
        },
        "required": ["destination_city", "name", "amount"],
        "additionalProperties": False
    }
}

# # And this is included in a list of tools:
tools = [{"type": "function", "function": price_function},{"type": "function", "function": book_function}]


# We have to write that function handle_tool_call:
def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    print("debugging here:")
    print(f"tool_call:{tool_call}" )
    print("______________________")
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    city = arguments.get('destination_city')
    
    if function_name == "get_ticket_price":
        price = get_ticket_price(city)
        response = {
            "role": "tool",
            "content": json.dumps({"destination_city": city,"price": price}),
            "tool_call_id": tool_call.id
        }
        return response, city
    
    elif function_name == "book_flight":
        amount = arguments.get('amount')
        name = arguments.get('name')
        answer = book_flight(city, amount, name)
        
        response = {
            "role": "tool",
            "content": json.dumps({"destination_city": city, "answer": answer}),
            "tool_call_id": tool_call.id
        }
        return response, city
    else:
        print("debugging error")
        return response, city


# image generator
def artist(city):
    image_response = openai.images.generate(
            model="dall-e-3",
            prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))


def play_audio(audio_segment):
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_audio.wav")
    try:
        audio_segment.export(temp_path, format="wav")
        time.sleep(3) # Student Dominic found that this was needed. You could also try commenting out to see if not needed on your PC
        subprocess.call([
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-hide_banner",
            temp_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass
 
def talker(message):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",  # Also, try replacing onyx with alloy
        input=message
    )
    audio_stream = BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")
    play_audio(audio)


# Translator function using Claude API
def translator(history):
    translated_text = ""

    # Only translate new messages (those after the last translated index)
    for i, message in enumerate(history):
        # Translate message using Claude API
        result = claude.messages.stream(
            model="claude-3-5-sonnet-latest",
            max_tokens=200,
            temperature=0.7,
            system=system_message_translator,
            messages=[
                {"role": "user", "content": message['content']},
            ],
        )

        response = ""
        with result as stream:
            for text in stream.text_stream:
                response += text or ""

        translated_text += "\n" + response + "\n"
    
    return translated_text


# Speech-to-text function using SpeechRecognition library
def speech_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    # Ensure that the path matches an actual file
    if isinstance(audio_file_path, str):
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            text = "Could not understand audio"
        except sr.RequestError:
            text = "Speech recognition service unavailable"
    else:
        text = "Invalid file path for audio input"
    return text

    
# chat functionality backend
def chat(history):
    messages = [{"role": "system", "content": system_message}] + history
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    image = None
    
    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        image = artist(city)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
        
    reply = response.choices[0].message.content
    history += [{"role":"assistant", "content":reply}]

    # Comment out or delete the next line if you'd rather skip Audio for now..
    talker(reply)
    
    return history, image


# Passing in inbrowser=True in the last line will cause a Gradio window to pop up immediately.
# More involved Gradio code as we're not using the preset Chat interface!
with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        translated_textbox = gr.Textbox(label="Переклад:")
        image_output = gr.Image(height=500)
    with gr.Row():
        entry = gr.Textbox(label="Chat with our AI Assistant:")
        record_button = gr.Audio(sources="microphone", type="filepath", label="Record Voice")
    with gr.Row():
        clear = gr.Button("Clear")

    def do_entry(message, history):
        history.append({"role": "user", "content": message})
        return "", history

    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
        chat, inputs=chatbot, outputs=[chatbot, image_output]
    ).then(
        translator, inputs=chatbot, outputs=translated_textbox  # Update with translator output
    )
    clear.click(lambda: None, inputs=None, outputs=[chatbot, translated_textbox], queue=False)  # Clear both elements
    
    # Process audio input
    record_button.change(speech_to_text, inputs=record_button, outputs=entry)

ui.launch(inbrowser=True)