import os
import time
import random
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from threading import Thread, Event
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from generate_full_response import SemanticSearchService
import logging
import re

logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = App(token=os.environ["SLACK_BOT_TOKEN"])

weaviate_url = os.environ["WEAVIATE_URL"]

service = SemanticSearchService(weaviate_url)

def update_message(client, channel, timestamp, text, blocks=None):
    client.chat_update(
        channel=channel,
        ts=timestamp,
        text=text,
        blocks=blocks
    )

def generate_response_with_animation(text, channel, thread_ts, client):
    # Send an initial message with a random thinking phrase
    thinking_phrases = [
        "Thinking...",
        "Processing your request...",
        "Hmmmmm..."
    ]
    initial_message = random.choice(thinking_phrases)
    result = client.chat_postMessage(
        channel=channel,
        text=initial_message,
        thread_ts=thread_ts  # This ensures the message is posted in the thread if there is one
    )
    message_ts = result['ts']
    
    # Start the loading animation
    loading_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    stop_event = Event()
    
    def animate():
        nonlocal i
        while not stop_event.is_set():
            update_message(client, channel, message_ts, f"{loading_chars[i]} {initial_message}")
            i = (i + 1) % len(loading_chars)
            time.sleep(0.1)
    
    # Start the animation in a separate thread
    animation_thread = Thread(target=animate)
    animation_thread.start()
    
    try:
        # Generate a response using the SemanticSearchService with a timeout
        with ThreadPoolExecutor() as executor:
            future = executor.submit(service.generate_response, text)
            try:
                response = future.result(timeout=30)  # 30-second timeout
            except TimeoutError:
                response = "I'm sorry, but it's taking me longer than expected to generate a response. Could you try rephrasing your question or asking something else?"
    
    except Exception as e:
        response = f"Oops! Something went wrong: {str(e)}"
    
    finally:
        # Stop the animation thread
        stop_event.set()
        animation_thread.join()
    
    formatted_response = f"*Here's what I found:*\n\n{response}"
    
    update_message(client, channel, message_ts, formatted_response)

def process_message(event, client):
    text = event['text']
    channel = event['channel']
    thread_ts = event.get('thread_ts', event['ts'])
    
    # Remove the bot mention if it exists
    bot_id = client.auth_test()["user_id"]
    text = re.sub(f'<@{bot_id}>', '', text).strip()
    
    Thread(target=generate_response_with_animation, args=(text, channel, thread_ts, client)).start()

@app.event("app_mention")
def handle_mention(event, client):
    process_message(event, client)

@app.event("message")
def handle_message(event, client):
    # Ignore messages from bots
    if "bot_id" in event:
        return
    # Process messages in DMs
    logger.info(event)
    if event['channel_type'] == 'im':
        process_message(event, client)

if __name__ == "__main__":
    handler = SocketModeHandler(app, str(os.environ["SLACK_APP_TOKEN"]))
    handler.start()