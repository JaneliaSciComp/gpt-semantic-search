# import os
# import time
# from slack_bolt import App
# from slack_bolt.adapter.socket_mode import SocketModeHandler
# from threading import Thread, Event

# # Import your SemanticSearchService
# from generate_full_response import SemanticSearchService

# app = App(token=os.environ["SLACK_BOT_TOKEN"])

# # Initialize the SemanticSearchService
# weaviate_url = "http://localhost:8777"
# service = SemanticSearchService(weaviate_url)

# def update_message(client, channel, timestamp, text):
#     client.chat_update(
#         channel=channel,
#         ts=timestamp,
#         text=text
#     )

# def generate_response_with_animation(event, say, client):
#     # Extract the text after the bot mention
#     text = event['text'].split('>')[1].strip()
    
#     # Send an initial message
#     result = say("Thinking...")
#     message_ts = result['ts']
    
#     # Start the loading animation
#     loading_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
#     i = 0
#     stop_event = Event()
    
#     def animate():
#         nonlocal i
#         while not stop_event.is_set():
#             update_message(client, event['channel'], message_ts, f"{loading_chars[i]} Thinking...")
#             i = (i + 1) % len(loading_chars)
#             time.sleep(0.1)
    
#     # Start the animation in a separate thread
#     animation_thread = Thread(target=animate)
#     animation_thread.start()
    
#     try:
#         # Generate a response using the SemanticSearchService
#         response = service.generate_response(text)
#     finally:
#         # Stop the animation thread
#         stop_event.set()
#         animation_thread.join()
    
#     # Update the message with the final response
#     update_message(client, event['channel'], message_ts, response)

# @app.event("app_mention")
# def handle_mention(event, say, client):
#     Thread(target=generate_response_with_animation, args=(event, say, client)).start()

# if __name__ == "__main__":
#     handler = SocketModeHandler(app, str(os.environ["SLACK_APP_TOKEN"]))
#     handler.start()

import os
import time
import random
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from threading import Thread, Event
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Import your SemanticSearchService
from generate_full_response import SemanticSearchService
import logging
logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = App(token=os.environ["SLACK_BOT_TOKEN"])

# # Initialize the SemanticSearchService
weaviate_url = "http://localhost:8777"
service = SemanticSearchService(weaviate_url)

# # Conversation history (we'll keep this for future use)
# conversation_history = {}

def update_message(client, channel, timestamp, text, blocks=None):
    client.chat_update(
        channel=channel,
        ts=timestamp,
        text=text,
        blocks=blocks
    )

def generate_response_with_animation(event, say, client):
    user_id = event['user']
    channel = event['channel']
    text = event['text'].split('>')[1].strip()
    
    # Initialize or update conversation history
    # if user_id not in conversation_history:
    #     conversation_history[user_id] = []
    # conversation_history[user_id].append(f"Human: {text}")
    
    # Send an initial message with a random thinking phrase
    thinking_phrases = [
        "Thinking...",
        "Processing your request...",
        "Hmmmmm..."
    ]
    initial_message = random.choice(thinking_phrases)
    result = say(initial_message)
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
    
    # Update conversation history
    # conversation_history[user_id].append(f"Assistant: {response}")
    
    # # Truncate conversation history if it gets too long
    # if len(conversation_history[user_id]) > 10:
    #     conversation_history[user_id] = conversation_history[user_id][-10:]
    
    # Create a formatted response with markdown
    formatted_response = f"*Here's what I found:*\n\n{response}"
    
    # Update the message with the final response
    # blocks = [
    #     {
    #         "type": "section",
    #         "text": {"type": "mrkdwn", "text": formatted_response}
    #     },
    #     {
    #         "type": "context",
    #         "elements": [
    #             {
    #                 "type": "mrkdwn",
    #                 "text": "If you found this helpful, react with :thumbsup:. If not, react with :thumbsdown:."
    #             }
    #         ]
    #     }
    # ]
    update_message(client, channel, message_ts, formatted_response)

@app.event("app_mention")
def handle_mention(event, say, client):
    Thread(target=generate_response_with_animation, args=(event, say, client)).start()

# @app.event("reaction_added")
# def handle_reaction(event, say):
#     logger.debug(f"Reaction event received: {event}")
    
#     reaction = event.get("reaction")
#     user = event.get("user")
#     item = event.get("item", {})
#     ts = item.get("ts")
#     channel = item.get("channel")

#     logger.info(f"Reaction: {reaction}, User: {user}, Timestamp: {ts}, Channel: {channel}")

#     if reaction == "thumbsup":
#         say(text="Thank you for the positive feedback! I'm glad I could help.", channel=channel, thread_ts=ts)
#         logger.info("Positive feedback received")
#     elif reaction == "thumbsdown":
#         say(text="I'm sorry my response wasn't helpful. Could you provide more details about what you're looking for?", channel=channel, thread_ts=ts)
#         logger.info("Negative feedback received")
#     else:
#         logger.info(f"Reaction {reaction} received but not handled")


if __name__ == "__main__":
    handler = SocketModeHandler(app, str(os.environ["SLACK_APP_TOKEN"]))
    handler.start()
