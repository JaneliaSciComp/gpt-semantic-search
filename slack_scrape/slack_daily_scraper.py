#!/usr/bin/env python3
"""
Daily Slack History Scraper
Designed to run as a scheduled job (cron) to collect Slack messages daily.
Maintains incremental state and organizes data by channel and date.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def setup_logging():
    """Setup logging for the scheduled job."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"slack_scraper_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_environment():
    """Load environment variables and validate token."""
    token = os.getenv("SCRAPING_SLACK_BOT_TOKEN")
    
    if not token:
        raise ValueError("SCRAPING_SLACK_BOT_TOKEN not found in environment variables. Please set it in your bash profile.")
    
    return token


def get_state_info(state_file="last_run.txt"):
    """Get the timestamp of the last successful run."""
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                last_ts = float(f.read().strip())
                return last_ts
        except (ValueError, IOError) as e:
            logging.warning(f"Could not read state file: {e}. Starting from 24 hours ago.")
            return time.time() - 86400  # 24 hours ago as fallback
    else:
        logging.info("No state file found. Starting from 24 hours ago.")
        return time.time() - 86400  # 24 hours ago for first run


def update_state(new_ts, state_file="last_run.txt"):
    """Update the state file with the current run timestamp."""
    try:
        with open(state_file, 'w') as f:
            f.write(str(new_ts))
        logging.info(f"State updated to timestamp: {new_ts}")
    except IOError as e:
        logging.error(f"Failed to update state file: {e}")


def get_channels(client, logger):
    """Fetch all public channels."""
    logger.info("Fetching channel list...")
    channels = []
    cursor = None
    
    while True:
        try:
            resp = client.conversations_list(
                types="public_channel", 
                exclude_archived=True, 
                limit=200, 
                cursor=cursor
            )
            channels.extend(resp["channels"])
            cursor = resp.get("response_metadata", {}).get("next_cursor")
            
            if not cursor:
                break
                
        except SlackApiError as e:
            logger.error(f"Failed to fetch channels: {e}")
            raise
    
    logger.info(f"Found {len(channels)} channels")
    return channels


def fetch_channel_messages(client, channel_id, channel_name, oldest_ts, logger):
    """Fetch messages from a specific channel since the given timestamp."""
    all_msgs = []
    
    try:
        result = client.conversations_history(
            channel=channel_id, 
            oldest=oldest_ts, 
            limit=200
        )
        
    except SlackApiError as e:
        if e.response.status_code == 429:
            # Handle rate limit
            retry_after = int(e.response.headers.get("Retry-After", 1))
            logger.warning(f"Rate limited for channel {channel_name}. Waiting {retry_after + 1} seconds...")
            time.sleep(retry_after + 1)
            result = client.conversations_history(
                channel=channel_id, 
                oldest=oldest_ts, 
                limit=200
            )
        elif e.response["error"] == "not_in_channel":
            logger.warning(f"Skipping channel {channel_name} - bot not in channel")
            return []
        else:
            logger.error(f"API error for channel {channel_name}: {e}")
            raise
    
    all_msgs.extend(result["messages"])
    
    # Handle pagination
    while result.get("has_more"):
        cursor = result["response_metadata"]["next_cursor"]
        try:
            result = client.conversations_history(
                channel=channel_id, 
                oldest=oldest_ts, 
                limit=200, 
                cursor=cursor
            )
            all_msgs.extend(result["messages"])
        except SlackApiError as e:
            logger.error(f"Error during pagination for channel {channel_name}: {e}")
            break
    
    return all_msgs


def get_workspace_info(client, logger):
    """Get workspace information for folder organization."""
    try:
        auth_resp = client.auth_test()
        workspace_name = auth_resp.get("team", "unknown-workspace")
        # Clean workspace name for use in folder paths
        workspace_name = workspace_name.lower().replace(" ", "-")
        return workspace_name
    except Exception as e:
        logger.warning(f"Could not get workspace info: {e}. Using 'unknown-workspace'")
        return "unknown-workspace"


def save_messages(messages, channel_name, workspace_name, logger):
    """Save messages to the date-based folder structure matching existing pattern."""
    if not messages:
        return
    
    # Get today's date for the slack_to folder
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    # Create the folder structure: data/slack/{workspace}/slack_to_{today}/{channel}/
    base_dir = f"./../data/slack/{workspace_name}/slack_to_{today_date}/{channel_name}"
    os.makedirs(base_dir, exist_ok=True)
    
    # Group messages by date based on their timestamp
    messages_by_date = {}
    
    for msg in messages:
        # Convert Slack timestamp to date
        msg_timestamp = float(msg.get("ts", 0))
        msg_date = datetime.fromtimestamp(msg_timestamp).strftime("%Y-%m-%d")
        
        if msg_date not in messages_by_date:
            messages_by_date[msg_date] = []
        messages_by_date[msg_date].append(msg)
    
    # Save each date's messages to its own file
    saved_count = 0
    for date_str, date_messages in messages_by_date.items():
        out_path = f"{base_dir}/{date_str}.json"
        
        # If file exists, load existing messages and merge
        existing_messages = []
        if os.path.exists(out_path):
            try:
                with open(out_path, 'r') as f:
                    existing_messages = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not read existing file {out_path}: {e}")
        
        # Merge new messages with existing ones (avoid duplicates by timestamp)
        existing_ts = {msg.get("ts") for msg in existing_messages}
        new_messages = [msg for msg in date_messages if msg.get("ts") not in existing_ts]
        
        if new_messages:
            all_messages = existing_messages + new_messages
            # Sort by timestamp
            all_messages.sort(key=lambda x: float(x.get("ts", 0)))
            
            try:
                with open(out_path, 'w') as f:
                    json.dump(all_messages, f, indent=2)
                logger.info(f"Saved {len(new_messages)} new messages to {out_path}")
                saved_count += len(new_messages)
            except IOError as e:
                logger.error(f"Failed to save messages to {out_path}: {e}")
    
    return saved_count


def main():
    """Main function to run the daily Slack scraper."""
    logger = setup_logging()
    logger.info("Starting daily Slack message scraper...")
    
    try:
        # Load environment and setup
        token = load_environment()
        client = WebClient(token=token)
        
        # Get workspace info for folder structure
        workspace_name = get_workspace_info(client, logger)
        logger.info(f"Workspace: {workspace_name}")
        
        # Get state info
        last_ts = get_state_info()
        current_ts = time.time()
        
        logger.info(f"Fetching messages since {datetime.fromtimestamp(last_ts).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Fetch all public channels
        channels = get_channels(client, logger)
        
        if not channels:
            logger.warning("No channels found! Check bot permissions.")
            return
        
        # Process each channel
        total_messages = 0
        processed_channels = 0
        skipped_channels = []
        
        for channel in channels:
            channel_id = channel["id"]
            channel_name = channel["name"]
            
            logger.info(f"Processing channel: {channel_name}")
            
            messages = fetch_channel_messages(client, channel_id, channel_name, last_ts, logger)
            
            if messages:
                saved_count = save_messages(messages, channel_name, workspace_name, logger)
                total_messages += saved_count
                processed_channels += 1
                logger.info(f"Channel {channel_name}: {len(messages)} fetched, {saved_count} new messages saved")
            else:
                logger.info(f"Channel {channel_name}: No new messages")
            
            # Small delay to be respectful to API
            time.sleep(0.1)
        
        # Update state
        update_state(current_ts)
        
        logger.info("="*50)
        logger.info(f"Scraping completed successfully!")
        logger.info(f"Total channels found: {len(channels)}")
        logger.info(f"Channels with new messages: {processed_channels}")
        logger.info(f"Total new messages saved: {total_messages}")
        logger.info(f"Data saved to: data/slack/{workspace_name}/slack_to_{datetime.now().strftime('%Y-%m-%d')}/")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Scraper failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name __ == "__main__":
    main()