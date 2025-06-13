#!/usr/bin/env python3
"""
Slack Historical Message Scraper
Designed to collect ALL historical messages from Slack channels and organize them
in the same folder structure as the daily scraper.

This script:
- Fetches ALL messages from each channel (no time limits)
- Organizes data by actual message timestamps
- Handles large datasets with proper pagination
- Includes resume capability for interrupted runs
- Respects API rate limits with intelligent backoff
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def setup_logging():
    """Setup detailed logging for the historical scraper."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"slack_past_scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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


def get_workspace_info(client, logger):
    """Get workspace information for folder organization."""
    try:
        auth_resp = client.auth_test()
        workspace_name = auth_resp.get("team", "unknown-workspace")
        workspace_name = workspace_name.lower().replace(" ", "-")
        return workspace_name
    except Exception as e:
        logger.warning(f"Could not get workspace info: {e}. Using 'unknown-workspace'")
        return "unknown-workspace"


def get_channels(client, logger):
    """Fetch all public channels."""
    logger.info("Fetching all public channels...")
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
    
    logger.info(f"Found {len(channels)} public channels")
    return channels


def load_progress_state(channel_name):
    """Load progress state for a channel to enable resume capability."""
    progress_dir = "progress"
    os.makedirs(progress_dir, exist_ok=True)
    progress_file = os.path.join(progress_dir, f"{channel_name}_progress.json")
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except:
            pass
    
    return {
        "oldest_processed": None,
        "total_messages": 0,
        "last_cursor": None,
        "completed": False
    }


def save_progress_state(channel_name, progress):
    """Save progress state for a channel."""
    progress_dir = "progress"
    os.makedirs(progress_dir, exist_ok=True)
    progress_file = os.path.join(progress_dir, f"{channel_name}_progress.json")
    
    try:
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not save progress for {channel_name}: {e}")


def fetch_all_channel_messages(client, channel_id, channel_name, logger):
    """
    Fetch ALL historical messages from a channel.
    Uses pagination and handles rate limits appropriately.
    """
    logger.info(f"Starting historical fetch for channel: {channel_name}")
    
    # Load existing progress
    progress = load_progress_state(channel_name)
    
    if progress["completed"]:
        logger.info(f"Channel {channel_name} already completed. Skipping...")
        return []
    
    all_messages = []
    cursor = progress.get("last_cursor")
    total_fetched = progress["total_messages"]
    
    # Set up pagination parameters
    oldest = None  # No time limit - get everything
    limit = 200    # Maximum allowed by Slack API
    
    logger.info(f"Resuming from: {total_fetched} messages already processed")
    
    while True:
        try:
            # Fetch messages with pagination
            params = {
                "channel": channel_id,
                "limit": limit,
                "inclusive": True
            }
            
            if cursor:
                params["cursor"] = cursor
            if oldest:
                params["oldest"] = oldest
            
            logger.debug(f"Fetching batch with cursor: {cursor}")
            
            result = client.conversations_history(**params)
            
            if not result["ok"]:
                logger.error(f"API returned error: {result}")
                break
            
            messages = result["messages"]
            
            if not messages:
                logger.info(f"No more messages found for {channel_name}")
                break
            
            all_messages.extend(messages)
            total_fetched += len(messages)
            
            # Update progress
            progress["total_messages"] = total_fetched
            progress["last_cursor"] = cursor
            if messages:
                progress["oldest_processed"] = messages[-1]["ts"]
            
            logger.info(f"  Fetched {len(messages)} messages (total: {total_fetched})")
            
            # Check if there are more messages
            if not result.get("has_more", False):
                logger.info(f"Reached end of messages for {channel_name}")
                progress["completed"] = True
                break
            
            # Get next cursor
            cursor = result.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                logger.info(f"No more cursors available for {channel_name}")
                progress["completed"] = True
                break
            
            # Save progress every 1000 messages
            if total_fetched % 1000 == 0:
                save_progress_state(channel_name, progress)
                logger.info(f"Progress saved: {total_fetched} messages processed")
            
            # Rate limiting: be respectful to the API
            time.sleep(1.1)  # Slack allows ~1 request per second for this endpoint
            
        except SlackApiError as e:
            if e.response.status_code == 429:
                # Handle rate limit
                retry_after = int(e.response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after + 5)  # Add extra buffer
                continue
            elif e.response["error"] == "not_in_channel":
                logger.warning(f"Bot not in channel {channel_name}. Skipping...")
                progress["completed"] = True
                break
            else:
                logger.error(f"API error for channel {channel_name}: {e}")
                save_progress_state(channel_name, progress)
                raise
        
        except Exception as e:
            logger.error(f"Unexpected error for channel {channel_name}: {e}")
            save_progress_state(channel_name, progress)
            raise
    
    # Final progress save
    save_progress_state(channel_name, progress)
    
    logger.info(f"Completed fetching {total_fetched} total messages from {channel_name}")
    return all_messages


def organize_messages_by_date(messages):
    """Organize messages by their timestamp date."""
    messages_by_date = defaultdict(list)
    
    for msg in messages:
        try:
            # Convert Slack timestamp to date
            msg_timestamp = float(msg.get("ts", 0))
            msg_date = datetime.fromtimestamp(msg_timestamp).strftime("%Y-%m-%d")
            messages_by_date[msg_date].append(msg)
        except (ValueError, OSError) as e:
            # Skip messages with invalid timestamps
            logging.getLogger(__name__).warning(f"Invalid timestamp in message: {msg.get('ts', 'unknown')}")
            continue
    
    return messages_by_date


def save_historical_messages(messages, channel_name, workspace_name, logger):
    """Save historical messages using the organized folder structure."""
    if not messages:
        logger.info(f"No messages to save for {channel_name}")
        return 0
    
    logger.info(f"Organizing and saving {len(messages)} messages for {channel_name}")
    
    # Organize messages by date
    messages_by_date = organize_messages_by_date(messages)
    
    total_saved = 0
    dates_processed = []
    
    for date_str, date_messages in messages_by_date.items():
        # Create folder structure: data/slack/{workspace}/slack_to_{date}/{channel}/
        base_dir = f"data/slack/{workspace_name}/slack_to_{date_str}/{channel_name}"
        os.makedirs(base_dir, exist_ok=True)
        
        out_path = f"{base_dir}/{date_str}.json"
        
        # Load existing messages if file exists
        existing_messages = []
        if os.path.exists(out_path):
            try:
                with open(out_path, 'r') as f:
                    existing_messages = json.load(f)
                logger.debug(f"Loaded {len(existing_messages)} existing messages from {out_path}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not read existing file {out_path}: {e}")
        
        # Merge and deduplicate by timestamp
        existing_ts = {msg.get("ts") for msg in existing_messages}
        new_messages = [msg for msg in date_messages if msg.get("ts") not in existing_ts]
        
        if new_messages:
            all_messages = existing_messages + new_messages
            # Sort by timestamp (oldest first)
            all_messages.sort(key=lambda x: float(x.get("ts", 0)))
            
            try:
                with open(out_path, 'w') as f:
                    json.dump(all_messages, f, indent=2)
                
                total_saved += len(new_messages)
                dates_processed.append(date_str)
                logger.info(f"  Saved {len(new_messages)} new messages for {date_str}")
                
            except IOError as e:
                logger.error(f"Failed to save messages to {out_path}: {e}")
        else:
            logger.debug(f"  No new messages for {date_str} (all already exist)")
    
    logger.info(f"Saved {total_saved} total new messages across {len(dates_processed)} dates")
    if dates_processed:
        logger.info(f"Date range: {min(dates_processed)} to {max(dates_processed)}")
    
    return total_saved


def main():
    """Main function to run the historical Slack scraper."""
    logger = setup_logging()
    logger.info("="*60)
    logger.info("Starting Slack Historical Message Scraper")
    logger.info("="*60)
    
    try:
        # Load environment and setup
        token = load_environment()
        client = WebClient(token=token)
        
        # Get workspace info
        workspace_name = get_workspace_info(client, logger)
        logger.info(f"Workspace: {workspace_name}")
        
        # Fetch all channels
        channels = get_channels(client, logger)
        
        if not channels:
            logger.warning("No channels found! Check bot permissions.")
            return
        
        # Process each channel
        total_messages_saved = 0
        completed_channels = 0
        
        logger.info("="*60)
        logger.info("Starting historical data collection...")
        logger.info("="*60)
        
        for i, channel in enumerate(channels, 1):
            channel_id = channel["id"]
            channel_name = channel["name"]
            
            logger.info(f"\n[{i}/{len(channels)}] Processing channel: {channel_name}")
            logger.info("-" * 40)
            
            start_time = time.time()
            
            # Fetch all historical messages
            messages = fetch_all_channel_messages(client, channel_id, channel_name, logger)
            
            if messages:
                # Save organized messages
                saved_count = save_historical_messages(messages, channel_name, workspace_name, logger)
                total_messages_saved += saved_count
                completed_channels += 1
                
                elapsed = time.time() - start_time
                logger.info(f"Channel {channel_name} completed in {elapsed:.1f}s")
                logger.info(f"  Total messages fetched: {len(messages)}")
                logger.info(f"  New messages saved: {saved_count}")
            else:
                logger.info(f"Channel {channel_name}: No messages accessible")
            
            # Pause between channels to be respectful
            if i < len(channels):
                logger.info("Pausing 5 seconds before next channel...")
                time.sleep(5)
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("HISTORICAL SCRAPING COMPLETED!")
        logger.info("="*60)
        logger.info(f"Total channels processed: {len(channels)}")
        logger.info(f"Channels with accessible messages: {completed_channels}")
        logger.info(f"Total new messages saved: {total_messages_saved}")
        logger.info(f"Data organized in: data/slack/{workspace_name}/")
        
        # Clean up progress files for completed channels
        try:
            import shutil
            if os.path.exists("progress"):
                logger.info("Cleaning up progress files...")
                shutil.rmtree("progress")
        except:
            pass
        
        logger.info("="*60)
        logger.info("Historical scraping is now complete!")
        logger.info("You can now use slack_daily_scraper.py for ongoing updates.")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("\n" + "="*40)
        logger.info("Scraping interrupted by user")
        logger.info("Progress has been saved and can be resumed")
        logger.info("="*40)
        
    except Exception as e:
        logger.error(f"Historical scraper failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()