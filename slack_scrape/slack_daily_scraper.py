#!/usr/bin/env python3
"""Daily Slack Message Scraper - Fast message collection to file system."""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Set
from collections import defaultdict
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def setup_logging() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"slack_daily_scraper_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("slack_daily_scraper")

def get_workspace_info(client: WebClient) -> str:
    try:
        auth_resp = client.auth_test()
        workspace_name = auth_resp.get("team", "unknown-workspace")
        return workspace_name.lower().replace(" ", "-")
    except Exception:
        return "unknown-workspace"


def get_all_users(client: WebClient, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Fetch all users from the workspace."""
    try:
        users = []
        cursor = None
        
        while True:
            resp = client.users_list(limit=200, cursor=cursor)
            users.extend(resp["members"])
            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        
        logger.info(f"Found {len(users)} users")
        return users
        
    except SlackApiError as e:
        logger.error(f"Error fetching users: {e}")
        return []


def get_all_channels(client: WebClient, logger: logging.Logger) -> List[Dict[str, Any]]:
    try:
        channels = []
        cursor = None
        
        while True:
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
        
        logger.info(f"Found {len(channels)} channels")
        return channels
        
    except SlackApiError as e:
        if e.response["error"] == "missing_scope":
            return discover_channels_from_data(logger)
        else:
            raise


def get_specific_channels(channel_names: List[str]) -> List[Dict[str, Any]]:
    return [{"id": name, "name": name} for name in channel_names]


def discover_channels_from_data(logger: logging.Logger) -> List[Dict[str, Any]]:
    discovered_channels: Set[str] = set()
    slack_data_path = "data/slack"
    
    if os.path.exists(slack_data_path):
        try:
            for workspace_dir in os.listdir(slack_data_path):
                workspace_path = os.path.join(slack_data_path, workspace_dir)
                if os.path.isdir(workspace_path):
                    for date_dir in os.listdir(workspace_path):
                        date_path = os.path.join(workspace_path, date_dir)
                        if os.path.isdir(date_path) and date_dir.startswith("slack_to_"):
                            for channel_dir in os.listdir(date_path):
                                channel_path = os.path.join(date_path, channel_dir)
                                if (os.path.isdir(channel_path) and 
                                    not channel_dir.endswith('.json')):
                                    discovered_channels.add(channel_dir)
        except Exception:
            pass
    
    channels = [{"id": name, "name": name} for name in sorted(discovered_channels)]
    logger.info(f"Discovered {len(channels)} channels from data")
    return channels


def fetch_thread_replies(client: WebClient, channel_id: str, thread_ts: str, 
                        logger: logging.Logger) -> List[Dict[str, Any]]:
    """Fetch all replies in a thread using conversations.replies."""
    all_replies = []
    cursor = None
    
    while True:
        try:
            params = {
                "channel": channel_id,
                "ts": thread_ts,
                "limit": 200,
                "include_all_metadata": True
            }
            if cursor:
                params["cursor"] = cursor
            
            result = client.conversations_replies(**params)
            messages = result["messages"]
            
            # Filter out the parent message - only keep actual thread replies
            # Parent message has thread_ts == ts, replies have thread_ts != ts
            thread_replies_only = []
            for msg in messages:
                msg_ts = msg.get("ts")
                msg_thread_ts = msg.get("thread_ts")
                # Only include if this is actually a reply (not the parent)
                if msg_thread_ts and msg_ts and msg_thread_ts != msg_ts:
                    thread_replies_only.append(msg)
            
            all_replies.extend(thread_replies_only)
            
            if not result.get("has_more", False):
                break
            
            cursor = result.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
            
            time.sleep(1.1)
            
        except SlackApiError as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 60))
                time.sleep(retry_after + 5)
                continue
            else:
                logger.warning(f"Error fetching thread replies for {thread_ts}: {e}")
                break
        except Exception as e:
            logger.error(f"Error fetching thread replies: {e}")
            break
    
    return all_replies


def fetch_channel_messages(client: WebClient, channel_id: str, channel_name: str, 
                         oldest_ts: float, logger: logging.Logger) -> List[Dict[str, Any]]:
    all_messages = []
    cursor = None
    
    logger.debug(f"  Fetching messages for {channel_name} since {datetime.fromtimestamp(oldest_ts)}")
    
    while True:
        try:
            params = {
                "channel": channel_id,
                "limit": 200,
                "oldest": str(oldest_ts),
                "inclusive": True,
                "include_all_metadata": True
            }
            if cursor:
                params["cursor"] = cursor
            
            logger.debug(f"  API call params: {params}")
            result = client.conversations_history(**params)
            messages = result["messages"]
            
            logger.debug(f"  API returned {len(messages)} messages")
            
            if not messages:
                break
            
            # Process each message to check for threads
            for message in messages:
                all_messages.append(message)
                
                # Check if this message has thread replies
                thread_ts = message.get("thread_ts")
                reply_count = message.get("reply_count", 0)
                
                # If this is a parent message with replies, fetch the thread
                if (thread_ts and 
                    thread_ts == message.get("ts") and  # This is the parent message
                    reply_count > 0):
                    
                    logger.debug(f"  Fetching {reply_count} thread replies for message {thread_ts}")
                    thread_replies = fetch_thread_replies(client, channel_id, thread_ts, logger)
                    all_messages.extend(thread_replies)
            
            if not result.get("has_more", False):
                break
            
            cursor = result.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
            
            time.sleep(1.1)
            
        except SlackApiError as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 60))
                time.sleep(retry_after + 5)
                continue
            elif e.response["error"] in ["channel_not_found", "not_in_channel"]:
                break
            else:
                raise
        except Exception as e:
            logger.error(f"Error for channel {channel_name}: {e}")
            raise
    
    return all_messages


def organize_messages_by_date(messages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    messages_by_date = defaultdict(list)
    
    for msg in messages:
        try:
            msg_timestamp = float(msg.get("ts", 0))
            msg_date = datetime.fromtimestamp(msg_timestamp).strftime("%Y-%m-%d")
            messages_by_date[msg_date].append(msg)
        except (ValueError, OSError):
            continue
    
    return messages_by_date


def enrich_messages_with_user_profiles(messages: List[Dict[str, Any]], 
                                      users: List[Dict[str, Any]], 
                                      logger: logging.Logger) -> List[Dict[str, Any]]:
    """Enrich messages with user profile information from users list."""
    # Build user ID to profile mapping
    user_profiles = {}
    for user in users:
        if user.get("id") and not user.get("deleted", False):
            user_id = user["id"]
            profile = {
                "real_name": user.get("profile", {}).get("real_name", 
                           user.get("real_name", 
                           user.get("name", user_id))),
                "display_name": user.get("profile", {}).get("display_name", 
                              user.get("name", user_id)),
                "first_name": user.get("profile", {}).get("first_name", ""),
                "name": user.get("name", user_id),
                "avatar_hash": user.get("profile", {}).get("avatar_hash", ""),
                "image_72": user.get("profile", {}).get("image_72", ""),
                "team": user.get("team_id", ""),
                "is_restricted": user.get("is_restricted", False),
                "is_ultra_restricted": user.get("is_ultra_restricted", False)
            }
            user_profiles[user_id] = profile
    
    logger.debug(f"Built user profile mapping for {len(user_profiles)} users")
    
    # Enrich each message with user profile
    enriched_messages = []
    for message in messages:
        # Create a copy of the message
        enriched_message = message.copy()
        
        # Add user_profile if user exists in our mapping
        user_id = message.get("user")
        if user_id and user_id in user_profiles:
            enriched_message["user_profile"] = user_profiles[user_id]
        
        enriched_messages.append(enriched_message)
    
    return enriched_messages


def save_messages(messages: List[Dict[str, Any]], channel_name: str, 
                 workspace_name: str, users: List[Dict[str, Any]], 
                 logger: logging.Logger) -> int:
    if not messages:
        return 0
    
    # Enrich messages with user profile information
    enriched_messages = enrich_messages_with_user_profiles(messages, users, logger)
    
    data_base_path = "data/slack"
    messages_by_date = organize_messages_by_date(enriched_messages)
    total_saved = 0
    
    for date_str, date_messages in messages_by_date.items():
        base_dir = os.path.join(
            data_base_path, 
            workspace_name, 
            f"slack_to_{date_str}", 
            channel_name
        )
        os.makedirs(base_dir, exist_ok=True)
        
        out_path = os.path.join(base_dir, f"{date_str}.json")
        
        existing_messages = []
        if os.path.exists(out_path):
            try:
                with open(out_path, 'r') as f:
                    existing_messages = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        existing_ts = {msg.get("ts") for msg in existing_messages}
        new_messages = [msg for msg in date_messages if msg.get("ts") not in existing_ts]
        
        logger.debug(f"  {channel_name} {date_str}: {len(existing_messages)} existing, {len(date_messages)} fetched, {len(new_messages)} new")
        
        if new_messages:
            all_messages = existing_messages + new_messages
            all_messages.sort(key=lambda x: float(x.get("ts", 0)))
            
            try:
                with open(out_path, 'w') as f:
                    json.dump(all_messages, f, indent=2)
                total_saved += len(new_messages)
                logger.info(f"  Saved {len(new_messages)} new messages for {channel_name} on {date_str}")
            except IOError as e:
                logger.error(f"Failed to save to {out_path}: {e}")
        else:
            logger.debug(f"  No new messages for {channel_name} on {date_str} (all {len(date_messages)} already exist)")
    
    return total_saved


def save_metadata_files(users: List[Dict[str, Any]], channels: List[Dict[str, Any]], 
                       workspace_name: str, date_str: str, logger: logging.Logger) -> None:
    """Save users.json and channels.json files in index_slack.py compatible format."""
    data_base_path = "data/slack"
    export_dir = os.path.join(data_base_path, workspace_name, f"slack_to_{date_str}")
    os.makedirs(export_dir, exist_ok=True)
    
    # Save users.json - format expected by index_slack.py
    users_data = []
    for user in users:
        if not user.get("deleted", False) and user.get("id"):
            user_entry = {
                "id": user["id"],
                "name": user.get("name", user.get("id", "unknown")),
                "profile": {
                    "real_name": user.get("profile", {}).get("real_name", 
                                user.get("real_name", 
                                user.get("name", user.get("id", "Unknown User"))))
                }
            }
            users_data.append(user_entry)
    
    users_path = os.path.join(export_dir, "users.json")
    try:
        with open(users_path, 'w') as f:
            json.dump(users_data, f, indent=2)
        logger.info(f"Saved {len(users_data)} users to {users_path}")
    except IOError as e:
        logger.error(f"Failed to save users.json: {e}")
    
    # Save channels.json - format expected by index_slack.py
    channels_data = []
    for channel in channels:
        if channel.get("id") and channel.get("name"):
            channel_entry = {
                "id": channel["id"],
                "name": channel["name"]
            }
            channels_data.append(channel_entry)
    
    channels_path = os.path.join(export_dir, "channels.json")
    try:
        with open(channels_path, 'w') as f:
            json.dump(channels_data, f, indent=2)
        logger.info(f"Saved {len(channels_data)} channels to {channels_path}")
    except IOError as e:
        logger.error(f"Failed to save channels.json: {e}")


def main():
    try:
        logger = setup_logging()
        
        slack_bot_token = os.getenv("SCRAPING_SLACK_USER_TOKEN", "")
        if not slack_bot_token:
            logger.error("SCRAPING_SLACK_USER_TOKEN not found in environment variables")
            return 1
        
        client = WebClient(token=slack_bot_token)
        
        workspace_name = get_workspace_info(client)
        
        # Get today's date range (00:00:00 to 23:59:59)
        now = datetime.now()
        start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        oldest_ts = start_of_today.timestamp()
        date_str = now.strftime('%Y-%m-%d')
        
        logger.info(f"Scraping messages for {date_str} (since {start_of_today.strftime('%H:%M:%S')})")
        
        # Fetch users and channels metadata
        logger.info("Fetching workspace metadata...")
        users = get_all_users(client, logger)
        
        slack_channels = os.getenv("SLACK_CHANNELS")
        if slack_channels:
            channel_names = [name.strip() for name in slack_channels.split(",")]
            channels = get_specific_channels(channel_names)
        else:
            channels = get_all_channels(client, logger)
        
        if not channels:
            logger.warning("No channels found")
            return 0
        
        # Save metadata files in index_slack.py compatible format
        save_metadata_files(users, channels, workspace_name, date_str, logger)
        
        total_saved = 0
        for i, channel in enumerate(channels, 1):
            logger.info(f"[{i}/{len(channels)}] {channel['name']}")
            
            messages = fetch_channel_messages(
                client, channel["id"], channel["name"], oldest_ts, logger
            )
            
            logger.debug(f"  Fetched {len(messages)} messages from {channel['name']}")
            
            if messages:
                saved = save_messages(messages, channel["name"], workspace_name, users, logger)
                total_saved += saved
                logger.info(f"  Saved {saved} messages")
            else:
                logger.info(f"  No messages found")
            
            time.sleep(1)
        
        # Note: Using date-based collection, no need to track last run timestamp
        logger.info(f"Completed - saved {total_saved} total messages")
        
    except Exception as e:
        logging.error(f"Scraper failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())