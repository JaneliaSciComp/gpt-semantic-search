#!/usr/bin/env python3
"""Slack Incremental Scraper - Flexible interval message collection with built-in status management."""

import os
import sys
import json
import time
import logging
import argparse
import shutil
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict
from pathlib import Path
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def setup_logging() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"slack_scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("slack_scraper")


def find_latest_successful_run(data_path: str = "data/slack") -> Optional[float]:
    """Find the most recent successful run timestamp."""
    if not os.path.exists(data_path):
        return None
    
    latest_timestamp = 0.0
    
    for workspace_dir in os.listdir(data_path):
        workspace_path = os.path.join(data_path, workspace_dir)
        if not os.path.isdir(workspace_path):
            continue
        
        for folder in os.listdir(workspace_path):
            if folder.startswith("success_run_"):
                try:
                    timestamp = float(folder.split("_")[-1])
                    latest_timestamp = max(latest_timestamp, timestamp)
                except ValueError:
                    continue
    
    return latest_timestamp if latest_timestamp > 0 else None


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
                limit=999, # This limit includes archived channels--incrased to max limit to avoid hitting 200 total channels
                cursor=cursor
            )
            channels.extend(resp["channels"])
            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        
        logger.info(f"Found {len(channels)} channels")
        return channels
        
    except SlackApiError as e:
        logger.error(f"Error fetching channels: {e}")
        return []


def fetch_channel_messages(client: WebClient, channel_id: str, channel_name: str, 
                         oldest_ts: float, logger: logging.Logger) -> List[Dict[str, Any]]:
    all_messages = []
    cursor = None
    threaded_messages = set()  # Track which messages we've already processed as threads
    
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
            
            result = client.conversations_history(**params)
            messages = result["messages"]
            
            if not messages:
                break
            
            for message in messages:
                msg_ts = message.get("ts")
                thread_ts = message.get("thread_ts")
                reply_count = message.get("reply_count", 0)
                
                # If this is a threaded message with replies, get the full thread
                if reply_count > 0 and msg_ts not in threaded_messages:

                    thread_messages = fetch_complete_thread(client, channel_id, msg_ts, logger)
                    all_messages.extend(thread_messages)
                    # Mark all messages in this thread as processed
                    for thread_msg in thread_messages:
                        threaded_messages.add(thread_msg.get("ts"))
                elif msg_ts not in threaded_messages:
                    # This is a regular message (not part of a thread we've already processed)
                    all_messages.append(message)
            
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
    
    return all_messages


def fetch_complete_thread(client: WebClient, channel_id: str, thread_ts: str, 
                         logger: logging.Logger) -> List[Dict[str, Any]]:
    """Fetch complete thread including parent message and all replies using conversations.replies."""
    all_thread_messages = []
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
            
            # Use all messages from conversations.replies - includes parent + replies
            all_thread_messages.extend(messages)
            
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
                logger.warning(f"Error fetching complete thread for {thread_ts}: {e}")
                break
    
    return all_thread_messages


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
                                      users: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enrich messages with user profile information from users list."""
    # Build user ID to profile mapping
    # Emulate block syntax from slack web export
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
    
    # Enrich each message with user profile
    enriched_messages = []
    for message in messages:
        enriched_message = message.copy()
        user_id = message.get("user")
        if user_id and user_id in user_profiles:
            enriched_message["user_profile"] = user_profiles[user_id]
        enriched_messages.append(enriched_message)
    
    return enriched_messages


def save_messages(messages: List[Dict[str, Any]], channel_name: str, 
                 workspace_name: str, run_timestamp: int, users: List[Dict[str, Any]], 
                 logger: logging.Logger) -> int:
    if not messages:
        return 0
    
    # Enrich messages with user profile information
    enriched_messages = enrich_messages_with_user_profiles(messages, users)
    
    data_base_path = "data/slack"
    messages_by_date = organize_messages_by_date(enriched_messages)
    total_saved = 0
    
    for date_str, date_messages in messages_by_date.items():
        base_dir = os.path.join(
            data_base_path, 
            workspace_name, 
            f"run_{run_timestamp}", 
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
        
        if new_messages:
            all_messages = existing_messages + new_messages
            all_messages.sort(key=lambda x: float(x.get("ts", 0)))
            
            try:
                with open(out_path, 'w') as f:
                    json.dump(all_messages, f, indent=2)
                total_saved += len(new_messages)
            except IOError as e:
                logger.error(f"Failed to save to {out_path}: {e}")
    
    return total_saved


def save_metadata_files(users: List[Dict[str, Any]], channels: List[Dict[str, Any]], 
                       workspace_name: str, run_timestamp: int, logger: logging.Logger) -> None:
    """Save users.json and channels.json files."""
    data_base_path = "data/slack"
    export_dir = os.path.join(data_base_path, workspace_name, f"run_{run_timestamp}")
    os.makedirs(export_dir, exist_ok=True)
    
    # Save users.json
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
    
    try:
        with open(os.path.join(export_dir, "users.json"), 'w') as f:
            json.dump(users_data, f, indent=2)
    except IOError as e:
        logger.error(f"Failed to save users.json: {e}")
    
    # Save channels.json
    channels_data = []
    for channel in channels:
        if channel.get("id") and channel.get("name"):
            channel_entry = {
                "id": channel["id"],
                "name": channel["name"]
            }
            channels_data.append(channel_entry)
    
    try:
        with open(os.path.join(export_dir, "channels.json"), 'w') as f:
            json.dump(channels_data, f, indent=2)
    except IOError as e:
        logger.error(f"Failed to save channels.json: {e}")


def validate_run_success(workspace_name: str, run_timestamp: int) -> bool:
    """Validate that the scraping run completed successfully."""
    folder_path = Path(f"data/slack/{workspace_name}/run_{run_timestamp}")
    
    if not folder_path.exists():
        return False
    
    # Check for required metadata files
    users_file = folder_path / "users.json"
    channels_file = folder_path / "channels.json"
    
    if not users_file.exists() or not channels_file.exists():
        return False
    
    # Validate JSON files
    try:
        with open(users_file, 'r') as f:
            users_data = json.load(f)
            if not isinstance(users_data, list):
                return False
        
        with open(channels_file, 'r') as f:
            channels_data = json.load(f)
            if not isinstance(channels_data, list):
                return False
                
    except (json.JSONDecodeError, IOError):
        return False
    
    return True


def mark_run_status(workspace_name: str, run_timestamp: int, status: str, logger: logging.Logger) -> bool:
    """Mark the run with success or failed status."""
    current_folder = f"run_{run_timestamp}"
    current_path = Path(f"data/slack/{workspace_name}/{current_folder}")
    
    if not current_path.exists():
        return False
    
    if status == "success":
        new_folder = f"success_run_{run_timestamp}"
    elif status == "failed":
        new_folder = f"failed_run_{run_timestamp}"
    else:
        return False
    
    new_path = Path(f"data/slack/{workspace_name}/{new_folder}")
    
    try:
        current_path.rename(new_path)
        logger.info(f"Marked run as {status}: {current_folder} -> {new_folder}")
        return True
    except OSError as e:
        logger.error(f"Failed to mark run status: {e}")
        return False


def cleanup_failed_run(workspace_name: str, run_timestamp: int, logger: logging.Logger) -> None:
    """Delete a failed run folder entirely."""
    folder_path = Path(f"data/slack/{workspace_name}/run_{run_timestamp}")
    
    if folder_path.exists():
        try:
            shutil.rmtree(folder_path)
            logger.info(f"Cleaned up failed run: run_{run_timestamp}")
        except OSError as e:
            logger.error(f"Failed to cleanup run: {e}")


def main():
    parser = argparse.ArgumentParser(description='Slack Incremental Scraper - automatically continues from last successful run')
    parser.add_argument('--start-from', type=float,
                       help='Unix timestamp to start scraping from (overrides auto-detection)')
    parser.add_argument('--end-at', type=float,
                       help='Unix timestamp to end scraping at (default: now)')
    parser.add_argument('-d', '--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    run_timestamp = int(datetime.now().timestamp())
    
    try:
        slack_bot_token = os.getenv("SCRAPING_SLACK_USER_TOKEN", "")
        if not slack_bot_token:
            logger.error("SCRAPING_SLACK_USER_TOKEN not found in environment variables")
            return 1
        
        client = WebClient(token=slack_bot_token)
        workspace_name = get_workspace_info(client)
        
        if args.start_from:
            start_timestamp = args.start_from
            logger.info(f"Using custom start timestamp: {datetime.fromtimestamp(start_timestamp)}")
        else:
            # Find the most recent successful run to continue from
            latest_timestamp = find_latest_successful_run()
            if latest_timestamp:
                start_timestamp = latest_timestamp
                logger.info(f"Auto-continuing from last successful run: {datetime.fromtimestamp(latest_timestamp)}")
            else:
                # No previous runs, start from 24 hours ago
                start_timestamp = (datetime.now() - timedelta(days=1)).timestamp()
                logger.info(f"No previous runs found, starting from 24 hours ago: {datetime.fromtimestamp(start_timestamp)}")
        
        # Calculate scraping window - from start_timestamp to now
        oldest_ts = start_timestamp
        newest_ts = args.end_at if args.end_at else datetime.now().timestamp()
        
        logger.info(f"Scraping window: {datetime.fromtimestamp(oldest_ts)} to {datetime.fromtimestamp(newest_ts)}")
        logger.info(f"Starting scraping run: run_{run_timestamp}")
        logger.info("Fetching workspace metadata...")
        users = get_all_users(client, logger)
        
        channels = get_all_channels(client, logger)
        
        if not channels:
            logger.warning("No channels found")
            # Even with no channels, if we got this far without API errors, it's a success
            save_metadata_files([], [], workspace_name, run_timestamp, logger)
            
            if validate_run_success(workspace_name, run_timestamp):
                mark_run_status(workspace_name, run_timestamp, "success", logger)
                logger.info("Marked as successful (no channels found, but no errors)")
                return 0
            else:
                cleanup_failed_run(workspace_name, run_timestamp, logger)
                logger.error("Validation failed")
                return 1
        

        save_metadata_files(users, channels, workspace_name, run_timestamp, logger)
        
        total_saved = 0
        for i, channel in enumerate(channels, 1):
            logger.info(f"[{i}/{len(channels)}] {channel['name']}")
            
            messages = fetch_channel_messages(
                client, channel["id"], channel["name"], oldest_ts, logger
            )
            
            if messages:
                saved = save_messages(messages, channel["name"], workspace_name, run_timestamp, users, logger)
                total_saved += saved
                logger.info(f"  Saved {saved} messages")
            else:
                logger.info(f"  No messages found")
            
            time.sleep(1)
        
        logger.info(f"Completed scraping - saved {total_saved} total messages")
        
        # Validate and mark folder status
        if validate_run_success(workspace_name, run_timestamp):
            mark_run_status(workspace_name, run_timestamp, "success", logger)
            logger.info("Scraping completed successfully")
            return 0
        else:
            cleanup_failed_run(workspace_name, run_timestamp, logger)
            logger.error("Scraping validation failed")
            return 1
        
    except Exception as e:
        logger.error(f"Scraper failed with exception: {e}")
        # Cleanup failed folder on exception
        try:
            cleanup_failed_run(workspace_name, run_timestamp, logger)
            logger.info("Cleaned up failed folder due to exception")
        except:
            pass
        return 1


if __name__ == "__main__":
    exit(main())