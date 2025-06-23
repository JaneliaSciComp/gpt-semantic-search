#!/usr/bin/env python3
"""Daily Slack Message Scraper - Fast message collection to file system."""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
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


def get_last_run_timestamp() -> float:
    state_path = os.path.join("logs", "last_scrape_timestamp.txt")
    
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r') as f:
                return float(f.read().strip())
        except (ValueError, IOError):
            return time.time() - 86400
    else:
        return time.time() - 86400


def update_last_run_timestamp(timestamp: float) -> None:
    os.makedirs("logs", exist_ok=True)
    state_path = os.path.join("logs", "last_scrape_timestamp.txt")
    
    try:
        with open(state_path, 'w') as f:
            f.write(str(timestamp))
    except IOError as e:
        logging.error(f"Failed to update state: {e}")


def get_workspace_info(client: WebClient) -> str:
    try:
        auth_resp = client.auth_test()
        workspace_name = auth_resp.get("team", "unknown-workspace")
        return workspace_name.lower().replace(" ", "-")
    except Exception:
        return "unknown-workspace"


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
    slack_data_path = "../data/slack"
    
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
                "inclusive": True
            }
            if cursor:
                params["cursor"] = cursor
            
            logger.debug(f"  API call params: {params}")
            result = client.conversations_history(**params)
            messages = result["messages"]
            
            logger.debug(f"  API returned {len(messages)} messages")
            
            if not messages:
                break
            
            all_messages.extend(messages)
            
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


def save_messages(messages: List[Dict[str, Any]], channel_name: str, 
                 workspace_name: str, logger: logging.Logger) -> int:
    if not messages:
        return 0
    
    data_base_path = "../data/slack"
    messages_by_date = organize_messages_by_date(messages)
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


def main():
    try:
        logger = setup_logging()
        
        slack_bot_token = os.getenv("SCRAPING_SLACK_BOT_TOKEN", "")
        if not slack_bot_token:
            logger.error("SCRAPING_SLACK_BOT_TOKEN not found in environment variables")
            return 1
        
        client = WebClient(token=slack_bot_token)
        
        workspace_name = get_workspace_info(client)
        
        # Get today's date range (00:00:00 to 23:59:59)
        now = datetime.now()
        start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        oldest_ts = start_of_today.timestamp()
        
        logger.info(f"Scraping messages for {now.strftime('%Y-%m-%d')} (since {start_of_today.strftime('%H:%M:%S')})")
        
        slack_channels = os.getenv("SLACK_CHANNELS")
        if slack_channels:
            channel_names = [name.strip() for name in slack_channels.split(",")]
            channels = get_specific_channels(channel_names)
        else:
            channels = get_all_channels(client, logger)
        
        if not channels:
            logger.warning("No channels found")
            return 0
        
        total_saved = 0
        for i, channel in enumerate(channels, 1):
            logger.info(f"[{i}/{len(channels)}] {channel['name']}")
            
            messages = fetch_channel_messages(
                client, channel["id"], channel["name"], oldest_ts, logger
            )
            
            logger.debug(f"  Fetched {len(messages)} messages from {channel['name']}")
            
            if messages:
                saved = save_messages(messages, channel["name"], workspace_name, logger)
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