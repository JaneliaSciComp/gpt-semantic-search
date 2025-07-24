#!/usr/bin/env python3
"""
Filesystem RAG Interactive CLI Entry Point

Enhanced interactive command-line interface for filesystem RAG with beautiful,
modern terminal output, search-first interface, and comprehensive file monitoring.

Usage:
    python filesystem_rag_cli.py [options]

🔍 Search Mode (Default):
    Just type your query naturally - no commands needed!
    
📋 Interactive Commands:
    /add <directory> [--class-prefix <prefix>]  Add directory to monitor
    /remove <directory>                         Remove directory from monitoring
    /list                                       Show monitored directories
    /search <query>                             Explicit search (same as natural input)
    /status                                     Show system status
    /settings [key] [value]                     View/update settings
    /restart [directory]                        Restart monitoring
    /index <directory> [--remove-existing]     Force re-indexing
    /help                                       Show all commands
    /exit                                       Exit gracefully
    
⚡ Shortcuts (can be used without /):
    help, status, list, exit, quit

✨ Enhanced Features:
    - Beautiful Rich library interface with colors and formatting
    - Search-first workflow - no need for /search command
    - Smart command detection and shortcuts
    - Visual status indicators and enhanced prompts
    - Persistent configuration across sessions
    - Real-time filesystem monitoring with auto-indexing
    - Multi-directory concurrent monitoring
    - Semantic search across all indexed content
    - Comprehensive file type support (59+ types via Docling)
    - Background processing with status tracking
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cli.interactive_session import InteractiveSession

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """Main entry point for the interactive filesystem RAG CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Filesystem RAG Interactive CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🔍 Interactive Filesystem RAG CLI

This CLI provides an interactive interface for managing and searching
filesystem content using semantic RAG (Retrieval-Augmented Generation).

Key Features:
• Persistent directory monitoring across sessions
• Real-time file indexing with comprehensive format support
• Semantic search across all monitored directories
• Background processing with status tracking
• Automatic configuration management

Interactive Commands:
  /add <directory> [--class-prefix <prefix>]  Add directory to monitor
  /remove <directory>                         Remove directory from monitoring
  /list                                       Show monitored directories
  /search <query>                             Search indexed content
  /status                                     Show system status
  /settings [key] [value]                     View/update settings
  /restart [directory]                        Restart monitoring
  /index <directory> [--remove-existing]     Force re-indexing
  /help                                       Show all commands
  /exit                                       Exit gracefully

Quick Start:
  1. Start the CLI: python filesystem_rag_cli.py
  2. Add a directory: /add /path/to/documents --class-prefix MyDocs
  3. Search naturally: how to configure system
  4. View status: status (or /status)
  5. Exit: exit (or /exit)

Configuration:
  Configuration is automatically saved to ~/.filesystem_rag_config.json
  and restored when you restart the CLI.

Examples:
  # Basic usage
  python filesystem_rag_cli.py
  
  # Debug mode
  python filesystem_rag_cli.py --debug
  
  # Custom configuration file
  python filesystem_rag_cli.py --config ./my_config.json
  
  # Custom Weaviate URL
  python filesystem_rag_cli.py --weaviate-url http://localhost:8777
        """)
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Configuration file path (default: ~/.filesystem_rag_config.json)'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Enable debug logging for detailed operation tracing'
    )
    parser.add_argument(
        '-w', '--weaviate-url',
        type=str,
        help='Weaviate database URL (default: http://localhost:8777)'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='Filesystem RAG CLI v1.0.0'
    )
    
    args = parser.parse_args()
    
    # Validate environment
    try:
        import weaviate
        import openai
        from llama_index.core import Settings
    except ImportError as e:
        print(f"Error: Missing required dependencies: {e}")
        print("Please install required packages:")
        print("  pixi install")
        print("  # or manually install: weaviate-client, openai, llama-index")
        sys.exit(1)
    
    # Check for OpenAI API key
    import os
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Some features may not work without a valid OpenAI API key.")
        print()
    
    # Create and configure session
    try:
        session = InteractiveSession(
            config_path=args.config,
            debug=args.debug
        )
        
        # Override Weaviate URL if provided
        if args.weaviate_url:
            session.config.weaviate_url = args.weaviate_url
        
        # Start the interactive session
        await session.start()
        
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start interactive session: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}")
            print("Run with --debug for detailed error information")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)