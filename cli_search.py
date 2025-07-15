#!/usr/bin/env python

import os
import re
import sys
import argparse
import textwrap
import logging
import warnings
from typing import Dict, List
import time
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core import PromptHelper, GPTVectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

import weaviate
from slack_sdk import WebClient
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich import print as rprint
import readline

# Initialize console
console = Console()

# Constants
EMBED_MODEL_NAME = "text-embedding-3-large"
CONTEXT_WINDOW = 4096
NUM_OUTPUT = 256
CHUNK_OVERLAP_RATIO = 0.1
SURVEY_CLASS = "SurveyResponses"

# Configure logging
logging.basicConfig(level=logging.WARNING)
warnings.simplefilter("ignore", ResourceWarning)

logger = logging.getLogger(__name__)

class CLISearchApp:
    def __init__(self, weaviate_url: str = "http://localhost:8080"):
        self.weaviate_url = weaviate_url
        self.weaviate_client = None
        self.slack_client = None
        self.query_engine = None
        
        # Default parameters
        self.config = {
            "model": "gpt-4o",
            "class_prefix": "Janelia",
            "temperature": 0.0,
            "search_alpha": 0.55,
            "num_results": 3,
            "hyde_enabled": False
        }
        
        # Slash commands
        self.slash_commands = [
            "/config", "/set", "/help", "/quit", "/exit",
            "/model", "/temperature", "/alpha", "/results", "/hyde"
        ]
        
        self._init_clients()
        self._setup_readline()
    
    def _init_clients(self):
        """Initialize Weaviate and Slack clients"""
        try:
            self.weaviate_client = weaviate.Client(self.weaviate_url)
            if not self.weaviate_client.is_live():
                raise Exception(f"Weaviate is not live at {self.weaviate_url}")
            
            # Initialize Slack client if token is available
            slack_token = os.environ.get('SLACK_TOKEN')
            if slack_token:
                self.slack_client = WebClient(token=slack_token)
                res = self.slack_client.api_test()
                if not res["ok"]:
                    console.print(f"[yellow]Warning: Slack API error: {res['error']}[/yellow]")
                    self.slack_client = None
            else:
                console.print("[yellow]Warning: SLACK_TOKEN not found. Slack links will not be available.[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Error initializing clients: {e}[/red]")
            sys.exit(1)
    
    def _setup_readline(self):
        """Setup readline for command completion"""
        def completer(text, state):
            if text.startswith('/'):
                matches = [cmd for cmd in self.slash_commands if cmd.startswith(text)]
                try:
                    return matches[state]
                except IndexError:
                    return None
            return None
        
        readline.set_completer(completer)
        readline.parse_and_bind("tab: complete")
        readline.set_completer_delims(' \t\n')
    
    def _get_input_with_completion(self, prompt_text: str = "Search") -> str:
        """Get user input with slash command completion"""
        try:
            # Display the prompt using rich styling
            console.print(f"[bold cyan]{prompt_text}[/bold cyan]", end="")
            
            # Get input with readline completion
            user_input = input(": ")
            return user_input
            
        except (EOFError, KeyboardInterrupt):
            return "/quit"
    
    def _get_query_engine(self):
        """Create and configure the query engine"""
        llm = OpenAI(model=self.config["model"], temperature=self.config["temperature"])
        embed_model = OpenAIEmbedding(model=EMBED_MODEL_NAME)
        prompt_helper = PromptHelper(CONTEXT_WINDOW, NUM_OUTPUT, CHUNK_OVERLAP_RATIO)

        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 512
        Settings.prompt_helper = prompt_helper

        vector_store = WeaviateVectorStore(
            weaviate_client=self.weaviate_client, 
            class_prefix=self.config["class_prefix"]
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = GPTVectorStoreIndex([], storage_context=storage_context)

        # Configure retriever
        retriever = VectorIndexRetriever(
            index,
            similarity_top_k=self.config["num_results"],
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            alpha=self.config["search_alpha"],
        )

        # Construct query engine
        query_engine = RetrieverQueryEngine.from_args(retriever)
        
        # Apply HyDE transformation if enabled
        if self.config["hyde_enabled"]:
            try:
                hyde_transform = HyDEQueryTransform(include_original=True)
                query_engine = TransformQueryEngine(query_engine, hyde_transform)
                console.print("[green]✓ HyDE query transformation applied[/green]")
            except Exception as e:
                console.print(f"[yellow]✗ HyDE transformation failed: {e}[/yellow]")
        
        return query_engine
    
    def _get_unique_nodes(self, nodes):
        """Filter out duplicate nodes based on document ID"""
        docs_ids = set()
        unique_nodes = list()
        for node in nodes:
            if node.node.ref_doc_id not in docs_ids:
                docs_ids.add(node.node.ref_doc_id)
                unique_nodes.append(node)
        return unique_nodes
    
    def _get_message_link(self, channel, ts):
        """Get Slack message permalink"""
        if not self.slack_client:
            return None
        
        try:
            res = self.slack_client.chat_getPermalink(channel=channel, message_ts=ts)
            if res['ok']:
                return res['permalink']
        except Exception as e:
            logger.warning(f"Error getting Slack permalink: {e}")
        return None
    
    def _format_sources(self, source_nodes):
        """Format source nodes for display"""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Source", style="cyan")
        table.add_column("Content", style="white")
        table.add_column("Link", style="blue")
        
        for node in self._get_unique_nodes(source_nodes):
            extra_info = node.node.extra_info
            text = node.node.text
            
            # Clean up text
            text = re.sub("\n+", " ", text)
            text = textwrap.shorten(text, width=80, placeholder="...")
            
            source = extra_info['source']
            link = "N/A"
            
            if source.lower() == 'slack':
                channel_id = extra_info['channel']
                ts = extra_info['ts']
                permalink = self._get_message_link(channel_id, ts)
                link = permalink if permalink else "N/A"
            else:
                link = extra_info.get('link', 'N/A')
                
            table.add_row(source, text, link)
        
        return table
    
    def search(self, query: str):
        """Perform search and return formatted results"""
        if not query.strip():
            console.print("[yellow]Please enter a search query[/yellow]")
            return
        
        # Clean query
        query = re.sub('"', '', query)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Searching...", total=None)
            
            try:
                start_time = time.time()
                
                # Get query engine
                query_engine = self._get_query_engine()
                
                # Execute query
                response = query_engine.query(query)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                progress.update(task, completed=True)
                
                # Display results
                console.print("\n" + "="*80)
                console.print(Panel(
                    f"[bold]Query:[/bold] {query}", 
                    title="Search Results",
                    title_align="left"
                ))
                
                # Display response
                console.print(Panel(
                    Markdown(response.response),
                    title="Answer",
                    title_align="left",
                    border_style="green"
                ))
                
                # Display sources
                if response.source_nodes:
                    console.print("\n[bold]Sources:[/bold]")
                    console.print(self._format_sources(response.source_nodes))
                
                # Display metadata
                console.print(f"\n[dim]Response time: {response_time:.2f}s | Model: {self.config['model']} | HyDE: {self.config['hyde_enabled']}[/dim]")
                
            except Exception as e:
                progress.update(task, completed=True)
                console.print(f"[red]Error: {e}[/red]")
                logger.exception("Search error")
    
    def interactive_mode(self):
        """Run interactive search mode"""
        console.print(Panel(
            "[bold]JaneliaGPT CLI Search[/bold]\n\n"
            "Available commands:\n"
            "• Type your search query and press Enter\n"
            "• /config - Show current configuration\n"
            "• /set <param> <value> - Set configuration parameter\n"
            "• /help - Show this help message\n"
            "• /quit - Exit the application\n\n"
            "[dim]💡 Tip: Type '/' and press TAB to see available slash commands[/dim]",
            title="Welcome",
            title_align="left"
        ))
        
        while True:
            try:
                console.print()  # Add newline before prompt
                query = self._get_input_with_completion()
                
                if query.lower() in ['/quit', '/exit', 'quit', 'exit']:
                    console.print("[green]Goodbye![/green]")
                    break
                elif query.lower() == '/help':
                    self._show_help()
                elif query.lower() == '/config':
                    self._show_config()
                elif query.lower().startswith('/set'):
                    self._handle_set_command(query)
                else:
                    self.search(query)
                    
            except KeyboardInterrupt:
                console.print("\n[green]Goodbye![/green]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    def _show_help(self):
        """Show help information"""
        console.print(Panel(
            "[bold]Available Commands:[/bold]\n\n"
            "• Search: Type any query to search the knowledge base\n"
            "• /config: Display current configuration\n"
            "• /set model <model_name>: Set the OpenAI model\n"
            "• /set temperature <0.0-2.0>: Set response creativity\n"
            "• /set alpha <0.0-1.0>: Set search alpha (0=semantic, 1=keyword)\n"
            "• /set results <number>: Set number of results to return\n"
            "• /set hyde <true/false>: Enable/disable HyDE transformation\n"
            "• /help: Show this help message\n"
            "• /quit: Exit the application",
            title="Help",
            title_align="left"
        ))
    
    def _show_config(self):
        """Show current configuration"""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="white")
        
        for key, value in self.config.items():
            table.add_row(key, str(value))
        
        console.print(Panel(table, title="Current Configuration", title_align="left"))
    
    def _handle_set_command(self, command):
        """Handle configuration set commands"""
        parts = command.split()
        if len(parts) != 3:
            console.print("[red]Usage: /set <parameter> <value>[/red]")
            return
        
        _, param, value = parts
        
        if param not in self.config:
            console.print(f"[red]Unknown parameter: {param}[/red]")
            return
        
        try:
            # Type conversion
            if param == "temperature" or param == "search_alpha":
                value = float(value)
            elif param == "num_results":
                value = int(value)
            elif param == "hyde_enabled":
                value = value.lower() in ['true', '1', 'yes', 'on']
            
            self.config[param] = value
            console.print(f"[green]Set {param} = {value}[/green]")
            
        except ValueError:
            console.print(f"[red]Invalid value for {param}: {value}[/red]")


def main():
    parser = argparse.ArgumentParser(description='CLI for JaneliaGPT semantic search')
    parser.add_argument('-w', '--weaviate-url', type=str, default="http://localhost:8777",
                        help='Weaviate database URL')
    parser.add_argument('-q', '--query', type=str, help='Search query (non-interactive mode)')
    parser.add_argument('-m', '--model', type=str, default="gpt-4o", help='OpenAI model to use')
    parser.add_argument('-t', '--temperature', type=float, default=0.0, help='Response temperature')
    parser.add_argument('-a', '--alpha', type=float, default=0.55, help='Search alpha (0=semantic, 1=keyword)')
    parser.add_argument('-r', '--results', type=int, default=3, help='Number of results to return')
    parser.add_argument('--hyde', action='store_true', help='Enable HyDE query transformation')
    parser.add_argument('--class-prefix', type=str, default="Janelia", help='Weaviate class prefix')
    
    args = parser.parse_args()
    
    # Create search app
    app = CLISearchApp(args.weaviate_url)
    
    # Update configuration with command line arguments
    app.config.update({
        "model": args.model,
        "temperature": args.temperature,
        "search_alpha": args.alpha,
        "num_results": args.results,
        "hyde_enabled": args.hyde,
        "class_prefix": args.class_prefix
    })
    
    if args.query:
        # Non-interactive mode
        app.search(args.query)
    else:
        # Interactive mode
        app.interactive_mode()


if __name__ == "__main__":
    main()