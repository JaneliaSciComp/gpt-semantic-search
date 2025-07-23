#!/usr/bin/env python3
"""
Comprehensive Integration Tests for Filesystem RAG Pipeline

This module provides end-to-end integration tests for the filesystem RAG service,
following the patterns from the existing web service test suite.

Tests cover:
1. File processing and indexing pipeline
2. Search accuracy and retrieval quality
3. Real-time monitoring functionality
4. Multiple file type support
5. Performance benchmarking
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import weaviate
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from filesystem_rag_service import FilesystemRAGService
from scraping.filesystem.universal_scraper import UniversalFileScraper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
DEFAULT_WEAVIATE_URL = "http://localhost:8777"
TEST_CLASS_PREFIX = "FilesystemTest"
TEST_TIMEOUT = 60  # seconds


class FilesystemRAGTester:
    """
    Test suite for filesystem RAG integration testing.
    
    Provides comprehensive testing of the filesystem RAG pipeline including:
    - Document indexing accuracy
    - Search quality validation
    - Real-time monitoring tests
    - Performance benchmarks
    """
    
    def __init__(self, weaviate_url: str = DEFAULT_WEAVIATE_URL, debug: bool = False):
        self.weaviate_url = weaviate_url
        self.debug = debug
        self.service = FilesystemRAGService(weaviate_url, debug)
        self.test_data_dir = None
        self.scraper = UniversalFileScraper(debug)
        
        logger.info(f"Initialized Filesystem RAG Tester with Weaviate at {weaviate_url}")
    
    def setup_test_data(self) -> Path:
        """Create comprehensive test data directory with various file types."""
        if self.test_data_dir:
            return self.test_data_dir
            
        self.test_data_dir = Path(tempfile.mkdtemp(prefix="filesystem_rag_test_"))
        logger.info(f"Created test data directory: {self.test_data_dir}")
        
        # Create test documents with known content for search validation
        test_files = {
            # Python code file
            "example_script.py": '''#!/usr/bin/env python3
"""
Example Python script for testing filesystem RAG capabilities.

This module demonstrates various Python features and patterns that should
be searchable through the RAG system.
"""

import os
import sys
from pathlib import Path

class DataProcessor:
    """Process data files with various formats."""
    
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.processed_count = 0
    
    def process_csv_file(self, filename: str) -> dict:
        """Process CSV files and return statistics."""
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            return {
                'filename': filename,
                'line_count': len(lines),
                'status': 'processed successfully'
            }
        except Exception as e:
            return {'filename': filename, 'error': str(e)}
    
    def batch_process(self) -> list:
        """Process all files in the input directory."""
        results = []
        for file_path in self.input_path.glob("*.csv"):
            result = self.process_csv_file(str(file_path))
            results.append(result)
            self.processed_count += 1
        
        return results

def main():
    """Main function for data processing workflow."""
    processor = DataProcessor("/data/input", "/data/output")
    results = processor.batch_process()
    print(f"Processed {len(results)} files")

if __name__ == "__main__":
    main()
''',
            
            # Markdown documentation
            "project_documentation.md": '''# Filesystem RAG System Documentation

## Overview

The Filesystem RAG (Retrieval-Augmented Generation) system provides real-time indexing and search capabilities for local file systems. It combines advanced file processing with vector search and large language models.

## Key Features

- **Multi-format Support**: Processes 59+ file types including:
  - Documents (PDF, Word, PowerPoint)
  - Code files (Python, Java, JavaScript, etc.)
  - Text formats (Markdown, JSON, XML)
  - Jupyter notebooks
  - Images with OCR capabilities

- **Real-time Monitoring**: Uses watchfiles for high-performance file system monitoring
- **Vector Search**: Integrates with Weaviate for semantic search capabilities
- **CLI Interface**: Command-line tools for indexing, searching, and monitoring

## Architecture

The system consists of several key components:

1. **File Processor**: Extracts text from various file formats
2. **Filesystem Monitor**: Watches directories for changes
3. **Vector Indexer**: Creates embeddings and stores in Weaviate
4. **Query Engine**: Provides hybrid search (keyword + vector)
5. **CLI Interface**: User-facing commands for all operations

## Configuration

Key environment variables:
- `WEAVIATE_URL`: Vector database connection
- `OPENAI_API_KEY`: For embeddings and LLM
- `CLASS_PREFIX`: Weaviate class organization
- `FILESYSTEM_PATH`: Base path for monitoring

## Usage Examples

### Index a Directory
```bash
python filesystem_rag_service.py index /path/to/docs --class-prefix MyDocs
```

### Search Documents
```bash
python filesystem_rag_service.py search "How do I configure the system?"
```

### Monitor for Changes
```bash
python filesystem_rag_service.py monitor /path/to/docs --class-prefix MyDocs
```

## Testing

The system includes comprehensive integration tests that validate:
- File processing accuracy across all supported formats
- Search quality using DeepEval metrics
- Real-time monitoring functionality
- Performance benchmarks for large document sets

## Performance Considerations

- File processing speed varies by format (text files fastest, PDFs slower)
- Vector embedding generation is the primary latency factor
- Weaviate performance scales with document volume
- Real-time monitoring has minimal overhead with watchfiles

## Troubleshooting

Common issues and solutions:
- **Weaviate Connection**: Ensure database is running and accessible
- **File Permissions**: Verify read access to monitored directories
- **Memory Usage**: Large documents may require memory optimization
- **Processing Errors**: Check logs for file-specific issues

For support, contact the development team or check the project documentation.
''',
            
            # JSON configuration
            "system_config.json": json.dumps({
                "application": {
                    "name": "Filesystem RAG Service",
                    "version": "1.0.0",
                    "description": "CLI service for filesystem-based RAG operations"
                },
                "processing": {
                    "supported_extensions": [".py", ".md", ".json", ".txt", ".pdf", ".docx"],
                    "methods": ["unstructured", "native", "jupyter"],
                    "max_file_size": "100MB",
                    "timeout_seconds": 30
                },
                "search": {
                    "embedding_model": "text-embedding-3-large",
                    "temperature": 0.1,
                    "search_alpha": 0.5,
                    "max_results": 10
                },
                "monitoring": {
                    "debounce_ms": 1000,
                    "recursive": True,
                    "include_hidden": False
                },
                "weaviate": {
                    "default_url": "http://localhost:8777",
                    "default_class_prefix": "Filesystem",
                    "schema_version": "1.0"
                }
            }, indent=2),
            
            # Plain text file
            "readme.txt": '''Filesystem RAG Service - Quick Start Guide

This service provides comprehensive filesystem indexing and search capabilities.
It can process dozens of file formats and provides real-time monitoring.

Key capabilities:
- Index entire directories recursively
- Search using natural language queries
- Monitor for file changes in real-time
- Support for code, documents, and data files

The system uses Weaviate as the vector database and OpenAI for embeddings.
All operations are available through a command-line interface.

For detailed documentation, see project_documentation.md.
For configuration options, check system_config.json.
For code examples, review example_script.py.

Contact the development team for support and feature requests.
''',
            
            # CSV data file
            "sample_data.csv": '''name,role,department,email
Alice Smith,Software Engineer,Engineering,alice@example.com
Bob Johnson,Data Scientist,Research,bob@example.com
Carol Brown,Product Manager,Product,carol@example.com
David Wilson,DevOps Engineer,Engineering,david@example.com
Emma Davis,UX Designer,Design,emma@example.com
Frank Miller,Research Scientist,Research,frank@example.com
Grace Lee,Technical Writer,Documentation,grace@example.com
Henry Chen,System Administrator,IT,henry@example.com
''',
        }
        
        # Write test files
        for filename, content in test_files.items():
            file_path = self.test_data_dir / filename
            file_path.write_text(content)
            logger.info(f"Created test file: {filename}")
        
        # Create subdirectory with additional files
        subdir = self.test_data_dir / "subdir"
        subdir.mkdir()
        
        (subdir / "nested_file.txt").write_text(
            "This is a nested file for testing recursive directory processing. "
            "It should be indexed along with files in the parent directory."
        )
        
        return self.test_data_dir
    
    def cleanup_test_data(self):
        """Clean up test data directory."""
        if self.test_data_dir and self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
            logger.info(f"Cleaned up test data directory: {self.test_data_dir}")
            self.test_data_dir = None
    
    def cleanup_weaviate_class(self, class_prefix: str = TEST_CLASS_PREFIX):
        """Clean up test class in Weaviate."""
        try:
            client = weaviate.Client(self.weaviate_url)
            class_name = f"{class_prefix}_Node"
            client.schema.delete_class(class_name)
            logger.info(f"Cleaned up Weaviate class: {class_name}")
        except Exception as e:
            logger.debug(f"Failed to cleanup Weaviate class (may not exist): {e}")
    
    def test_file_processing(self) -> Dict[str, bool]:
        """Test file processing capabilities across different formats."""
        logger.info("Testing file processing capabilities")
        
        test_dir = self.setup_test_data()
        results = {}
        
        try:
            # Test individual file processing
            for file_path in test_dir.rglob("*"):
                if file_path.is_file():
                    try:
                        if self.scraper.is_supported(file_path):
                            text_content, metadata = self.scraper.scrape_file(file_path)
                            
                            # Validate extraction
                            results[file_path.name] = (
                                len(text_content.strip()) > 0 and
                                metadata['character_count'] > 0 and
                                metadata['word_count'] > 0
                            )
                            
                            logger.info(f"✓ {file_path.name}: {metadata['character_count']} chars, {metadata['word_count']} words")
                        else:
                            results[file_path.name] = False
                            logger.warning(f"✗ {file_path.name}: Unsupported file type")
                            
                    except Exception as e:
                        results[file_path.name] = False
                        logger.error(f"✗ {file_path.name}: {e}")
            
            success_rate = sum(results.values()) / len(results) if results else 0
            logger.info(f"File processing success rate: {success_rate:.2%} ({sum(results.values())}/{len(results)})")
            
            return results
            
        finally:
            self.cleanup_test_data()
    
    def test_indexing_pipeline(self) -> bool:
        """Test end-to-end indexing pipeline."""
        logger.info("Testing indexing pipeline")
        
        test_dir = self.setup_test_data()
        
        try:
            # Clean up any existing test data
            self.cleanup_weaviate_class()
            
            # Index the test directory
            self.service.index_directory(str(test_dir), TEST_CLASS_PREFIX, remove_existing=True)
            
            # Verify indexing succeeded
            client = weaviate.Client(self.weaviate_url)
            class_name = f"{TEST_CLASS_PREFIX}_Node"
            
            # Count indexed documents
            result = client.query.aggregate(class_name).with_fields("meta { count }").do()
            count = result["data"]["Aggregate"][class_name][0]["meta"]["count"]
            
            logger.info(f"Indexed {count} documents")
            
            # Should have indexed all supported files
            expected_files = [f for f in test_dir.rglob("*") 
                             if f.is_file() and self.scraper.is_supported(f)]
            
            success = count >= len(expected_files)
            
            if success:
                logger.info("✓ Indexing pipeline test passed")
            else:
                logger.error(f"✗ Indexing pipeline test failed: expected >= {len(expected_files)}, got {count}")
            
            return success
            
        except Exception as e:
            logger.error(f"✗ Indexing pipeline test failed: {e}")
            return False
        finally:
            self.cleanup_test_data()
            self.cleanup_weaviate_class()
    
    def test_search_quality(self) -> Dict[str, Dict]:
        """Test search quality with predefined queries."""
        logger.info("Testing search quality")
        
        test_dir = self.setup_test_data()
        
        # Test queries with expected content
        test_queries = [
            {
                "query": "How do I process CSV files?",
                "expected_content": ["csv", "process", "DataProcessor"],
                "expected_source": "example_script.py"
            },
            {
                "query": "What are the key features of the system?",
                "expected_content": ["Multi-format Support", "Real-time Monitoring", "Vector Search"],
                "expected_source": "project_documentation.md"
            },
            {
                "query": "What is the default Weaviate URL?",
                "expected_content": ["localhost:8777", "weaviate"],
                "expected_source": "system_config.json"
            },
            {
                "query": "Who works in the Engineering department?",
                "expected_content": ["Alice Smith", "David Wilson", "Engineering"],
                "expected_source": "sample_data.csv"
            }
        ]
        
        results = {}
        
        try:
            # Clean up and index test data
            self.cleanup_weaviate_class()
            self.service.index_directory(str(test_dir), TEST_CLASS_PREFIX, remove_existing=True)
            
            # Wait for indexing to complete
            time.sleep(2)
            
            for test_query in test_queries:
                query = test_query["query"]
                expected_content = test_query["expected_content"]
                expected_source = test_query["expected_source"]
                
                try:
                    # Perform search
                    response = self.service.search_documents(query, TEST_CLASS_PREFIX)
                    
                    # Check if expected content appears in response
                    content_found = all(content.lower() in response.lower() 
                                      for content in expected_content)
                    
                    # Check if expected source appears
                    source_found = expected_source.lower() in response.lower()
                    
                    results[query] = {
                        "content_found": content_found,
                        "source_found": source_found,
                        "response_length": len(response),
                        "success": content_found and source_found
                    }
                    
                    status = "✓" if results[query]["success"] else "✗"
                    logger.info(f"{status} Query: '{query}' - Content: {content_found}, Source: {source_found}")
                    
                except Exception as e:
                    results[query] = {"error": str(e), "success": False}
                    logger.error(f"✗ Query failed: '{query}' - {e}")
            
            success_rate = sum(r["success"] for r in results.values()) / len(results)
            logger.info(f"Search quality success rate: {success_rate:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"✗ Search quality test failed: {e}")
            return {}
        finally:
            self.cleanup_test_data()
            self.cleanup_weaviate_class()
    
    def test_monitoring_functionality(self) -> bool:
        """Test real-time monitoring functionality."""
        logger.info("Testing monitoring functionality")
        
        # This is a simplified test - full monitoring test requires async setup
        test_dir = self.setup_test_data()
        
        try:
            # Create a simple test to verify monitoring components work
            from scraping.filesystem.filesystem_monitor import FilesystemMonitor
            
            monitor = FilesystemMonitor(test_dir, debounce=100)
            
            # Test that monitor can be created and configured
            success = (
                monitor.root == test_dir and
                monitor.debounce == 100 and
                hasattr(monitor, 'start') and
                hasattr(monitor, 'stop')
            )
            
            if success:
                logger.info("✓ Monitoring functionality test passed")
            else:
                logger.error("✗ Monitoring functionality test failed")
            
            return success
            
        except Exception as e:
            logger.error(f"✗ Monitoring functionality test failed: {e}")
            return False
        finally:
            self.cleanup_test_data()
    
    def test_performance_benchmark(self) -> Dict[str, float]:
        """Run performance benchmarks."""
        logger.info("Running performance benchmarks")
        
        test_dir = self.setup_test_data()
        benchmarks = {}
        
        try:
            self.cleanup_weaviate_class()
            
            # Benchmark indexing time
            start_time = time.time()
            self.service.index_directory(str(test_dir), TEST_CLASS_PREFIX, remove_existing=True)
            indexing_time = time.time() - start_time
            benchmarks["indexing_time"] = indexing_time
            
            # Wait for indexing to complete
            time.sleep(1)
            
            # Benchmark search time
            test_query = "What are the key features of the system?"
            start_time = time.time()
            self.service.search_documents(test_query, TEST_CLASS_PREFIX)
            search_time = time.time() - start_time
            benchmarks["search_time"] = search_time
            
            # Count documents for rate calculation
            client = weaviate.Client(self.weaviate_url)
            result = client.query.aggregate(f"{TEST_CLASS_PREFIX}_Node").with_fields("meta { count }").do()
            doc_count = result["data"]["Aggregate"][f"{TEST_CLASS_PREFIX}_Node"][0]["meta"]["count"]
            
            benchmarks["documents_per_second"] = doc_count / indexing_time if indexing_time > 0 else 0
            benchmarks["document_count"] = doc_count
            
            logger.info(f"Indexing time: {indexing_time:.2f}s")
            logger.info(f"Search time: {search_time:.3f}s")
            logger.info(f"Documents per second: {benchmarks['documents_per_second']:.2f}")
            
            return benchmarks
            
        except Exception as e:
            logger.error(f"✗ Performance benchmark failed: {e}")
            return {}
        finally:
            self.cleanup_test_data()
            self.cleanup_weaviate_class()
    
    def run_all_tests(self) -> Dict[str, any]:
        """Run all integration tests and return comprehensive results."""
        logger.info("Running all integration tests")
        
        results = {
            "timestamp": time.time(),
            "weaviate_url": self.weaviate_url,
            "tests": {}
        }
        
        try:
            # Test 1: File processing
            results["tests"]["file_processing"] = self.test_file_processing()
            
            # Test 2: Indexing pipeline
            results["tests"]["indexing_pipeline"] = self.test_indexing_pipeline()
            
            # Test 3: Search quality
            results["tests"]["search_quality"] = self.test_search_quality()
            
            # Test 4: Monitoring functionality
            results["tests"]["monitoring_functionality"] = self.test_monitoring_functionality()
            
            # Test 5: Performance benchmarks
            results["tests"]["performance"] = self.test_performance_benchmark()
            
            # Calculate overall success
            all_passed = (
                all(results["tests"]["file_processing"].values()) and
                results["tests"]["indexing_pipeline"] and
                all(r.get("success", False) for r in results["tests"]["search_quality"].values()) and
                results["tests"]["monitoring_functionality"]
            )
            
            results["overall_success"] = all_passed
            
            logger.info(f"Overall test result: {'PASSED' if all_passed else 'FAILED'}")
            
            return results
            
        except Exception as e:
            logger.error(f"✗ Test suite failed: {e}")
            results["error"] = str(e)
            results["overall_success"] = False
            return results


def run_cli_tests() -> bool:
    """Test CLI interface functionality."""
    logger.info("Testing CLI interface")
    
    try:
        # Test help command
        result = subprocess.run([
            sys.executable, "filesystem_rag_service.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        help_success = (
            result.returncode == 0 and
            "index" in result.stdout and
            "search" in result.stdout and
            "monitor" in result.stdout
        )
        
        if help_success:
            logger.info("✓ CLI help command works")
        else:
            logger.error("✗ CLI help command failed")
        
        return help_success
        
    except Exception as e:
        logger.error(f"✗ CLI test failed: {e}")
        return False


# Pytest integration for automated testing
class TestFilesystemRAGIntegration:
    """Pytest test class for automated testing."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class."""
        cls.tester = FilesystemRAGTester(debug=True)
    
    def test_file_processing(self):
        """Test file processing functionality."""
        results = self.tester.test_file_processing()
        assert len(results) > 0, "No files were processed"
        assert all(results.values()), f"Some files failed processing: {results}"
    
    def test_indexing_pipeline(self):
        """Test indexing pipeline."""
        success = self.tester.test_indexing_pipeline()
        assert success, "Indexing pipeline test failed"
    
    def test_search_quality(self):
        """Test search quality."""
        results = self.tester.test_search_quality()
        assert len(results) > 0, "No search queries were tested"
        success_rate = sum(r.get("success", False) for r in results.values()) / len(results)
        assert success_rate >= 0.75, f"Search quality too low: {success_rate:.2%}"
    
    def test_monitoring_functionality(self):
        """Test monitoring functionality."""
        success = self.tester.test_monitoring_functionality()
        assert success, "Monitoring functionality test failed"
    
    def test_cli_interface(self):
        """Test CLI interface."""
        success = run_cli_tests()
        assert success, "CLI interface test failed"


def main():
    """Main function for running tests directly."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Filesystem RAG Integration Tests')
    parser.add_argument('-w', '--weaviate-url', type=str, default=DEFAULT_WEAVIATE_URL,
                       help='Weaviate database URL')
    parser.add_argument('-d', '--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--test', choices=['all', 'processing', 'indexing', 'search', 'monitoring', 'performance'],
                       default='all', help='Specific test to run')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize tester
    tester = FilesystemRAGTester(args.weaviate_url, args.debug)
    
    try:
        if args.test == 'all':
            results = tester.run_all_tests()
            print(f"\nTest Results Summary:")
            print(f"Overall Success: {results['overall_success']}")
            
            if results["tests"].get("performance"):
                perf = results["tests"]["performance"]
                print(f"Performance: {perf.get('documents_per_second', 0):.2f} docs/sec")
            
            return 0 if results['overall_success'] else 1
            
        elif args.test == 'processing':
            results = tester.test_file_processing()
            success = all(results.values())
            print(f"File processing test: {'PASSED' if success else 'FAILED'}")
            return 0 if success else 1
            
        elif args.test == 'indexing':
            success = tester.test_indexing_pipeline()
            print(f"Indexing pipeline test: {'PASSED' if success else 'FAILED'}")
            return 0 if success else 1
            
        elif args.test == 'search':
            results = tester.test_search_quality()
            success_rate = sum(r.get("success", False) for r in results.values()) / len(results)
            success = success_rate >= 0.75
            print(f"Search quality test: {'PASSED' if success else 'FAILED'} ({success_rate:.2%})")
            return 0 if success else 1
            
        elif args.test == 'monitoring':
            success = tester.test_monitoring_functionality()
            print(f"Monitoring functionality test: {'PASSED' if success else 'FAILED'}")
            return 0 if success else 1
            
        elif args.test == 'performance':
            results = tester.test_performance_benchmark()
            if results:
                print(f"Performance benchmark completed:")
                print(f"  Indexing time: {results.get('indexing_time', 0):.2f}s")
                print(f"  Search time: {results.get('search_time', 0):.3f}s")
                print(f"  Documents/sec: {results.get('documents_per_second', 0):.2f}")
                return 0
            else:
                print("Performance benchmark failed")
                return 1
                
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)