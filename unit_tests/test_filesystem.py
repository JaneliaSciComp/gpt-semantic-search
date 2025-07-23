#!/usr/bin/env python3
"""
Filesystem processing unit tests for the unit testing framework.

This module provides test cases specifically for filesystem indexing and processing
to ensure file processing works correctly with various file types.
"""

import tempfile
import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scraping.filesystem import (
    UniversalFileScraper,
    get_supported_extensions,
    is_supported_file_type,
    get_processing_method
)


def create_test_files(test_dir: Path) -> dict:
    """Create a set of test files for validation."""
    test_files = {}
    
    # Create a Python file
    py_file = test_dir / "sample.py"
    py_content = '''#!/usr/bin/env python3
"""Sample Python script for testing."""

def hello_world():
    """Print hello world message."""
    print("Hello, World!")
    return True

if __name__ == "__main__":
    hello_world()
'''
    py_file.write_text(py_content)
    test_files['python'] = py_file
    
    # Create a Markdown file
    md_file = test_dir / "README.md"
    md_content = '''# Test Document

This is a test markdown document for filesystem processing validation.

## Features

- **Text extraction**: Supports various file formats
- **Processing**: Uses unstructured and native methods
- **Integration**: Works with existing RAG pipeline

## Code Example

```python
from scraping.filesystem import UniversalFileScraper
scraper = UniversalFileScraper()
text, metadata = scraper.scrape_file(Path("document.pdf"))
```

This concludes our test document.
'''
    md_file.write_text(md_content)
    test_files['markdown'] = md_file
    
    # Create a JSON file
    json_file = test_dir / "config.json"
    json_content = {
        "application": "filesystem-processor",
        "version": "1.0.0",
        "settings": {
            "debug": False,
            "supported_extensions": [".py", ".md", ".json", ".txt"],
            "processing_methods": ["unstructured", "native", "jupyter"]
        },
        "metadata": {
            "created": "2024-01-01",
            "author": "Test Suite",
            "description": "Configuration file for filesystem processing tests"
        }
    }
    json_file.write_text(json.dumps(json_content, indent=2))
    test_files['json'] = json_file
    
    return test_files


def test_filesystem_processing():
    """Test filesystem processing capabilities."""
    print("Testing Filesystem Processing")
    print("=" * 40)
    
    # Test 1: Extension support
    extensions = get_supported_extensions()
    print(f"✓ Supported extensions: {len(extensions)}")
    assert len(extensions) > 50, "Should support 50+ file extensions"
    
    # Test 2: File type detection
    test_cases = [
        (Path("test.py"), True, "text"),
        (Path("test.pdf"), True, "unstructured"), 
        (Path("test.ipynb"), True, "jupyter"),
        (Path("test.xyz"), False, "unsupported")
    ]
    
    for path, expected_supported, expected_method in test_cases:
        supported = is_supported_file_type(path)
        method = get_processing_method(path)
        assert supported == expected_supported, f"Support detection failed for {path}"
        assert method == expected_method, f"Method detection failed for {path}"
        print(f"✓ {path.name}: {method}")
    
    # Test 3: File processing with real files
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        test_files = create_test_files(test_dir)
        
        scraper = UniversalFileScraper()
        
        for file_type, file_path in test_files.items():
            try:
                text_content, metadata = scraper.scrape_file(file_path)
                
                # Validate extraction
                assert text_content and len(text_content.strip()) > 0, f"No content from {file_type}"
                assert metadata['character_count'] > 0, f"No character count for {file_type}"
                assert metadata['word_count'] > 0, f"No word count for {file_type}"
                
                print(f"✓ {file_type}: {metadata['character_count']} chars, {metadata['word_count']} words")
                
            except Exception as e:
                print(f"✗ {file_type}: {e}")
                raise
    
    # Test 4: Directory processing
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        test_files = create_test_files(test_dir)
        
        scraper = UniversalFileScraper()
        results = scraper.scrape_directory(test_dir)
        
        assert len(results) == len(test_files), f"Expected {len(test_files)} files, got {len(results)}"
        
        stats = scraper.get_scraping_stats(results)
        assert stats['total_files'] == len(test_files), "File count mismatch"
        assert stats['total_characters'] > 0, "No characters extracted"
        
        print(f"✓ Directory processing: {stats['total_files']} files, {stats['total_characters']} chars")
    
    print("✓ All filesystem tests passed!")
    return True


if __name__ == '__main__':
    """Run filesystem tests when executed directly."""
    try:
        success = test_filesystem_processing()
        if success:
            print("\n🎉 Filesystem processing tests completed successfully!")
            exit(0)
        else:
            print("\n❌ Some filesystem tests failed!")
            exit(1)
    except Exception as e:
        print(f"\n❌ Filesystem test error: {e}")
        exit(1)