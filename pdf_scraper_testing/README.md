# PDF Scraper Testing Pipeline

This directory contains a testing framework for comparing different PDF scraping libraries. It's isolated from the main project and is used only for evaluation purposes.

## Available PDF Scrapers

1. **Docling** - Already implemented in the main project
2. **PyPDF2** - Basic PDF text extraction
3. **PyMuPDF (fitz)** - Fast PDF processing with good text extraction
4. **pdfplumber** - Detailed PDF parsing with table support
5. **PDFMiner** - Low-level PDF parsing
6. **LlamaParse** - LlamaIndex's cloud-based PDF parsing service
7. **Unstructured.io** - Advanced document processing with structure detection

## Usage

Install the PDF testing environment:
```bash
pixi install -e pdf-testing
```

Set up API keys (if using cloud services):
```bash
export LLAMA_CLOUD_API_KEY="your_llamaparse_api_key"  # For LlamaParse
```

## Quick Start

**Command Line Usage (Recommended):**
```bash
# Activate the environment
pixi shell -e pdf-testing

# List available scrapers
python run_pipeline.py --list-scrapers

# Test all PDFs with all scrapers
python run_pipeline.py --pdf-dir test_pdfs/

# Test with specific scrapers only
python run_pipeline.py --scrapers docling pymupdf unstructured --pdf-dir test_pdfs/

# Test a single PDF
python run_pipeline.py --single-pdf test_pdfs/document.pdf --scrapers docling

# Skip Weaviate indexing for faster testing
python run_pipeline.py --pdf-dir test_pdfs/ --no-index
```

**Python API Usage:**
```python
from pdf_scraper_testing.pipeline import PDFScraperPipeline

# Initialize pipeline with all scrapers (default)
pipeline = PDFScraperPipeline()

# Or initialize with specific scrapers only
pipeline = PDFScraperPipeline(scraper_names=['docling', 'pymupdf', 'unstructured'])

# List available scraper names
print(pipeline.list_available_scrapers())

# Test a single PDF with initialized scrapers
results = pipeline.test_pdf("path/to/document.pdf")

# Test a single PDF with specific scrapers (on-the-fly)
results = pipeline.test_pdf_with_scrapers("path/to/document.pdf", ['docling', 'llamaparse'])

# Run evaluation with initialized scrapers
evaluation_results = pipeline.run_full_evaluation("test_pdfs/")

# Run evaluation with specific scrapers
evaluation_results = pipeline.run_evaluation_with_scrapers("test_pdfs/", ['docling', 'unstructured'])
```

## Structure

- `scrapers/` - Individual scraper implementations
- `pipeline.py` - Main testing pipeline
- `evaluation.py` - Evaluation metrics and comparison
- `test_pdfs/` - Sample PDFs for testing
- `results/` - Output results and comparisons