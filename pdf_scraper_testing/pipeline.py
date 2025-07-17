import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

from scrapers import (
    DoclingPDFScraper,
    PyPDF2Scraper,
    PyMuPDFScraper,
    PDFPlumberScraper,
    PDFMinerScraper,
    LlamaParseScraper,
    UnstructuredScraper,
    BasePDFScraper
)

# Import weaviate integration from existing codebase
import sys
sys.path.append('..')
from indexing.weaviate_indexer import Indexer
from llama_index.core import Document

logger = logging.getLogger(__name__)


class PDFScraperPipeline:
    """Pipeline for testing and comparing different PDF scrapers"""
    
    def __init__(self, weaviate_url: str = "http://localhost:8080", test_class_prefix: str = "PDFTest", scraper_names: Optional[List[str]] = None):
        self.weaviate_url = weaviate_url
        self.test_class_prefix = test_class_prefix
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize scrapers (all available or specified subset)
        self.scrapers = self._initialize_scrapers(scraper_names)
        
    def _initialize_scrapers(self, scraper_names: Optional[List[str]] = None) -> List[BasePDFScraper]:
        """Initialize PDF scrapers (all available or specified subset)"""
        scrapers = []
        
        # Mapping of scraper names to classes
        scraper_map = {
            'docling': DoclingPDFScraper,
            'pypdf2': PyPDF2Scraper,
            'pymupdf': PyMuPDFScraper,
            'pdfplumber': PDFPlumberScraper,
            'pdfminer': PDFMinerScraper,
            'llamaparse': LlamaParseScraper,
            'unstructured': UnstructuredScraper,
        }
        
        # Determine which scrapers to initialize
        if scraper_names is None:
            # Use all available scrapers
            scrapers_to_init = scraper_map.items()
        else:
            # Use only specified scrapers
            scrapers_to_init = [(name, scraper_map[name]) for name in scraper_names if name in scraper_map]
            
            # Warn about unknown scraper names
            unknown_scrapers = [name for name in scraper_names if name not in scraper_map]
            if unknown_scrapers:
                logger.warning(f"Unknown scraper names: {unknown_scrapers}. Available: {list(scraper_map.keys())}")
        
        # Initialize scrapers
        for scraper_name, scraper_class in scrapers_to_init:
            try:
                scraper = scraper_class()
                scrapers.append(scraper)
                logger.info(f"Initialized {scraper.name} scraper")
            except (ImportError, ValueError) as e:
                logger.warning(f"Could not initialize {scraper_name}: {e}")
        
        return scrapers
    
    def list_available_scrapers(self) -> List[str]:
        """List all available scraper names"""
        return ['docling', 'pypdf2', 'pymupdf', 'pdfplumber', 'pdfminer', 'llamaparse', 'unstructured']
    
    def test_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Test a single PDF with initialized scrapers"""
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        results = {
            'pdf_path': pdf_path,
            'timestamp': datetime.now().isoformat(),
            'scrapers': []
        }
        
        for scraper in self.scrapers:
            logger.info(f"Testing {scraper.name} on {pdf_path}")
            result = scraper.extract_with_metadata(pdf_path)
            results['scrapers'].append(result)
        
        return results
    
    def test_pdf_with_scrapers(self, pdf_path: str, scraper_names: List[str]) -> Dict[str, Any]:
        """Test a single PDF with specific scrapers"""
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Initialize only the specified scrapers for this test
        temp_scrapers = self._initialize_scrapers(scraper_names)
        
        results = {
            'pdf_path': pdf_path,
            'timestamp': datetime.now().isoformat(),
            'scrapers': []
        }
        
        for scraper in temp_scrapers:
            logger.info(f"Testing {scraper.name} on {pdf_path}")
            result = scraper.extract_with_metadata(pdf_path)
            results['scrapers'].append(result)
        
        return results
    
    def test_pdf_bytes(self, pdf_bytes: bytes, pdf_name: str = "unknown.pdf") -> Dict[str, Any]:
        """Test PDF bytes with initialized scrapers"""
        results = {
            'pdf_name': pdf_name,
            'timestamp': datetime.now().isoformat(),
            'scrapers': []
        }
        
        for scraper in self.scrapers:
            logger.info(f"Testing {scraper.name} on {pdf_name}")
            result = scraper.extract_bytes_with_metadata(pdf_bytes)
            results['scrapers'].append(result)
        
        return results
    
    def test_directory(self, pdf_directory: str) -> List[Dict[str, Any]]:
        """Test all PDFs in a directory"""
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Directory not found: {pdf_directory}")
        
        results = []
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_directory}")
        
        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file.name}")
            try:
                result = self.test_pdf(str(pdf_file))
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                results.append({
                    'pdf_path': str(pdf_file),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def index_with_weaviate(self, results: List[Dict[str, Any]], scraper_name: str) -> None:
        """Index extraction results with Weaviate for comparison"""
        # Create documents from scraper results
        documents = []
        
        for result in results:
            if 'scrapers' in result:
                for scraper_result in result['scrapers']:
                    if scraper_result['scraper'] == scraper_name and scraper_result['success']:
                        # Create document for indexing
                        doc = Document(
                            text=scraper_result['text'],
                            metadata={
                                'title': Path(result['pdf_path']).stem,
                                'source': f"PDF_Test_{scraper_name}",
                                'link': result['pdf_path'],
                                'scraped_at': datetime.now().timestamp(),
                                'scraper_used': scraper_name,
                                'processing_time': scraper_result['processing_time'],
                                'file_size': scraper_result['file_size'],
                                'characters_extracted': scraper_result['characters_extracted']
                            }
                        )
                        documents.append(doc)
        
        if not documents:
            logger.warning(f"No documents to index for {scraper_name}")
            return
        
        # Index with Weaviate
        class_prefix = f"{self.test_class_prefix}_{scraper_name}"
        indexer = Indexer(
            weaviate_url=self.weaviate_url,
            class_prefix=class_prefix,
            delete_database=True  # Clean slate for each test
        )
        
        logger.info(f"Indexing {len(documents)} documents with {scraper_name}")
        indexer.index(documents)
    
    def compare_scrapers(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compare performance across different scrapers"""
        comparison_data = []
        
        for result in results:
            if 'scrapers' in result:
                pdf_name = Path(result['pdf_path']).name
                
                for scraper_result in result['scrapers']:
                    comparison_data.append({
                        'pdf_name': pdf_name,
                        'scraper': scraper_result['scraper'],
                        'success': scraper_result['success'],
                        'processing_time': scraper_result['processing_time'],
                        'file_size': scraper_result['file_size'],
                        'characters_extracted': scraper_result['characters_extracted'],
                        'words_extracted': scraper_result['words_extracted'],
                        'error': scraper_result.get('error', None)
                    })
        
        df = pd.DataFrame(comparison_data)
        
        # Calculate additional metrics
        if not df.empty:
            df['chars_per_second'] = df['characters_extracted'] / df['processing_time']
            df['chars_per_mb'] = df['characters_extracted'] / (df['file_size'] / 1024 / 1024)
        
        return df
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = None) -> str:
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pdf_scraper_results_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return str(filepath)
    
    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate a comprehensive comparison report"""
        df = self.compare_scrapers(results)
        
        if df.empty:
            return "No valid results to generate report"
        
        # Performance summary
        summary = df.groupby('scraper').agg({
            'success': ['count', 'sum'],
            'processing_time': ['mean', 'std'],
            'characters_extracted': ['mean', 'std'],
            'chars_per_second': ['mean', 'std']
        }).round(3)
        
        # Success rates
        success_rates = df.groupby('scraper')['success'].agg(['count', 'sum']).round(3)
        success_rates['success_rate'] = success_rates['sum'] / success_rates['count']
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# PDF Scraper Comparison Report
Generated: {timestamp}

## Summary Statistics
{summary.to_string()}

## Success Rates
{success_rates.to_string()}

## Individual Results
{df.to_string()}
        """
        
        # Save report
        report_filename = f"pdf_scraper_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path = self.results_dir / report_filename
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {report_path}")
        return report
    
    def run_full_evaluation(self, pdf_directory: str, index_results: bool = True) -> str:
        """Run complete evaluation pipeline with initialized scrapers"""
        logger.info("Starting full PDF scraper evaluation")
        
        # Test all PDFs
        results = self.test_directory(pdf_directory)
        
        # Save raw results
        results_file = self.save_results(results)
        
        # Generate comparison report
        report = self.generate_report(results)
        
        # Optionally index results with Weaviate for each scraper
        if index_results:
            for scraper in self.scrapers:
                try:
                    self.index_with_weaviate(results, scraper.name)
                except Exception as e:
                    logger.error(f"Error indexing with {scraper.name}: {e}")
        
        logger.info("Full evaluation completed")
        return report
    
    def run_evaluation_with_scrapers(self, pdf_directory: str, scraper_names: List[str], index_results: bool = True) -> str:
        """Run evaluation pipeline with specific scrapers"""
        logger.info(f"Starting PDF scraper evaluation with: {scraper_names}")
        
        # Temporarily store current scrapers
        original_scrapers = self.scrapers
        
        # Set scrapers to specified ones for this evaluation
        self.scrapers = self._initialize_scrapers(scraper_names)
        
        try:
            # Run evaluation with specified scrapers
            results = self.test_directory(pdf_directory)
            results_file = self.save_results(results)
            report = self.generate_report(results)
            
            if index_results:
                for scraper in self.scrapers:
                    try:
                        self.index_with_weaviate(results, scraper.name)
                    except Exception as e:
                        logger.error(f"Error indexing with {scraper.name}: {e}")
        finally:
            # Restore original scrapers
            self.scrapers = original_scrapers
        
        logger.info("Evaluation with specific scrapers completed")
        return report