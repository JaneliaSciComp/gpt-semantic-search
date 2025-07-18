#!/usr/bin/env python3
"""
PDF Scraper Testing Pipeline Runner

Simple command-line interface for testing different PDF scrapers.
"""

import argparse
import logging
import sys
from pathlib import Path

from pipeline import PDFScraperPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PDF_DIR = "test_pdfs"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_SCRAPERS = None  # None means use all available scrapers


def main():
    parser = argparse.ArgumentParser(
        description="Test and compare different PDF scraping libraries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                         # Test all scrapers in default dir
  %(prog)s --scrapers docling pymupdf             # Test specific scrapers
  %(prog)s --pdf-dir my_pdfs/                     # Test different directory
  %(prog)s --list-scrapers                        # Show available scrapers
  %(prog)s --single-pdf document.pdf --scrapers docling    # Test single PDF
        """
    )
    
    parser.add_argument(
        '--pdf-dir',
        type=str,
        default=DEFAULT_PDF_DIR,
        help=f'Directory containing PDF files to test (default: {DEFAULT_PDF_DIR})'
    )
    
    parser.add_argument(
        '--single-pdf',
        type=str,
        help='Single PDF file to test'
    )
    
    parser.add_argument(
        '--scrapers',
        nargs='+',
        help='Specific scrapers to use (space-separated). See --list-scrapers for options.'
    )
    
    parser.add_argument(
        '--scraper',
        type=str,
        help='Single scraper to use. Alternative to --scrapers for convenience.'
    )
    
    parser.add_argument(
        '--list-scrapers',
        action='store_true',
        help='List all available PDF scrapers'
    )
    
    parser.add_argument(
        '--no-index',
        action='store_true',
        help='Skip Weaviate indexing (faster, but no vector search evaluation)'
    )
    
    parser.add_argument(
        '--weaviate-url',
        type=str,
        default='http://localhost:8080',
        help='Weaviate URL (default: http://localhost:8080)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Directory to save results (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    args = parser.parse_args()
    
    # Handle scraper vs scrapers arguments
    if args.scraper and args.scrapers:
        parser.error("Cannot specify both --scraper and --scrapers")
    elif args.scraper:
        args.scrapers = [args.scraper]
    
    # Handle list-scrapers command
    if args.list_scrapers:
        pipeline = PDFScraperPipeline()
        available = pipeline.list_available_scrapers()
        print("Available PDF scrapers:")
        for scraper in available:
            print(f"  - {scraper}")
        print(f"\nTotal: {len(available)} scrapers")
        print("\nUsage example:")
        print(f"  python {sys.argv[0]} --scrapers docling pymupdf --pdf-dir test_pdfs/")
        return
    
    # Validate arguments
    if not args.pdf_dir and not args.single_pdf:
        # If no arguments provided, use default directory
        args.pdf_dir = DEFAULT_PDF_DIR
    
    if args.pdf_dir and args.single_pdf:
        parser.error("Cannot specify both --pdf-dir and --single-pdf")
    
    # Check if paths exist
    if args.pdf_dir:
        pdf_path = Path(args.pdf_dir)
        if not pdf_path.exists():
            logger.error(f"PDF directory not found: {args.pdf_dir}")
            if args.pdf_dir == DEFAULT_PDF_DIR:
                logger.info(f"Creating default PDF directory: {DEFAULT_PDF_DIR}")
                pdf_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Please add PDF files to {DEFAULT_PDF_DIR} and run again")
                sys.exit(1)
            else:
                sys.exit(1)
        if not list(pdf_path.glob("*.pdf")):
            logger.error(f"No PDF files found in directory: {args.pdf_dir}")
            if args.pdf_dir == DEFAULT_PDF_DIR:
                logger.info(f"Please add PDF files to {DEFAULT_PDF_DIR} directory")
            sys.exit(1)
    
    if args.single_pdf:
        pdf_file = Path(args.single_pdf)
        if not pdf_file.exists():
            logger.error(f"PDF file not found: {args.single_pdf}")
            sys.exit(1)
    
    # Initialize pipeline
    try:
        pipeline = PDFScraperPipeline(
            weaviate_url=args.weaviate_url,
            scraper_names=args.scrapers
        )
        
        # Override results directory if specified
        if args.output_dir != DEFAULT_OUTPUT_DIR:
            pipeline.results_dir = Path(args.output_dir)
            pipeline.results_dir.mkdir(exist_ok=True)
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Show what we're going to do
    active_scrapers = [scraper.name for scraper in pipeline.scrapers]
    if not active_scrapers:
        logger.error("No scrapers could be initialized. Check dependencies and API keys.")
        logger.error("Are you running in the pdf-testing environment?")
        logger.error("Try: pixi run -e pdf-testing python run_pipeline.py")
        sys.exit(1)
    
    logger.info(f"Active scrapers: {', '.join(active_scrapers)}")
    indexing_status = "disabled" if args.no_index else "enabled"
    logger.info(f"Weaviate indexing: {indexing_status}")
    
    # Run the pipeline
    try:
        if args.single_pdf:
            # Test single PDF
            logger.info(f"Testing single PDF: {args.single_pdf}")
            results = pipeline.test_pdf(args.single_pdf)
            
            # Print quick summary
            print(f"\nResults for {args.single_pdf}:")
            for scraper_result in results['scrapers']:
                status = "✓" if scraper_result['success'] else "✗"
                chars = scraper_result['characters_extracted']
                time_taken = scraper_result['processing_time']
                error_msg = f" (Error: {scraper_result['error']})" if scraper_result['error'] else ""
                print(f"  {status} {scraper_result['scraper']}: {chars} chars in {time_taken:.3f}s{error_msg}")
            
            # Save results
            results_file = pipeline.save_results([results], f"single_pdf_results.json")
            print(f"\nDetailed results saved to: {results_file}")
            
        else:
            # Test directory
            logger.info(f"Testing PDF directory: {args.pdf_dir}")
            report = pipeline.run_full_evaluation(args.pdf_dir, index_results=not args.no_index)
            
            print(f"\nEvaluation completed!")
            print(f"Results saved to: {pipeline.results_dir}")
            
            # Print summary from report
            lines = report.split('\n')
            summary_start = None
            for i, line in enumerate(lines):
                if '## Summary Statistics' in line:
                    summary_start = i
                    break
            
            if summary_start:
                summary_lines = lines[summary_start:summary_start+20]  # Show first 20 lines of summary
                print("\n" + "\n".join(summary_lines))
            
            # Print error summary if all scrapers failed
            if all(scraper_result['success'] == False for result in pipeline.test_directory(args.pdf_dir) for scraper_result in result.get('scrapers', [])):
                print("\n⚠️  ALL SCRAPERS FAILED - Check errors in detailed results")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()