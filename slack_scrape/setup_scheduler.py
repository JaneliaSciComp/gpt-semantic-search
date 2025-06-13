#!/usr/bin/env python3
"""
Setup script for scheduling the daily Slack scraper.
Creates cron job and provides instructions for manual setup.
"""

import os
import sys
import subprocess
from pathlib import Path


def get_python_path():
    """Get the full path to the Python interpreter."""
    return sys.executable


def get_script_dir():
    """Get the absolute path to the current script directory."""
    return str(Path(__file__).parent.absolute())


def create_cron_entry():
    """Create the cron entry string."""
    python_path = get_python_path()
    script_dir = get_script_dir()
    script_path = os.path.join(script_dir, "slack_daily_scraper.py")
    
    # Run daily at 9:00 AM
    cron_entry = f"0 3 * * * cd {script_dir} && {python_path} {script_path} >> logs/cron.log 2>&1"
    
    return cron_entry


def setup_cron_job():
    """Setup the cron job automatically (requires user confirmation)."""
    print("Setting up daily Slack scraper cron job...")
    print()
    
    cron_entry = create_cron_entry()
    
    print("The following cron entry will be added:")
    print(f"  {cron_entry}")
    print()
    print("This will run the scraper daily at 9:00 AM")
    print()
    
    response = input("Do you want to add this cron job? (y/N): ").lower()
    
    if response in ['y', 'yes']:
        try:
            # Get current crontab
            result = subprocess.run(['crontab', '-l'], 
                                  capture_output=True, 
                                  text=True)
            
            current_cron = result.stdout if result.returncode == 0 else ""
            
            # Check if entry already exists
            if "slack_daily_scraper.py" in current_cron:
                print("Cron job for slack scraper already exists!")
                return
            
            # Add new entry
            new_cron = current_cron + cron_entry + "\n"
            
            # Install new crontab
            process = subprocess.Popen(['crontab', '-'], 
                                     stdin=subprocess.PIPE, 
                                     text=True)
            process.communicate(input=new_cron)
            
            if process.returncode == 0:
                print("✅ Cron job added successfully!")
                print("The scraper will run daily at 9:00 AM")
            else:
                print("❌ Failed to add cron job")
                
        except Exception as e:
            print(f"❌ Error setting up cron job: {e}")
            print_manual_instructions()
    else:
        print("Cron job not added. You can set it up manually using the instructions below.")
        print_manual_instructions()


def print_manual_instructions():
    """Print manual setup instructions."""
    cron_entry = create_cron_entry()
    
    print("\n" + "="*60)
    print("MANUAL SETUP INSTRUCTIONS")
    print("="*60)
    print()
    print("1. Open your crontab for editing:")
    print("   crontab -e")
    print()
    print("2. Add the following line:")
    print(f"   {cron_entry}")
    print()
    print("3. Save and exit the editor")
    print()
    print("SCHEDULE EXPLANATION:")
    print("  0 9 * * *  - Runs daily at 9:00 AM")
    print("  You can modify the time by changing '0 9' to:")
    print("    0 6   - 6:00 AM")
    print("    30 12 - 12:30 PM") 
    print("    0 18  - 6:00 PM")
    print("    etc.")
    print()
    print("TESTING:")
    print("  You can test the scraper manually by running:")
    print(f"  cd {get_script_dir()}")
    print("  python slack_daily_scraper.py")
    print()


def create_directories():
    """Create necessary directories."""
    directories = ["logs", "../data"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def check_requirements():
    """Check if required files and environment are set up."""
    issues = []
    
    if not os.getenv("SCRAPING_SLACK_BOT_TOKEN"):
        issues.append("❌ SCRAPING_SLACK_BOT_TOKEN not found in environment")
    else:
        print("✅ SCRAPING_SLACK_BOT_TOKEN found in environment")
    
    # Check for requirements.txt
    if not os.path.exists("requirements.txt"):
        issues.append("❌ requirements.txt not found")
    else:
        print("✅ requirements.txt found")
    
    # Check if dependencies are installed
    try:
        import slack_sdk
        print("✅ Required Python packages are installed")
    except ImportError as e:
        issues.append(f"❌ Missing Python package: {e}")
    
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print("\nPlease resolve these issues before setting up the scheduler.")
        return False
    
    return True


def main():
    """Main setup function."""
    print("Slack Daily Scraper Setup")
    print("=" * 30)
    print()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create necessary directories
    print("\nCreating directories...")
    create_directories()
    
    # Make the scraper script executable
    scraper_path = "slack_daily_scraper.py"
    if os.path.exists(scraper_path):
        os.chmod(scraper_path, 0o755)
        print(f"✅ Made {scraper_path} executable")
    
    print("\n" + "="*30)
    print("CRON JOB SETUP")
    print("="*30)
    
    # Setup cron job
    setup_cron_job()
    
    print("\n" + "="*30)
    print("SETUP COMPLETE!")
    print("="*30)
    print()
    print("Your Slack scraper is now ready to run daily.")
    print("Log files will be saved in the 'logs' directory.")
    print("Message data will be saved in the 'data' directory.")


if __name__ == "__main__":
    main()