# -*- coding: utf-8 -*-
"""
Download the sample project.
"""

import argparse
from pathlib import Path
import sys
import tempfile

# Because orthoseg isn't installed as package + it is higher in dir hierarchy, add root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orthoseg.util import git_downloader

def main():
    
    ### Interprete arguments ###
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('dest_dir', help='The directory to download the orthoseg sample projects to. Use ~ for you home directory.')
    args = parser.parse_args()
    dest_dir = Path(args.dest_dir).expanduser() / "orthoseg"

    dest_dir_full = dest_dir / "sample_projects"
    if dest_dir_full.exists():
        raise Exception(f"Destination directory already exists: {dest_dir_full}")

    ### Download ###        
    print(f"Start download of sample projects to {str(dest_dir_full)}")
    git_downloader.download(
            repo_url="https://github.com/orthoseg/orthoseg/tree/master/sample_projects", 
            output_dir=dest_dir)
    print(f"Download finished")

if __name__ == '__main__':
    main()
