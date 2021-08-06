# -*- coding: utf-8 -*-
"""
Download the orthoseg sample projects.
"""

from pathlib import Path
from orthoseg.util import git_downloader

if __name__ == '__main__':

    # Download sampleproject and run
    orthoseg_dir = Path.home() / 'orthoseg'
    projects_dir = orthoseg_dir / 'sample_projects'
    if not projects_dir.exists():
        print("Download sample projects")
        git_repo_dir = 'https://github.com/theroggy/orthoseg/tree/master/sample_projects'
        git_downloader.download(repo_url=git_repo_dir, output_dir=orthoseg_dir)
