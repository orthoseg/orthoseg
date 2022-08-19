#!/usr/bin/python3
# Downloaded from https://github.com/sdushantha/gitdir/blob/master/gitdir/gitdir.py

import re
import os
import urllib.request
import json
from pathlib import Path


def create_url(url):
    """
    From the given url, produce a URL that is compatible with Github's REST API.
    Can handle blob or tree paths.
    """

    # extract the branch name from the given url (e.g master)
    branch = re.findall(r"/tree/(.*?)/", url)
    api_url = url.replace("https://github.com", "https://api.github.com/repos")
    if len(branch) == 0:
        branch = re.findall(r"/blob/(.*?)/", url)[0]
        download_dirs = re.findall(r"/blob/" + branch + r"/(.*)", url)[0]
        api_url = re.sub(r"/blob/.*?/", "/contents/", api_url)
    else:
        branch = branch[0]
        download_dirs = re.findall(r"/tree/" + branch + r"/(.*)", url)[0]
        api_url = re.sub(r"/tree/.*?/", "/contents/", api_url)

    api_url = api_url + "?ref=" + branch
    return api_url, download_dirs


def download(repo_url: str, output_dir: Path):
    """
    Downloads the files and directories in repo_url. If flatten is specified, the
    contents of any and all sub-directories will be pulled upwards into the root folder.
    """

    # generate the url which returns the JSON data
    api_url, download_dirs = create_url(repo_url)

    # To handle file names.
    if len(download_dirs.split(".")) == 0:
        dir_out = output_dir / download_dirs
    else:
        dir_out = output_dir / "/".join(download_dirs.split("/")[0:-1])

    response = urllib.request.urlretrieve(api_url)

    # make a directory with the name which is taken from
    # the actual repo
    os.makedirs(str(dir_out), exist_ok=True)

    # total files count
    total_files = 0

    with open(response[0], "r") as f:
        raw_data = f.read()
        data = json.loads(raw_data)

        # Get the total number of files
        total_files += len(data)

        # If the data is a file, download it as one.
        if isinstance(data, dict) and data["type"] == "file":
            # download the file
            urllib.request.urlretrieve(data["download_url"], dir_out / data["name"])
            return total_files

        # Loop over all files/dirs found
        for file in data:
            file_url = file["download_url"]
            path = output_dir / file["path"]
            os.makedirs(path.parent, exist_ok=True)

            # If it is a file, download it, if dir, start recursively
            if file_url is not None:
                urllib.request.urlretrieve(file_url, path)
            else:
                download(file["html_url"], output_dir)

    return total_files
