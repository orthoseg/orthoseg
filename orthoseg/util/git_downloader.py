"""Module to download files from a github repo.

Based on https://github.com/sdushantha/gitdir/blob/master/gitdir/gitdir.py
"""
# mypy: ignore-errors

import json
import re
import ssl
import time
import urllib.request
from pathlib import Path


def create_url(url: str) -> tuple[str, list[Path]]:
    """From the given url, produce a URL that is compatible with Github's REST API.

    Can handle blob or tree paths.

    Args:
        url (str): the base url.

    Returns:
        (str, list[Path]): Github REST API compatible url + directories to download.
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


def download(
    repo_url: str,
    output_dir: Path,
    ssl_verify: bool | str | None = None,
    limit_rate: bool = True,
) -> int | None:
    """Downloads the files and directories in repo_url.

    Args:
        repo_url (str): url to the repository to download.
        output_dir (Path): directory to download the repository to.
        ssl_verify (bool or str, optional): True or None to use the default
            certificate bundle as installed on your system. False disables
            certificate validation (NOT recommended!). If a path to a
            certificate bundle file (.pem) is passed, this will be used.
            In corporate networks using a proxy server this is often needed
            to evade CERTIFICATE_VERIFY_FAILED errors. Defaults to None.
        limit_rate (bool, optional): If True, limit the rate requests are done to
            github to the maximum level allowed for unauthenticated users.
            Defaults to True.
    """
    total_files = None
    context = None
    if ssl_verify is not None:
        # If it is a string, make sure it isn't actually a bool
        if isinstance(ssl_verify, str):
            if ssl_verify.lower() == "true":
                context = None
            elif ssl_verify.lower() == "false":
                ssl_verify = False
            else:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                context.load_verify_locations(ssl_verify)

        if isinstance(ssl_verify, bool) and not ssl_verify:
            context = ssl._create_unverified_context()
            print("SSL VERIFICATION IS TURNED OFF!!!")

    # Generate the url which returns the JSON data
    api_url, download_dirs = create_url(repo_url)

    # To handle file names.
    if len(download_dirs.split(".")) == 0:
        dir_out = output_dir / download_dirs
    else:
        dir_out = output_dir / "/".join(download_dirs.split("/")[0:-1])

    # Download the file
    # If limit_rate enabled, always sleep 1 second before doing a request!
    if limit_rate:
        time.sleep(1)

    url = api_url
    try:
        with urllib.request.urlopen(url, context=context) as u:
            # Make a directory with the name which is taken from
            # the actual repo
            dir_out.mkdir(parents=True, exist_ok=True)

            # Total files count
            total_files = 0
            raw_data = u.read()
            data = json.loads(raw_data)

            # Get the total number of files
            total_files += len(data)

            # If the data is a file, download it as one.
            if isinstance(data, dict) and data["type"] == "file":
                # Download the file
                if limit_rate:
                    time.sleep(1)
                url = data["download_url"]
                with (
                    urllib.request.urlopen(url, context=context) as u,
                    open(dir_out / data["name"], "wb") as f,
                ):
                    f.write(u.read())

                return total_files

            # Loop over all files/dirs found
            for file in data:
                url = file["download_url"]
                path = output_dir / file["path"]
                path.parent.mkdir(parents=True, exist_ok=True)

                # If it is a file, download it, if dir, start recursively
                if url is not None:
                    if limit_rate:
                        time.sleep(1)
                    with (
                        urllib.request.urlopen(url, context=context) as u,
                        open(path, "wb") as f,
                    ):
                        f.write(u.read())
                else:
                    download(file["html_url"], output_dir)

    except urllib.error.HTTPError as ex:  # pragma: no cover
        if ex.code == 403:
            message = f"Error: API Rate limit exceeded for url {url}"
        else:
            message = f"Error with url: {url}: {ex}"
        raise RuntimeError(message) from ex
    except Exception as ex:
        raise RuntimeError(f"Error with url: {url}: {ex}") from ex

    return total_files
