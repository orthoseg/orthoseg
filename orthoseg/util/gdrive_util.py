from pathlib import Path
from typing import Union
import requests


def download_file(
    id: str, destination: Path, ssl_verify: Union[bool, str, None] = None
):
    """
    Downloads the file to the destination filepath.

    Args
        id (str): id of the file to download.
        destination (Path): file path to download the file to.
        ssl_verify (bool or str, optional): True or None to use the default
            certificate bundle as installed on your system. False disables
            certificate validation (NOT recommended!). If a path to a
            certificate bundle file (.pem) is passed, this will be used.
            In corporate networks using a proxy server this is often needed
            to evade CERTIFICATE_VERIFY_FAILED errors. Defaults to None.
    """

    URL = "https://docs.google.com/uc?export=download"

    if ssl_verify is not None:
        # If it is a string, make sure it isn't actually a bool
        if isinstance(ssl_verify, str):
            if ssl_verify.lower() == "true":
                ssl_verify = True
            elif ssl_verify.lower() == "false":
                ssl_verify = False

        if isinstance(ssl_verify, bool) and not ssl_verify:
            print("SSL VERIFICATION IS TURNED OFF!!!")

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True, verify=ssl_verify)
    token = _get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    _save_response_content(response, destination)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def _save_response_content(response, destination: Path):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
