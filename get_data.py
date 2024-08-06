import gdown
import tarfile
import yaspin
import os

URL = "https://drive.google.com/uc?id=11FWSZov9NA6tbliJTOr7NFR38bEn8lRU"


if __name__ == "__main__":
    # Download file from Google Drive
    gdown.download(URL, output="data.tar.gz")

    # Extract the contents of the tarball
    with yaspin.yaspin(text="Extracting data.tar.gz") as sp:
        with tarfile.open("data.tar.gz", "r:gz") as tar:
            tar.extractall(".")

    # Remove the tarball
    os.remove("data.tar.gz")