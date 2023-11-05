import os
import shutil
import tarfile

import requests
from tqdm import tqdm


def find_and_delete_folder(start_path, folder_name):
    for root, dirs, files in os.walk(start_path):
        if folder_name in dirs:
            folder_path = os.path.join(root, folder_name)
            print(f"Found folder at {folder_path}. Deleting...")
            shutil.rmtree(folder_path)
            print("Folder deleted.")
            return True
    print(f"Folder named {folder_name} not found.")
    return False


def download_and_unzip(url, save_path):
    # Get the file name from the URL
    file_name = url.split('/')[-1]

    # Stream the download
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    if response.status_code == 200:
        # Show progress bar using tqdm
        progress_bar = tqdm(total=total_size_in_bytes,
                            unit='iB', unit_scale=True)
        with open(file_name, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")

        # Unzip the tar file
        if file_name.endswith('.tar'):
            with tarfile.open(file_name) as tar:
                tar.extractall(path=save_path)
                print(f"Extracted all files in {file_name} to {save_path}")

        # Clean up the tar file from local disk
        os.remove(file_name)
        print(f"Removed the downloaded file: {file_name}")
    else:
        print(
            f"Failed to download the file. Status code: {response.status_code}")
