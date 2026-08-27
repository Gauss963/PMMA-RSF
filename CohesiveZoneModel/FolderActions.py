import os
import re
import shutil

def delete_pycache() -> None:
    current_directory = os.getcwd()
    pycache_directory = os.path.join(current_directory, '__pycache__')

    if os.path.exists(pycache_directory):
        shutil.rmtree(pycache_directory)

    return None