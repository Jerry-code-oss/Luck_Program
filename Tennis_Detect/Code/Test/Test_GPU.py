import os
import tempfile
import gc
import shutil

def clean_temp_files():
    temp_dir = tempfile.gettempdir()
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            try:
                os.remove(os.path.join(root, file))
            except PermissionError:
                pass

def clean_memory():
    gc.collect()

def clean_specific_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

clean_temp_files()
clean_memory()
clean_specific_directory('D:/Tennis_Detect/cache/')