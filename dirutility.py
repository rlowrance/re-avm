'''utilities for managing directories'''
import os


def assure_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)  # make all intermediate directories
    return dir_path
