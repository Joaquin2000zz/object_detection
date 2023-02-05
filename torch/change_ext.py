"""
module which contains change_ext function
"""
import os
from glob import glob


def change_ext(path: str='./images/', ext='.jpg'):
    """
    function which changes the format of given path to .png
    CAUTION: this changes all files in specifyed directory
    """
    if not isinstance(path, str):
        raise TypeError('path must be a str representing the path used')
    if path[-1] != '/':
        path += '/'
    for i, f in enumerate(glob(f"{path}*")):
        try:
            if ext == '.jpg':
                os.rename(f, f"{path}{i}{ext}")
            else:
                os.rename(f, f"{f[:-4]}{ext}")
        except:
            pass
