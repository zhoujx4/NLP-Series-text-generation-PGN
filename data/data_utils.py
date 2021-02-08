"""
@Time : 2021/2/814:08
@Auth : 周俊贤
@File ：data_utils.py
@DESCRIPTION:

"""

import os
import pathlib
import sys

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))

def isChinese(word):
    """Distinguish Chinese words from non-Chinese ones.

    Args:
        word (str): The word to be distinguished.

    Returns:
        bool: Whether the word is a Chinese word.
    """
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False
