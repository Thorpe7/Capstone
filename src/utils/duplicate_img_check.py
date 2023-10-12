""" Quick and dirty method for identifying duplicate images in dataset """
""" Dup data can create overly accurate test result despite data splitting"""

from difPy import dif
import json
from pathlib import Path


def search_dups(dir_name: str, filename: str) -> None:
    search = dif([dir_name], recursive=True, delete=True)

    with open(filename, "w", encoding="utf-8") as fp:
        json.dump(search.stats, fp, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    dir_input = input("Enter directory path: ")
    user_input = input("Enter file name to use including extension: ")
    search_dups(dir_input, user_input)
