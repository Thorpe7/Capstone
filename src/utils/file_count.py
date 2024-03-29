""" Checking the number of files in data directories"""
from pathlib import Path


def get_file_count(dir_path: str):
    count = 0
    data_dir = Path(dir_path)
    if data_dir.exists():
        for x in data_dir.iterdir():
            if x.is_file():
                count += 1
            elif x.is_dir():
                count += get_file_count(x)
    return count


if __name__ == "__main__":
    user_input = input("Provide file path here: ")
    total_count = get_file_count(user_input)
    print(total_count)
