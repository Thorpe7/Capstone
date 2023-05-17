'''
Takes dataset and currates it for adequate use w/ pytorch dataset & loader

'''
import pathlib
from pathlib import Path
import pandas as pd
import logging as log
import shutil
from PIL import Image

log.basicConfig(level=log.INFO)
log = log.getLogger(__name__)

def curate_torch_csv(data_dir:pathlib.PosixPath, testing:bool=False) -> None:
    '''
    Takes in a string path of data dir. Iterates through all sub directories.
    Creates w/ associated label. Also creates directory of all files in one directory.

    '''
    log.info("Data curation started...")
    label = 0
    df_list = []
    if not data_dir.is_dir(): 
        log.error(f"{data_dir} is not valid directory...")


    # Iterate through data dir
    for item in data_dir.iterdir():
        tmp_df = pd.DataFrame()
        if item.is_dir():
            if not testing:
                file_paths = [x for x in item.glob("*.*") if x.is_file()]
            else:
                file_paths = [x for x in item.glob("*.*") if x.is_file()]
            tmp_df["images"] = file_paths
            tmp_df["label"] = label

        label +=1
        df_list.append(tmp_df)
    
    final_df = pd.concat(df_list, axis=0, ignore_index=True)
    final_df.to_csv(f"{data_dir}/{data_dir.name}.csv",index=False)


def find_smallest_img_dim(path_to_dir:pathlib.PosixPath) -> tuple:
    '''

    Iterates through data directory and all subdirectories and finds smallest
    image dimesions to be used for image transformations. 
    
    '''
    min_dim = (3000,3000)
    for item in path_to_dir.iterdir():
        
        if item.is_dir():
                min_dim = find_smallest_img_dim(item)
        else:
            if isinstance(item,pathlib.PosixPath):
                if item.is_file() and item.suffix == ".jpg":
                    img = Image.open(item)
                    img.size
                    if img.size < min_dim:
                        min_dim = img.size
        
                
    return min_dim

