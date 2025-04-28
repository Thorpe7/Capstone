""" Analysis and review of imaging data distributions"""
import pathlib
import pandas as pd
import numpy as np
import cv2
import logging

from pathlib import Path

# Init logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log = logger


class ImageDataProcessor:
    def __init__(self, data_dir: pathlib.PosixPath):
        self.data_dir = data_dir
        self.file_list = Path(data_dir).glob("**/*.jpg")
        self.data_frame = pd.DataFrame()

    def data_to_dataframe(self):
        self.dir_data_to_df(self.data_dir)

    def get_image_characteristics(self, image_path: pathlib.PosixPath) -> dict:
        """Get image attributes from image object

        Args:
            image_path (pathlib.PosixPath): Image file to be processed

        Returns:
            dict: Dictionary of image attributes
        """
        img = cv2.imread(str(image_path))
        height, width, channels = img.shape
        mean_intensity = img.mean()
        std_intensity = np.std(img)
        image_attributes = {
            "image_path": image_path,
            "height": height,
            "width": width,
            "channels": channels,
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "class": image_path.parent.name,
        }

        return image_attributes

    def dir_data_to_df(self, parent_dir: pathlib.PosixPath):
        """Iterate through directories and extract image attributes
        Modifies class' dataframe attribute

        Args:
            parent_dir (pathlib.PosixPath): Top level data directory
        """
        for path_obj in parent_dir.iterdir():
            if path_obj.is_dir():
                self.dir_data_to_df(path_obj)
            elif path_obj.is_file() and path_obj.suffix == ".jpg":
                image_attributes = self.get_image_characteristics(path_obj)
                tmp_df = pd.DataFrame([image_attributes])
                self.data_frame = pd.concat(
                    [self.data_frame, tmp_df], ignore_index=True
                )

            else:
                log.info(f"Unknown or incorrect file type, ignored: {path_obj}...")

    def save_df_to_csv(self, output_dir: pathlib.PosixPath):
        """Save dataframe to csv file

        Args:
            output_dir (pathlib.PosixPath): Output directory for csv file
        """
        self.data_frame.to_csv(f"{output_dir}/image_attributes.csv", index=False)


if __name__ == "__main__":
    TrainingDataProcessor = ImageDataProcessor(
        Path("/home/thorpe/git_repos/Capstone/data/v1/Training")
    )
    TrainingDataProcessor.data_to_dataframe()
    TrainingDataProcessor.save_df_to_csv(
        Path("/home/thorpe/git_repos/Capstone/results/eda/training")
    )

    TestingDataProcessor = ImageDataProcessor(
        Path("/home/thorpe/git_repos/Capstone/data/v1/Testing")
    )
    TestingDataProcessor.data_to_dataframe()
    TestingDataProcessor.save_df_to_csv(
        Path("/home/thorpe/git_repos/Capstone/results/eda/testing")
    )
