import cv2
import pytesseract
from PIL import Image
import numpy as np
from loguru import logger
import pandas as pd

class IonogramTableExtractor:
    """
    IonogramTableExtractor is a utility class for extracting and parsing tabular data from ionogram images using OCR.
    Attributes:
        image_path (str): Path to the input image file.
        image (numpy.ndarray): Loaded image as a NumPy array.
    Methods:
        __init__(image_path):
            Initializes the extractor by loading the image from the specified path.
            Raises FileNotFoundError if the image cannot be loaded.
        extract_table_text(x=0, y=30, w=150, h=None, ocr_config='--oem 3 --psm 3'):
            Crops a region of interest (ROI) from the image, applies adaptive thresholding,
            and performs OCR to extract text from the table area.
            Args:
                x (int): X-coordinate of the top-left corner of the ROI. Default is 0.
                y (int): Y-coordinate of the top-left corner of the ROI. Default is 30.
                w (int): Width of the ROI. Default is 150.
                h (int or None): Height of the ROI. If None, uses the remaining height from y.
                ocr_config (str): Configuration string for Tesseract OCR.
            Returns:
                str: Raw OCR output as a string.
        parse_table(text):
            Parses the OCR-extracted text to extract key-value pairs from lines containing ':'.
            Args:
                text (str): Raw OCR output.
            Returns:
                dict: Dictionary mapping keys to values extracted from the table.
    Usage:
        extractor = IonogramTableExtractor('/path/to/image.png')
        text = extractor.extract_table_text()
        table_data = extractor.parse_table(text)
    """
    def __init__(self, image_path, verbose=False):
        self.image_path = image_path
        self.verbose = verbose
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            if self.verbose:
                logger.error(f"Failed to load image: {self.image_path}")
            raise FileNotFoundError(f"Image not found: {self.image_path}")

    def extract_table_text(self, x=0, y=30, w=150, h=None, ocr_config=r'--oem 3 --psm 3'):
        if h is None:
            h = self.image.shape[0] - y
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        table_crop = gray[y:y+h, x:x+w]
        thresh = cv2.adaptiveThreshold(
            table_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        text = pytesseract.image_to_string(thresh, config=ocr_config)
        if self.verbose:
            logger.info("Raw OCR output:\n" + text)
        return text

    def parse_table(self, text:str, word_filtes_for_table_values: dict = {"N/A": "nan", ":": "."},):
        table_data = {}
        keys = [
            "foF2", "foFl", "foF1p", "foE",
            "foEp", "fxI", "foEs", "fmin",
            "MUF(D)", "M(D)", "D", "h`F",
            "h`F2","h`E","h`Es", "hmF2",
            "hmF1", "hmE",
        ]
        location = [1]*len(keys)
        location[15] = 2
        lines = [t for t in text.splitlines() if len(t.strip()) > 0]
        for line, key, loc in zip(lines, keys, location):
            words = list(filter(None, line.split(" ")))
            if self.verbose:
                print(key, loc, words)
            if len(words) >= 2:
                for fw in word_filtes_for_table_values.keys():
                        words[1] = words[1].replace(
                            fw, word_filtes_for_table_values[fw]
                        )
                try:
                    if loc==2 and len(words) == 2:
                        loc = 1
                    table_data[key] = float(words[loc])
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"Could not convert '{words}' to float for key '{key}'")
                    table_data[key] = np.nan
        return pd.DataFrame.from_records([table_data])

    @staticmethod
    def extract_ionogram_table(image_path, verbose=False):
        extractor = IonogramTableExtractor(image_path, verbose=verbose)
        ocr_text = extractor.extract_table_text()
        table = extractor.parse_table(ocr_text)
        return table

if __name__ == "__main__":
    image_path = "tmp/ion_BC840_000505.png"
    extractor = IonogramTableExtractor(image_path)
    ocr_text = extractor.extract_table_text()
    table = extractor.parse_table(ocr_text)
    print(table)