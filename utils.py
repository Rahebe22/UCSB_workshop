import numpy as np
import pandas as pd
import os
import math
import random
import itertools
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
import logging
import pickle
from datetime import datetime

import torch
from normalization import do_normalization


def load_data(data_path, usage, nodata_val_ls=[], is_label=False, apply_normalization=True, 
              normal_strategy="z_value", stat_procedure="gpb", global_stats=None, 
              dtype=np.float64, window=None, clip_val=None):
    """
    Open data using gdal, read it as an array and normalize it.

    Args:
        data_path (string): Full path including filename of the data source we wish to load.
        usage (string): Either "train", "validation", "inference".
        nodata_val_ls (list of floats, optional): The 'no data' values. Pixels with these values are not used in 
                                  normalization. If None, defaults to the nodata attribute in the dataset.
        is_label (bool): If True then the layer is a ground truth (category index) and if
                         set to False the layer is a reflectance band.
        apply_normalization (bool): If true min/max normalization will be applied on each band.
        normal_strategy (str): Strategy for normalization. Either 'min_max' or 'z_value'.
        stat_procedure (str): Procedure to calculate the statistics used in normalization.
                              Options:
                                     - 'lab': local tile over all bands.
                                     - 'gab': global over all bands.
                                     - 'lpb': local tile per band.
                                     - 'gpb': global per band.
        global_stats (dict, Optional): dictionary containing the 'min', 'max', 'mean', and 'std' arrays 
                                       for each band. If not provided, these values will be calculated 
                                       from the data.
        dtype (np.dtype): Data type of the output image chips.
        window (tuple, optional): A tuple defining the window to read from the data. Format is 
                                  (col_off, row_off, width, height). If None, the whole image is read.
        clip_val (float, optional): Maximum allowed value after normalization. Pixels with values 
                                    higher than this are set to this value. If zero, no clipping is applied.

    Returns:
            image (np.ndarray): A numpy array representing the loaded and processed image data. The array will 
                                have the specified dtype, and normalization will be applied if requested.
    """

    # open dataset using rasterio library.
    with rasterio.open(data_path, "r") as src:

        if is_label:
            if src.count != 1:
                raise ValueError("Expected Label to have exactly one channel.")
            img = src.read(1)
            return img

        else:
            img_nodata = src.nodata
            nodata_val_ls = list(set(nodata_val_ls + [img_nodata])) if nodata_val_ls else [img_nodata]
            
            if apply_normalization:
                img = do_normalization(src.read(), normal_strategy, stat_procedure, 
                                       nodata=nodata_val_ls, clip_val=clip_val, global_stats=global_stats)
                img = img.astype(dtype)
            else:
                img = src.read()
                img = img.astype(dtype)

    if usage in ["train", "validate"]:
        img = img[:, max(0, window[1]): window[1] + window[3], 
                  max(0, window[0]): window[0] + window[2]]
    
    return img


def process_img(img_paths, usage, apply_normalization=False, normal_strategy="z_value", 
                stat_procedure="gpb", global_stats=None, window=None, nodata_val_ls=[], 
                clip_val=None, dtype=np.float64):
    """
    Process and normalize satellite image data from single or multiple timepoints.
    
    Args:
        img_paths (list of str): List of file paths to the images to be processed.
        usage (str): Specifies the usage of the images. Options are "train", "validate", 
                     or "predict".
        apply_normalization (bool): If True, normalization is applied to the images according 
                                    to the specified strategy.
        normal_strategy (str): Normalization strategy to be used. Options are "min_max" or "z_value".
        stat_procedure (str): Specifies the statistical procedure used in normalization. Options are
                              'lab', 'gab', 'lpb', and 'gpb'.
        global_stats (dict, optional): Global statistics for normalization (min, max, mean, std) 
                                       for each band.
        window (tuple, optional): Tuple specifying the window (col_offset, row_offset, width, height) 
                                  for cropping the images.
        nodata_val_ls (float, optional): Values representing 'no data' in the images.
        clip_val (float, optional): Value at which to clip the normalized data.
    
    Returns:
        np.ndarray: A multi-dimensional numpy array representing the processed image data.
                    The array's shape and contents vary based on the 'usage' parameter.
    """

    if len(img_paths) > 1:
        img_ls = [load_data(m, usage, apply_normalization=apply_normalization, 
                            normal_strategy=normal_strategy, stat_procedure=stat_procedure, 
                            global_stats=global_stats, window=window, nodata_val_ls=nodata_val_ls, 
                            clip_val=clip_val, dtype=dtype) 
                  for m in img_paths]
        img = np.concatenate(img_ls, axis=0).transpose(1, 2, 0)
    else:
        img = load_data(img_paths[0], usage, apply_normalization=apply_normalization, 
                        normal_strategy=normal_strategy, stat_procedure=stat_procedure, 
                        global_stats=global_stats, window=window, nodata_val_ls=nodata_val_ls, 
                        clip_val=clip_val, dtype=dtype).transpose(1, 2, 0)

    if usage in ["train", "validate","test"]:
        col_off, row_off, col_target, row_target = window
        row, col, c = img.shape

        if row < row_target or col < col_target:

            row_off = abs(row_off) if row_off < 0 else 0
            col_off = abs(col_off) if col_off < 0 else 0

            canvas = np.zeros((row_target, col_target, c))
            canvas[row_off: row_off + row, col_off : col_off + col, :] = img

            return canvas

        else:
            return img

    elif usage == "predict":
        return img

    else:
        raise ValueError


def get_buffered_window(src_path, dst_path, buffer):
    """
    Get bounding box representing subset of source image that overlaps with 
    bufferred destination image, in format of (column offsets, row offsets,
    width, height)

    Argss:
        src_path (str): Path of source image to get subset bounding box
        dst_path (str): Path of destination image as a reference to define the 
            bounding box. Size of the bounding box is
            (destination width + buffer * 2, destination height + buffer * 2)
        buffer (int): Buffer distance of bounding box edges to destination image 
            measured by pixel numbers

    Returns:
        tuple in form of (column offsets, row offsets, width, height)
    """

    if buffer < 0 or not isinstance(buffer, int):
        raise ValueError("Buffer should be a non-negative integer.")
    
    try:
        with rasterio.open(src_path, "r") as src:
            gt_src = src.transform
    except Exception as e:
        raise IOError(f"Could not open source file: {e}")
    
    try:
        with rasterio.open(dst_path, "r") as dst:
            gt_dst = dst.transform
            w_dst = dst.width
            h_dst = dst.height
    except Exception as e:
        raise IOError(f"Could not open destination file: {e}")

    # column and row offsets by finding the difference in the upper-left corner 
    # coordinates of the source and destination images, divided by their respective 
    # pixel resolutions  in x and y directions, respectively
    col_off = round((gt_dst[2] - gt_src[2]) / gt_src[0]) - buffer
    row_off = round((gt_dst[5] - gt_src[5]) / gt_src[4]) - buffer
    width = w_dst + buffer * 2
    height = h_dst + buffer * 2

    return col_off, row_off, width, height


def get_meta_from_bounds(file_path, buffer):
    """
    Get metadata of unbuffered region in given file
    
    Args:
        file_path (str):  Path to an image chip file.
        buffer (int): Buffer distance measured by pixel numbers
    
    Returns:
        dictionary: updated metadata
    """

    if buffer < 0:
        raise ValueError("Buffer distance should be non-negative.")
    
    try:
        with rasterio.open(file_path, "r") as src:
            meta = src.meta
            dst_width = src.width - 2 * buffer
            dst_height = src.height - 2 * buffer
            
            if dst_width <= 0 or dst_height <= 0:
                raise ValueError("Buffer size is too large, resulting in zero dimension(s).")
            
            window = rasterio.windows.Window(buffer, buffer, dst_width, dst_height)
            win_transform = src.window_transform(window)
    
    except Exception as e:
        raise IOError(f"Could not open source file: {e}")

    meta.update({
        'width': dst_width,
        'height': dst_height,
        'transform': win_transform,
        'count': 1,
        'nodata': -128,
        'dtype': 'int8'
    })

    return meta


def get_chips(img, dsize, buffer):
    """
    Generate small chips from input images and the corresponding index of each 
    chip The index marks the location of corresponding upper-left pixel of a 
    chip.
    
    Args:
        img (np.ndarray): Image in format of (H,W,C) to be cropped. in this case it is 
            the concatenated image of growing season and off season
        dsize (int): Cropped chip size
        buffer (int):Number of overlapping pixels when extracting images chips
    
    Returns:
        list of cropped chips and corresponding coordinates (Tuple[List[np.ndarray], List[Tuple[int, int]]])
    
    Note: 
    The input image can be a concatanation of more than 1 time-point like growing season and off season.
    """

    h, w, _ = img.shape
    x_ls = range(0,h - 2 * buffer, dsize - 2 * buffer)
    y_ls = range(0, w - 2 * buffer, dsize - 2 * buffer)

    index = list(itertools.product(x_ls, y_ls))

    img_ls = []
    for i in range(len(index)):
        x, y = index[i]
        img_ls.append(img[x:x + dsize, y:y + dsize, :])

    return img_ls, index


def make_reproducible(seed=42, cudnn=True):
    """
    Configures the environment to make the execution of the program deterministic, 
    with a fixed seed for random number generators in various libraries which helps 
    in reproducibility of the results.
    
    Args:
        seed (int, optional): The seed value to be used for random number generators. 
                              Defaults to 42.
        cudnn (bool, optional): If True, ensures that CUDA's cuDNN backend behaves 
                                deterministically. This might impact performance. 
                                Defaults to True.
    Note:
        - This function sets seeds for Python's `random` module, NumPy, and PyTorch (both CPU and CUDA).
        - Setting 'PYTHONHASHSEED' ensures that Python hashes are reproducible between runs.
        - Enabling cuDNN determinism might lead to a performance degradation or limit the ability to use 
          certain cuDNN algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cudnn:
        torch.cuda.manual_seed_all(seed)
        #torch.backends.cudnn.deterministic = True


def pickle_dataset(dataset, file_path):
    """
    Serializes and saves a dataset object that is pickle-able to a file using Python's pickle module.
    This is particularly useful for saving pre-processed datasets for later use.
    
    Args:
        dataset: The dataset object to be serialized and saved.
        file_path (str): The path of the file where the dataset will be stored. This should include 
                         the filename and extension, typically '.pkl' or '.pickle'.
    """
    with open(file_path, "wb") as fp:
        pickle.dump(dataset, fp)


def load_pickle(file_path):
    """
    Loads a Python object from a pickle file.
    
    Args:
        file_path (str): The path to the pickle file containing the serialized Python object.
    """
    return pd.read_pickle(file_path)


def progress_reporter(msg, verbose, logger=None):
    """
    Helps control print statements and log writes
    
    Args:
        msg (str): Message to write out
        verbose (bool): Prints or not to console logger (logging.logger)
        logger (defaults to none)
      
    Returns: 
        Message to console and or log
    """
    
    if verbose:
        print(msg)

    if logger:
        logger.info(msg)


def setup_logger(log_dir, log_name, use_date=False):
    """
    Create logger
    """
    if use_date:
        dt = datetime.now().strftime("%d%m%Y_%H%M")
        log = "{}/{}_{}.log".format(log_dir, log_name, dt)
    else: 
        log = "{}/{}.log".format(log_dir, log_name)
        
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_format = (
        f"%(asctime)s::%(levelname)s::%(name)s::%(filename)s::"
        f"%(lineno)d::%(message)s"
    )
    logging.basicConfig(filename=log, filemode='w',
                        level=logging.INFO, format=log_format)
    
    return logging.getLogger()


