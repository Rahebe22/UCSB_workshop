import os
#import dask.dataframe as dd
from datetime import datetime

from torch.utils.data import Dataset, DataLoader

from utils import *
from augmentation import *
import pdb
from IPython.core.debugger import set_trace


class ImageData(Dataset):
    """
    Custom Dataset for loading image files (typically Planet images)
    for PyTorch architecture.
    """

    def __init__(self, data_path, log_dir, catalog, data_size, buffer, buffer_comp, usage, 
                 img_path_cols, label_path_col=None, label_group=[0, 1, 2, 3, 4], apply_normalization=True, 
                 normal_strategy="min_max", stat_procedure="lab", global_stats=None, catalog_index=None, 
                 trans=None, parallel=False, **kwargs):

        """
        Initialize ImageData instance.
        
        Args:
            data_path (str): Directory storing files of variables and labels.
            log_dir (str): Directory to save the log file.
            catalog (Pandas.DataFrame): Pandas dataframe giving the list of data 
                    and their directories
            data_size (int): Size of chips that is not buffered (i.e., the size of labels)
            buffer (int): Distance to target chips' boundaries measured by 
                    number of pixels when extracting images (variables), i.e., 
                    variables size would be (dsize + 2*buffer) x (dsize + 2*buffer)
            buffer_comp (int): Buffer used when creating composite. In the case of Ghana, 
                   it is 11.
            usage (str): Usage of the dataset : "train", "validate" or "predict"
            img_path_cols (Union[List[str], str]): Column names in the catalog referring to image paths
            label_path_col (Optional[str] = None): Column name in the catalog referring to label paths.
                   Check the note.
            label_group (List[int]): Group indices of labels to load, where each group corresponds 
                    to a specific level of label quality
            apply_normalization (bool): whether to apply normalization or not
            normal_strategy (str): Strategy for normalization. Either 'min_max'
                    or 'z_value'.
            stat_procedure (str): Procedure to calculate the statistics used in normalization.
                        Options:
                            - 'lab': local tile over all bands.
                            - 'gab': global over all bands.
                            - 'lpb': local tile per band.
                            - 'gpb': global per band.
            global_stats (dict): Optional dictionary containing the 'min', 'max', 'mean', and 'std' arrays 
                    for each band. Required for 'gab' and 'gpb' stat procedures.
            catalog_index (Optional[int]): Row index in catalog to load data for prediction. Only need to 
                    be specified when usage is "prediction"
            trans (Optional[List[str]]): Data augmentation methods: one or multiple elements from the list of 
                    provided transformations:
                        1) 'vflip', vertical flip
                        2) 'hflip', horizontal flip
                        3) 'dflip', diagonal flip
                        4) 'rotate', rotation
                        5) 'resize', rescale image fitted into the specified data size
                        6) 'shift_brightness', shift brightness of images
                Any value out of the range would cause an error
            parallel (bool): Whether to read and load the input data in parallel (True) or not (False)
            **kwargs (dict, optional): Additional parameters for the specified transformation methods. 
                        These can include:
                            - scale_factor (tuple): Scaling factor for the 'resize' transformation 
                                (default is (0.75, 1.5)).
                            - crop_strategy (str): Either "center" or "random"
                            - rotation_degree (tuple): Degree range for 'rotate' transformation 
                                (default is (-180, -90, 90, 180).
                            - bshift_subs (tuple): Band subsets for 'shift_brightness' transformation 
                                (default is [4]).
                            - bshift_gamma_range (tuple): Gamma range for 'shift_brightness' transformation 
                                (default is (0.2, 2.0)).
                            - patch_shift (bool): Whether to apply patch shift for 'shift_brightness' transformation 
                                (default is True) 
                            - downfactor (int): The total downsampling factor in the network, used to
                                check that image chips are evenly divisible by that number
                            - clip_val: threshold to clip the tails of image value distribution
                            - nodata: list of values reserved for nodata
                       
        Note:
            - Catalog for train and validate contrains at least columns for image
                path, label path and "usage".
            - Catalog for prediction contains at least columns for image path, 
                "tile_col", and "tile_row", where the "tile_col" and "tile_row" is 
                the relative tile location for naming predictions in Learner.
        """

        self.data_path = data_path
        self.log_dir = log_dir
        self.data_size = data_size
        self.buffer = buffer
        self.composite_buffer = buffer_comp
        self.usage = usage
        self.img_cols = img_path_cols if isinstance(img_path_cols, list) \
            else [img_path_cols]
        self.label_col = label_path_col
        self.apply_normalization = apply_normalization
        self.normal_strategy = normal_strategy
        self.stat_procedure = stat_procedure
        self.global_stats = global_stats
        self.trans = trans
        self.parallel = parallel
        self.kwargs = kwargs
        self.downfactor = self.kwargs.get("downfactor", 32)
        self.nodata = self.kwargs.get("nodata", None)
        self.clip_val = self.kwargs.get("clip_val", None)
        self.chip_size = self.data_size + self.buffer * 2
        #print(f"chip_size: {self.chip_size}")
        
        # Initialize logger
        self.logger = setup_logger(self.log_dir, f"{self.usage}_dataset_processing_report", use_date=False)
        start = datetime.now()
        msg = f'Started dataset creation process for {self.usage} at: {start}'
        progress_reporter(msg, verbose=False, logger=self.logger)      

        if self.usage in ["train", "validate", "test"]:
            self.catalog = catalog.loc[
                (catalog['usage'] == self.usage) & 
                (catalog['label_group'].isin(label_group))].copy()
            
            self.img, self.label = self.get_train_validate_data()
            print(f'----------{len(self.img)} samples loaded in {self.usage} dataset-----------')
        
        elif self.usage == "predict":

            self.catalog = catalog.iloc[catalog_index].copy()
            self.year = self.catalog["year"]
            self.tile = self.catalog['tile']

            #self.tile = (self.catalog['tile_col'], self.catalog['tile_row'])
            #self.year = self.img_cols[0].split("_")[1].split("-")[0]
            #self.year = self.catalog["dir_img"].split("/")[0]
            #self.year = self.catalog["year"]
            self.img, self.index, self.meta = self.get_predict_data()
            print(f'----------Prediction data loaded for tile {self.tile}, year {self.year}-----------')
        
        else:
            raise ValueError("Bad usage value")
        
        end = datetime.now()
        msg = f'Completed dataset creation process for {self.usage} at: {end}'    
        progress_reporter(msg, verbose=False, logger=self.logger)
        
        if self.usage in ['train', 'validate','test']:
            total_samples_msg = f'Total number of samples in {self.usage} dataset: {len(self.img)}'
            print(total_samples_msg)
            progress_reporter(total_samples_msg, verbose=False, logger=self.logger)
    
    def get_train_validate_data(self):
        '''
        Get paris of image, label for train and validation
        Returns:
            tuple of list of images and label
        '''
        def load_label(row, data_path):
            buffer = self.buffer

            dir_label = os.path.join(data_path, row[self.label_col])
            label = load_data(dir_label, usage=self.usage, is_label=True)
            label = np.pad(label, buffer, 'constant')
            msg = f'.. processing lbl sample: {os.path.basename(dir_label)}'\
                f' with shape {label.shape} is complete.'
            progress_reporter(msg, verbose=False, logger=self.logger)

            return label

        def load_img(row, data_path):
            #print(f"Processing row: {row}")
            buffer = self.buffer

            dir_label = os.path.join(data_path, row[self.label_col])
            dir_imgs = [os.path.join(data_path, row[m]) for m in self.img_cols]

            print(dir_imgs)

            window = get_buffered_window(dir_imgs[0], dir_label, buffer)
            img = process_img(dir_imgs, self.usage, apply_normalization=self.apply_normalization, 
                              normal_strategy=self.normal_strategy, stat_procedure=self.stat_procedure,
                              global_stats=self.global_stats, window=window, nodata_val_ls=self.nodata, clip_val=self.clip_val)

            msg = f'.. processing img sample: {os.path.basename(dir_imgs[0])}'\
                f' with shape {img.shape} is complete.'
            progress_reporter(msg, verbose=False, logger=self.logger)

            return img

        if self.parallel:
            
            global list_data  # Local function not applicable in parallelism
            def list_data(catalog, data_path):
                catalog["img"] = catalog.apply(
                    lambda row: load_img(row, data_path), axis=1
                )
                catalog["label"] = catalog.apply(
                    lambda row: load_label(row, data_path), axis=1
                )
                return catalog.filter(items=['label', 'img'])
    
            catalog = parallelize_df(self.catalog, list_data, 
                                     data_path=self.data_path)
            
            img_ls = catalog['img'].tolist()
            label_ls = catalog['label'].tolist()
        
        else:
            self.catalog["img"] = self.catalog.apply(
                lambda row: load_img(row, data_path=self.data_path), axis=1
            )
            self.catalog["label"] = self.catalog.apply(
                lambda row: load_label(row, data_path=self.data_path), axis=1
            )
            img_ls = self.catalog['img'].tolist()
            label_ls = self.catalog['label'].tolist()
        
        return img_ls, label_ls

    
    def get_predict_data(self):
        """
        Get data for prediction
        Returns:
            list of cropped chips
            list of index representing location of each chip in tile
            dictionary of metadata of score map reconstructed from chips
        """
       
        dir_imgs = [os.path.join(self.data_path, self.catalog[m]) \
                    for m in self.img_cols]
        img = process_img(dir_imgs, self.usage, apply_normalization=self.apply_normalization, 
                          normal_strategy=self.normal_strategy, stat_procedure=self.stat_procedure,
                          global_stats=self.global_stats, nodata_val_ls=self.nodata, clip_val=self.clip_val)
          
        buffer_diff = self.buffer - self.composite_buffer
        h, w, c = img.shape
        
        #print(f"buffer difference: {buffer_diff}")
        #print(f"image size: {img.shape}")

        if buffer_diff > 0:
            canvas = np.zeros((h + buffer_diff * 2, w + buffer_diff * 2, c))

            for i in range(c):
                canvas[:, :, i] = np.pad(img[:, :, i], buffer_diff, 
                                         mode='reflect')
            img = canvas
            
            #print(f"canvas size: {canvas.shape}")

        else:
            buffer_diff = abs(buffer_diff)
            img = img[buffer_diff:h - buffer_diff, 
                      buffer_diff:w - buffer_diff, :]

        # meta of composite buffer removed
        meta = get_meta_from_bounds(dir_imgs[0], self.composite_buffer)
        img_ls, index_ls = get_chips(img, self.chip_size, self.buffer)
        
        #print(meta)
        
        if img_ls[0].shape[0] % self.downfactor != 0:
            assert f"Chip is not evenly divisible by {self.downfactor}"

        return img_ls, index_ls, meta

    
    def __getitem__(self, index):
        """
        Support dataset indexing and apply transformation
        Args:
            index -- Index of each small chips in the dataset
        Returns:
            tuple
        """

        start = datetime.now()
        msg = f'Started processing index {index} for {self.usage} at: {start}'
        progress_reporter(msg, verbose=False, logger=self.logger)
        
        #set_trace()
        if self.usage in ["train", "validate", "test"]:
            img = self.img[index]
            label = self.label[index]

            if self.usage in "train":               
                mask = np.pad(np.ones((self.data_size, self.data_size)),
                              self.buffer, 'constant')
                
                if self.trans:
                    
                    # 0.5 possibility to flip
                    trans_flip_ls = [m for m in self.trans if 'flip' in m]
                    if random.randint(0, 1) and len(trans_flip_ls) >= 1:
                        trans_flip = random.sample(trans_flip_ls, 1)
                        img, label, mask = flip(img, label, mask, 
                                                trans_flip[0])
                    
                    # 0.5 possibility to rotate
                    if random.randint(0, 1) and 'rotate' in self.trans:
                        deg_rotate = self.kwargs.get("rotation_degree", (-180, -90, 90, 180))
                        img, label, mask = center_rotate(img, label, mask, 
                                                        deg_rotate)
                    
                    # 0.5 possibility to resize
                    if random.randint(0, 1) and 'resize' in self.trans:
                        scale_factor = self.kwargs.get("scale_factor", (0.75, 1.5))
                        crop_strategy = self.kwargs.get("crop_strategy", "center")
                        img, label, mask = re_scale(
                            img, label.astype(np.uint8), mask.astype(np.uint8),
                            scale=scale_factor, crop_strategy=crop_strategy)
                    
                    # # 0.5 possibility to shift brightness
                    # if random.randint(0, 1) and 'shift_brightness' in self.trans:
                    #     bshift_gamma_range = self.kwargs.get("bshift_gamma_range", (0.2, 2.0))
                    #     shift_subset = self.kwargs.get("shift_subset", [4])
                    #     patch_shift = self.kwargs.get("patch_shift", True)
                    #     img = shift_brightness(img, gamma_range=bshift_gamma_range, 
                    #                            shift_subset=shift_subset, patch_shift=patch_shift)
                    
                    # 0.5 possibility to brightness manipulation
                    trans_br_ls = [m for m in self.trans if 'br_' in m]
                    if random.randint(0, 1) and len(trans_br_ls) >= 1:
                        trans_br = random.sample(trans_br_ls, 1)
                        sigma_range = self.kwargs.get("sigma_range", [0.03, 0.07])
                        br_range = self.kwargs.get("br_range", [-0.02, 0.02])
                        contrast_range = self.kwargs.get("contrast_range", [0.9, 1.2])
                        bshift_gamma_range = self.kwargs.get("bshift_gamma_range", (0.2, 2.0))
                        shift_subset = self.kwargs.get("shift_subset", [4])
                        patch_shift = self.kwargs.get("patch_shift", True)
                        img = br_manipulation(img, trans_br[0], sigma_range, br_range, contrast_range, 
                                              bshift_gamma_range, shift_subset, patch_shift)
                    
                # numpy to torch
                label = torch.from_numpy(label.copy()).long()
                mask = torch.from_numpy(mask.copy()).long()
                img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()

                ret_tuple = (img, label, mask)
            
            else:
                label = torch.from_numpy(label.copy()).long()
                img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()

                ret_tuple = (img, label)

        else:
            img = self.img[index]
            index = self.index[index]
            
            img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()

            ret_tuple = (img, index, self.tile, self.year)
        
        end = datetime.now()
        msg = f'Completed processing index {index} for {self.usage} at: {end}'
        progress_reporter(msg, verbose=False, logger=self.logger)
        
        return ret_tuple

    def __len__(self):
        """
        Get size of the dataset
        """

        return len(self.img)