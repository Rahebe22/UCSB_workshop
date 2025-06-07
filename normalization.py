import numpy as np

def do_normalization(img, normal_strategy, stat_procedure, nodata=[0, 65535], 
                     clip_val=0, global_stats=None, epsilon=1e-6):
    """
    Normalize the input image pixels to a user-defined range based on the
    minimum and maximum statistics of each band and optional clip value.

    Args:
        img (np.ndarray): Stacked image bands with a dimension of (C, H, W).
        normal_strategy (str): Strategy for normalization. Either 'min_max'
                               or 'z_value'.
        stat_procedure (str): Procedure to calculate the statistics used in normalization.
                              Options:
                                    - 'lab': local tile over all bands.
                                    - 'gab': global over all bands.
                                    - 'lpb': local tile per band.
                                    - 'gpb': global per band.
        nodata (list): Values reserved to show nodata.
        clip_val (float): Defines how much of the distribution tails to be cut off.
                          Default is 0.
        global_stats (dict): Optional dictionary containing the 'min', 'max', 'mean', and 'std' arrays
                             for each band. If not provided, these values will be calculated from the data.
        epsilon (float): A small constant added to the denominator to prevent division by zero.

    Returns:
        np.ndarray: Normalized image stack of size (C, H, W).
    """
    if normal_strategy not in ["min_max", "z_value"]:
        raise ValueError("Normalization strategy is not recognized.")
    
    if stat_procedure not in ["gpb", "lpb", "gab", "lab"]:
        raise ValueError("Statistics calculation strategy is not recognized.")
    
    if stat_procedure in ["gpb", "gab"] and global_stats is None:
        raise ValueError("Global statistics must be provided for global normalization.")

    # Create a mask for nodata values and replace them with nan for computation.
    img_tmp = np.where((img < 0) | np.isin(img, nodata), np.nan, img)

    if normal_strategy == "min_max":
        if clip_val > 0:
            lower_percentiles = np.nanpercentile(img_tmp, clip_val, axis=(1, 2))
            upper_percentiles = np.nanpercentile(img_tmp, 100 - clip_val, axis=(1, 2))
            for b in range(img.shape[0]):
                img[b] = np.clip(img[b], lower_percentiles[b], upper_percentiles[b])

        if stat_procedure == "gpb":
            gpb_mins = np.array(global_stats['min'])
            gpb_maxs = np.array(global_stats['max'])
            diff = gpb_maxs - gpb_mins
            diff[diff == 0] = epsilon
            normal_img = (img - gpb_mins[:, None, None]) / diff[:, None, None]
            normal_img = np.clip(normal_img, 0, 1)

        elif stat_procedure == "gab":
            gab_min = np.mean(global_stats['min'])
            gab_max = np.mean(global_stats['max'])
            if gab_max == gab_min:
                gab_max += epsilon
            normal_img = (img - gab_min) / (gab_max - gab_min)
            normal_img = np.clip(normal_img, 0, 1)

        elif stat_procedure == "lab":
            lab_min = np.nanmin(img_tmp)
            lab_max = np.nanmax(img_tmp)
            if lab_max == lab_min:
                lab_max += epsilon
            normal_img = (img - lab_min) / (lab_max - lab_min)
            normal_img = np.clip(normal_img, 0, 1)

        else:  # stat_procedure == "lpb"
            lpb_mins = np.nanmin(img_tmp, axis=(1, 2))
            lpb_maxs = np.nanmax(img_tmp, axis=(1, 2))
            diff = lpb_maxs - lpb_mins
            diff[diff == 0] = epsilon
            normal_img = (img - lpb_mins[:, None, None]) / diff[:, None, None]
            normal_img = np.clip(normal_img, 0, 1)

    elif normal_strategy == "z_value":
        if stat_procedure == "gpb":
            gpb_means = np.array(global_stats['mean'])
            gpb_stds = np.array(global_stats['std'])
            gpb_stds[gpb_stds == 0] = epsilon
            normal_img = (img - gpb_means[:, None, None]) / gpb_stds[:, None, None]

        elif stat_procedure == "gab":
            gab_mean = np.mean(global_stats['mean'])
            gab_std = np.sqrt(np.sum((global_stats['std'] ** 2) * img.shape[1] * img.shape[2]) / 
                              (img.shape[1] * img.shape[2] * len(global_stats['std'])))
            if gab_std == 0:
                gab_std += epsilon
            normal_img = (img - gab_mean) / gab_std

        elif stat_procedure == "lpb":
            img_means = np.nanmean(img_tmp, axis=(1, 2))
            img_stds = np.nanstd(img_tmp, axis=(1, 2))
            img_stds[img_stds == 0] = epsilon
            normal_img = (img - img_means[:, None, None]) / img_stds[:, None, None]

        elif stat_procedure == "lab":
            img_mean = np.nanmean(img_tmp)
            img_std = np.nanstd(img_tmp)
            if img_std == 0:
                img_std += epsilon
            normal_img = (img - img_mean) / img_std

    return normal_img
