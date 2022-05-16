# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os
import cv2
import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class Kaggle_Dataset(CustomDataset):
    """Kaggle_Dataset.
    """
    CLASSES = ('background','large_bowel', 'small_bowel', 'stomach')

    PALETTE = [[0,0,0], [64,64,64],[128,128,128], [255,255,255]]

    def __init__(self, **kwargs):
        super(Kaggle_Dataset, self).__init__(
            **kwargs)

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        """Format the results into dir (standard format for Cityscapes
        evaluation).
        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        result_files = []
        for res, idx in zip(results, indices):
            if len(res.shape) == 3 and not np.issubdtype(res.dtype, np.integer):
                result_file = osp.join(imgfile_prefix, self.img_infos[idx]["filename"][:-4] + ".png")
                if not osp.exists(osp.dirname(result_file)):
                    os.system(f"mkdir -p {osp.dirname(result_file)}")
                cv2.imwrite(result_file, (res * 65535).astype(np.uint16))
            else:
                result_file = osp.join(imgfile_prefix, self.img_infos[idx]["filename"][:-4] + ".png")
                if not osp.exists(osp.dirname(result_file)):
                    os.system(f"mkdir -p {osp.dirname(result_file)}")
                Image.fromarray(res.astype(np.uint8)).save(result_file)
            result_files.append(result_file)

        return result_files
