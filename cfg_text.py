
import numpy as np

class Config(object):
    
    ############################################################
    # train
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # dataset
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 640
    MAX_GT_INSTANCES = 20

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    STD_PIXEL = np.array([1., 1., 1.])

    num_classes = 1
    score_threshold = 0.15
    use_nms = True
    nms_thresh = 0.4
    epochs = 200
    down_ratio = 2

    
    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
 


