import os
import numpy as np
import json
import skimage.draw


import mrcnn.model as modellib
import mrcnn.visualize as visualize
import mrcnn.config as config
import mrcnn.utils as utils
import warnings

warnings.filterwarnings("ignore")

ANNOTS_PATH = os.path.abspath('./annots')
DATASET_PATH = os.path.abspath('./images/labeled')
# COCO_WEIGHTS_PATH = os.path.abspath('./mask_rcnn_coco.h5')
MODEL_PATH = os.path.abspath('./models')

############################################################
#  Configurations
############################################################


class FacesConfig(config.Config):
    """
    Configurations for training on the faces dataset.
    Inherits from the base Config class and overrides some values.
    """
    NAME = "Faces"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    BACKBONE = 'resnet50'

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    NUM_CLASSES = 4

    LEARNING_RATE = 0.001
    STEPS_PER_EPOCH = 500


class FacesInferConfig(FacesConfig):
    """
    Configurations for inferring after training has been complete.
    Inherits from FacesConfig and overrides some values
    """
    NAME = "InferFaces"
    IMAGES_PER_GPU = 1


############################################################
#  Dataset
############################################################


def load_images_data(filepath):
    """Load JSON COCO format annotations"""
    with open(filepath) as file:
        info = json.load(file)

    return info


class FacesDataset(utils.Dataset):
    """Dataset class for faces dataset"""

    def __init__(self, images_list: list, annotations_path: str,
                 dataset_dir: str):
        """
        Load dataset
        :param images_list: list of images to include in dataset, filenames/path
        :param annotations_path: str of the annotations filename/path
        :param dataset_dir: str of filepath that the dataset is located
        """
        super().__init__(self)

        # Add classes
        self.add_class('Faces', 1, 'face')
        self.add_class('Faces', 2, 'nose')
        self.add_class('Faces', 3, 'mouth')
        # load and save JSON file with annotations
        self.images_data = load_images_data(annotations_path)

        # Add images
        for i, filename in enumerate(images_list):
            images = self.images_data['images']
            for img in images:
                # gather individual image data
                if img['file_name'] == filename:
                    img_id = img['id']
                    fp = os.path.join(dataset_dir, filename)
                    img_h = img['height']
                    img_w = img['width']
                    # use that data to add image to dataset
                    self.add_image('Faces',
                                   image_id=img_id,
                                   path=fp,
                                   height=img_h,
                                   width=img_w)
        # Add annotations to images after all images have been loaded
        self.add_annotations()

    def image_reference(self, image_id: int) -> str or None:
        """Return the path of the image"""
        for img in self.image_info:
            if img['id'] == image_id:
                return img['path']
        return None

    def add_annotations(self):
        """
        Add the annotations to the image, loaded when class
        object is created.
        """
        for i, image in enumerate(self.image_info):
            # store segments and categorical ids per img_id
            img_id = image['id']
            segments = []
            cat_ids = []
            for img_ann in self.images_data['annotations']:
                # find segments and cats and add to lists
                if img_id == img_ann['image_id']:
                    segments.append(img_ann['segmentation'][0])
                    cat_ids.append(img_ann['category_id'])
            # add kwarg to img_id and move to next id
            self.image_info[i]['cat_ids'] = cat_ids
            self.image_info[i]['segments'] = segments

    def load_mask(self, image_id: int):
        """
        Generate instance masks for an image.
        :param image_id: int of image id
        :return: A bool array of shape [h, w, instance count] with
        one mask per instance.
        And a 1D array of class IDs of the instance masks.
        """
        assert image_id in self.image_ids, 'id not valid'

        # data for mask creation
        info = self.image_info[image_id]
        h = info['height']
        w = info['width']
        count = len(info['cat_ids'])

        # create class ids and mask
        class_ids = np.zeros((count,), dtype=np.int32)
        mask = np.zeros((h, w, count), dtype=np.uint8)

        # load mask and class ids
        for i, s in enumerate(info['segments']):
            class_ids[i] = info['cat_ids'][i]
            x_s = s[::2]
            y_s = s[1::2]
            rr, cc = skimage.draw.polygon(y_s, x_s)

            mask[rr, cc, i] = 1

        return mask, class_ids


def load_model(configs, weights_fp=None, exclude=None, mode='training'):
    # generate model
    model = modellib.MaskRCNN(mode=mode,
                              config=configs,
                              model_dir=MODEL_PATH)

    # load weights
    model.load_weights(filepath=weights_fp, by_name=True,
                       exclude=exclude)

    return model


def train(model, configs, annot_fn, train_imgs, val_imgs,
          epochs, LR_multiplier=1, layers='heads'):
    """Train model"""
    # Training set
    Faces_config = configs
    fp = os.path.join(ANNOTS_PATH, annot_fn)
    dataset_train = FacesDataset(train_imgs, fp, DATASET_PATH)
    dataset_train.prepare()

    # Validation set
    dataset_val = FacesDataset(val_imgs, fp, DATASET_PATH)
    dataset_val.prepare()

    # training
    print('Training Network heads')
    model.train(dataset_train, dataset_val,
                learning_rate=Faces_config.LEARNING_RATE * LR_multiplier,
                epochs=epochs,
                layers=layers)
