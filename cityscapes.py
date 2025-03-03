import cv2 as cv
import numpy as np
import albumentations as AT

from pathlib import Path
from collections import namedtuple
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

_NUM_CLASSES = 19

class CityScapes(Dataset):
    Label = namedtuple( 'Label', ['name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color'] )
    labels = [
        Label( 'unlabeled'           , 0 , 255, 'void'        , 0, False, True , ( 0 , 0  , 0  ) ),
        Label( 'ego vehicle'         , 1 , 255, 'void'        , 0, False, True , ( 0 , 0  , 0  ) ),
        Label( 'rectification border', 2 , 255, 'void'        , 0, False, True , ( 0 , 0  , 0  ) ),
        Label( 'out of roi'          , 3 , 255, 'void'        , 0, False, True , ( 0 , 0  , 0  ) ),
        Label( 'static'              , 4 , 255, 'void'        , 0, False, True , ( 0 , 0  , 0  ) ),
        Label( 'dynamic'             , 5 , 255, 'void'        , 0, False, True , (111, 74 , 0  ) ),
        Label( 'ground'              , 6 , 255, 'void'        , 0, False, True , ( 81, 0  , 81 ) ),
        Label( 'road'                , 7 , 0  , 'flat'        , 1, False, False, (128, 64 , 128) ),
        Label( 'sidewalk'            , 8 , 1  , 'flat'        , 1, False, False, (244, 35 , 232) ),
        Label( 'parking'             , 9 , 255, 'flat'        , 1, False, True , (250, 170, 160) ),
        Label( 'rail track'          , 10, 255, 'flat'        , 1, False, True , (230, 150, 140) ),
        Label( 'building'            , 11, 2  , 'construction', 2, False, False, ( 70, 70 , 70 ) ),
        Label( 'wall'                , 12, 3  , 'construction', 2, False, False, (102, 102, 156) ),
        Label( 'fence'               , 13, 4  , 'construction', 2, False, False, (190, 153, 153) ),
        Label( 'guard rail'          , 14, 255, 'construction', 2, False, True , (180, 165, 180) ),
        Label( 'bridge'              , 15, 255, 'construction', 2, False, True , (150, 100, 100) ),
        Label( 'tunnel'              , 16, 255, 'construction', 2, False, True , (150, 120, 90 ) ),
        Label( 'pole'                , 17, 5  , 'object'      , 3, False, False, (153, 153, 153) ),
        Label( 'polegroup'           , 18, 255, 'object'      , 3, False, True , (153, 153, 153) ),
        Label( 'traffic light'       , 19, 6  , 'object'      , 3, False, False, (250, 170, 30 ) ),
        Label( 'traffic sign'        , 20, 7  , 'object'      , 3, False, False, (220, 220, 0  ) ),
        Label( 'vegetation'          , 21, 8  , 'nature'      , 4, False, False, (107, 142, 35 ) ),
        Label( 'terrain'             , 22, 9  , 'nature'      , 4, False, False, (152, 251, 152) ),
        Label( 'sky'                 , 23, 10 , 'sky'         , 5, False, False, ( 70, 130, 180) ),
        Label( 'person'              , 24, 11 , 'human'       , 6, True , False, (220, 20 , 60 ) ),
        Label( 'rider'               , 25, 12 , 'human'       , 6, True , False, (255, 0  , 0  ) ),
        Label( 'car'                 , 26, 13 , 'vehicle'     , 7, True , False, ( 0 , 0  , 142) ),
        Label( 'truck'               , 27, 14 , 'vehicle'     , 7, True , False, ( 0 , 0  , 70 ) ),
        Label( 'bus'                 , 28, 15 , 'vehicle'     , 7, True , False, ( 0 , 60 , 100) ),
        Label( 'caravan'             , 29, 255, 'vehicle'     , 7, True , True , ( 0 , 0  , 90 ) ),
        Label( 'trailer'             , 30, 255, 'vehicle'     , 7, True , True , ( 0 , 0  , 110) ),
        Label( 'train'               , 31, 16 , 'vehicle'     , 7, True , False, ( 0 , 80 , 100) ),
        Label( 'motorcycle'          , 32, 17 , 'vehicle'     , 7, True , False, ( 0 , 0  , 230) ),
        Label( 'bicycle'             , 33, 18 , 'vehicle'     , 7, True , False, (119, 11 , 32 ) ),
        Label( 'license plate'       , -1, -1 , 'vehicle'     , 7, False, True , ( 0 , 0  , 142) ),
    ]

    num_classes = _NUM_CLASSES
    id_to_train_id = np.array([label.trainId if label.trainId != 255 else _NUM_CLASSES for label in labels])
    train_id_to_color = np.array([label.color for label in labels if label.trainId != 255] + [(0,0,0)])

    def __init__(self, root, split, num_images=0):
        imgs = Path(root, 'leftImg8bit', split)
        imgs = list(imgs.rglob('*_leftImg8bit.png'))

        if num_images > 0:
            imgs = imgs[:num_images]

        self.len = len(imgs)
        self.imgs = imgs
        self.msks = [
            str(i).replace('leftImg8bit', 'gtFine').replace('.png', '_labelIds.png')
            for i in imgs
        ]

        if split == 'train':
            self.transform = AT.Compose([
                AT.RandomCrop(height=224, width=224),
                AT.ColorJitter(),
                AT.HorizontalFlip(p=0.5),
                ToTensorV2(),
            ])
        else:
            self.transform = AT.Compose([
                ToTensorV2(),
            ])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = cv.imread(img)

        msk = self.msks[idx]
        msk = cv.imread(msk, cv.IMREAD_GRAYSCALE)

        augmented = self.transform(image=img, mask=msk)
        img, msk = augmented['image'], augmented['mask']

        msk = CityScapes.id_to_train_id[np.array(msk)]
        img = img / 255
        return img, msk
