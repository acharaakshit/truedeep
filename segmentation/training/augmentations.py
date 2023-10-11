import numpy as np
from albumentations import *
from albumentations.imgaug.transforms import IAAAffine
from random import randrange


def randAugmentation(image, mask, n_transforms=1, transforms_to_use=None):
    aug_out = dict()
    aug_out['image'] = image
    aug_out['mask'] =  mask

    available_transforms = ['flip', 'rotate', 'shiftScaleRotate', 'shear', 'translate1', 'translate2',
                            'downScale', 'CLAHE', 'gaussian_blur', 'median_blur', 'sharpen']
    
    if transforms_to_use is None:
        transforms_to_perform = available_transforms
    else:
        transforms_to_perform = []
        for transform in transforms_to_use:
            if transform in available_transforms:
                transforms_to_perform.append(transform)
    
    assert len(transforms_to_perform) >= n_transforms
    get_transforms = list(np.random.choice(transforms_to_perform, n_transforms, replace=False))

    for transform in get_transforms:
        if transform == 'flip':
            aug_func = Compose(Flip(p=1.0))
            aug_out = aug_func(image=aug_out['image'], mask=aug_out['mask'])
        elif transform == 'rotate':
            aug_func = Compose(Rotate(p=1.0, limit=(-90,90), interpolation=0, mask_value=0, border_mode=0, value=(0,0,0)))
            aug_out = aug_func(image=aug_out['image'], mask=aug_out['mask'])
        elif transform == 'shiftScaleRotate':
            aug_func = Compose(ShiftScaleRotate(always_apply=False, p=1.0, rotate_limit=(-90,90), interpolation=0,
                                                border_mode=0, value=(0,0,0), mask_value=0, shift_limit=0.2,
                                                scale_limit=0.5))
            aug_out = aug_func(image=aug_out['image'], mask=aug_out['mask'])
        elif transform == 'shear':
            aug_func = Compose(IAAAffine(p=1.0, shear=0.2))
            aug_out = aug_func(image=aug_out['image'], mask=aug_out['mask'])
        elif transform == 'translate1':
            aug_func = Compose(IAAAffine(p=1.0, translate_percent=0.2))
            aug_out = aug_func(image=aug_out['image'], mask=aug_out['mask'])
        elif transform == 'translate2':
            aug_func = Compose(IAAAffine(p=1.0, translate_percent=0.3))
            aug_out = aug_func(image=aug_out['image'], mask=aug_out['mask'])
        elif transform == 'downScale':
            aug_func = Compose(Downscale(p=1.0, scale_min=0.25, scale_max=0.5))
            aug_out = aug_func(image=aug_out['image'], mask=aug_out['mask'])
        elif transform == 'CLAHE':
            aug_func = Compose(CLAHE(p=1.0, clip_limit=2, tile_grid_size=(7,7)))
            aug_out = aug_func(image=aug_out['image'], mask=aug_out['mask'])
        elif transform == 'gaussian_blur':
            aug_func = Compose(GaussianBlur(p=1.0))
            aug_out = aug_func(image=aug_out['image'], mask=aug_out['mask'])
        elif transform == 'median_blur':
            aug_func = Compose(MedianBlur(p=1.0))
            aug_out = aug_func(image=aug_out['image'], mask=aug_out['mask'])
        elif transform == 'sharpen':
            aug_func = Compose(Sharpen(p=1.0, alpha=(0.2, 0.5), lightness=(0.5, 1.0)))
            aug_out = aug_func(image=aug_out['image'], mask=aug_out['mask'])
        
    return aug_out
