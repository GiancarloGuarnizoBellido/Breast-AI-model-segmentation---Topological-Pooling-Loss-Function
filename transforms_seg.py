import torch 
import numpy as np

class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, pred = sample['image'], sample['pred']
        return {'image': image, 'pred': pred}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        prediction=sample['pred']
        image_var_torch = torch.from_numpy(image).to(torch.float) / 255.0
        pred_tensor = torch.from_numpy(prediction).to(torch.float) / 255.0
        return {'image': image_var_torch, 'pred': pred_tensor}
