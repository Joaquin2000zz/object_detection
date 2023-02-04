"""
module which contains Dataset, Compose, ToTensor, RandomHorizontalFlip classes
and get_transform
"""
import os
import PIL
import torch
import torchvision.transforms.functional as F
import torchvision.transforms.transforms as T
xml_to_dict = __import__('get_label').xml_to_dict


class DataSet(torch.utils.data.Dataset):
    """
    prepare a dataset used to perform training
    """

    def __init__(self, root: str, labels: dict, transforms: object=None) -> tuple:
        """
        class constructor:
            - root: name of root folder
            - labels: objects to detect
            - transforms: Compose instance to transform images
                using one ToTensor() and RandomHorizontalFlip instances 
        Returns: a tuple with images and targets
        """
        self.root = root
        self.transforms = transforms
        self.files = sorted(os.listdir('images'))

        for i, file in enumerate(self.files):
            self.files[i] = self.files[i].split('.')[0]
        self.labels = labels
    
    def __getitem__(self, i):
        """
        getter item iterate through images
        - if self.transforms first transforms image as tensor and then
          applies a random horizontal flip
        """
        # Load image from disk
        img = PIL.Image.open(
            os.path.join(self.root,
                         f'images/{self.files[i]}.png')).convert('RGB')
        # Load annotation file from disk
        ann = xml_to_dict(os.path.join(self.root,
                          f'annotations/{self.files[i]}.xml'))

        # The target is given as a dict
        target = {}
        target['boxes'] = torch.as_tensor(
            [[
                ann['x1'], ann['y1'], ann['x2'], ann['y2']
            ]], dtype=torch.float32
        )
        target['labels'] = torch.as_tensor([self.labels[ann['label']]],
                                           dtype=torch.int64)
        target['image_id'] = torch.as_tensor(i)

        # Apply any transforms to the data if required
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.files)

class ToTensor(torch.nn.Module):
    """
    converts a PIL image into a torch tensor
    """
    def forward(self, image, target: dict=None) -> tuple:
        """
        - image: PIL image
        - target: dict
        Returns: tensor, dict
        """
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)

        return image, target

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    """
    randomly flips an image horizontally
    """

    def forward(self, image, target: dict=None) -> tuple:
        """
        - image: tensor
        - target: dict
        Returns: image, tensor 
        """
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target:
                width, _ = F.get_image_size(image)
                target['boxes'][:, [0, 2]] = width - \
                    target['boxes'][:, [2, 0]]
        return image, target

class Compose:
    """
    Composes several torchvision image transforms
    as a sequence of transformations
    """

    def __init__(self, transforms: list=[]):
        """
        transforms: list of torchvision image transformations
        """
        self.transforms = transforms
    
    def __call__(self, image, target):
        """
        - sequentially performs the image transformation on
            the input image
        Returns: the augmented image
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

def get_transform(train: bool) -> Compose:
    """
    Transforms a PIL Image into a torch tensor, and performs random horizontal
        flipping of the image if training a model.
    - train: indicates wheter model training will occur
    Returns:
        - compose: compose composition of image transforms
    """
    transforms = []

    # ToTensor is applied to all images.
    transforms.append(ToTensor())
    # The following transforms are applied only to the train set
    if train:
        transforms.append(RandomHorizontalFlip(.5))
        # other transforms can be added
    return Compose(transforms)
