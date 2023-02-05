"""
module which contains prepare_loaders function
"""
import torch
DataSet = __import__('create_dataset').DataSet
get_transform = __import__('create_dataset').get_transform

def prepare_loaders(root: str='./', labels: dict={},
                    batch_size: int=4, shuffle: bool=True):
    """
    prepares torch.utils.data.DataLoader objects to load our dataset
    root: path to root folder
    IMPORTANT: you must create images and annotations in root
               to save the images sample with their respective xml labels 
    Returns the train_dl, val_dl, test_dl
    """
    # set train=true to training image transforms
    train_ds = DataSet(root, labels, get_transform(train=True)) # train dataset
    val_ds = DataSet(root, labels, get_transform(train=False)) # validation dataset
    test_ds = DataSet(root, labels, get_transform(train=False)) # test dataset

    # randomly shuffle all the data
    indices = torch.randperm(len(train_ds)).tolist()
    # spliting entire data into 80/20 train-test splits
    # spliting train set into 82/20 train-calidation splits
    n = len(indices)

    # train dataset: 64% of the entire data, or 80% of 80%
    train_ds = torch.utils.data.Subset(train_ds,
                                       indices[:int(n * .64)])
    # validation dataset: 16% of the entire data, or 20% of 80%
    val_ds = torch.utils.data.Subset(val_ds,
                                     indices[int(n * .64):int(n * .8)])
    # test dataset: 20% of the entire data
    test_ds = torch.utils.data.Subset(test_ds,
                                      indices[int(n * .8):])
    def collate_fn(batch):
        """
        collate image-target pairs into a tuple
        """
        return tuple(zip(*batch))

    # Creating needed DataLoader objects for use with DataSet instances
    # collate_fn is used when for the batched loading
    # and so to map the dataset in x y for X(x, y) = yhat & Y(yhat, y) = loss
    # from the F(X, Y) = Z model
    train_dl = torch.utils.data.DataLoader(train_ds,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       collate_fn=collate_fn)
    val_dl = torch.utils.data.DataLoader(val_ds,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           collate_fn=collate_fn)
    test_dl = torch.utils.data.DataLoader(test_ds,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           collate_fn=collate_fn)

    return train_dl, val_dl, test_dl
