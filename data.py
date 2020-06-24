import torchvision
import numpy as np
from torchvision import datasets, transforms, models
import torch
import json
from PIL import Image

'''
This function divides the data into train, test and valid datasets.

parameters: 

    data_dir: a data directory which is expected to be separated into train, valid, and test datasets,
    and inside each dataset all images separated in folders with the label as the name of each folder.

retuns: 
    
    train, valid and test directories
'''

def get_dataset(data_dir):
    
    train_dir = data_dir + 'train'
    valid_dir = data_dir + 'valid'
    test_dir = data_dir + 'test'
    
    return train_dir, valid_dir, test_dir


'''
This function applies a transformation for a given data directory.

params:

    - The data directory 
    
    returns:
    
        A trasformed dataset so we can convert it ento a data loader later.
    
'''

def image_datasets(data_dir):
    
    normalization = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    
    # Checks if its getting a train dataset to apply a different traansformation. 
    # In training it is better to do additional transformations so the model can learn better.
    
    if data_dir.split("/")[1] == 'train':   
        transformation = transforms.Compose([transforms.RandomResizedCrop(224),
                                            transforms.RandomRotation(30),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normalization])
        
    else:
               
        transformation = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             normalization])
    
    dataset = datasets.ImageFolder(data_dir, transform = transformation)
    
    return dataset

'''
This function creates a generator (data loader) so we can loop through images and labels.

params: 
    - dataset: a dataset with applied tranformations.
    - batch size: number of images we want to pass in each loop
    - shuffle (True or False). This species if we want different images for new batches. 
    
returns:
    A data loader.
'''

def data_loader(dataset, batch_size, shuffle):
    
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle = shuffle)
    
    return loader



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    # Resizing the image
    img = img.resize((256,256))
    # Croping the image
    left = (256 - 224)/2
    top = (256 - 224)/2
    right = (256 + 224)/2
    bottom = (256 + 224)/2
    
    img = img.crop((left, top, right, bottom))
    
    # Converting values
    img = np.array(img)/255
    
    # Normalizing the Image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    img = img.transpose(2,0,1)
    
    return img

        