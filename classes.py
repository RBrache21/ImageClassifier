from data import get_dataset, data_loader, image_datasets, process_image
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import models
import torchvision

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet18 = models.resnet18(pretrained = True)
alexnet = models.alexnet(pretrained = True)
vgg16 = models.vgg16(pretrained = True)

models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}


'''

This Function updates the classifier of the pre-trained model.

params: 
    - model_name: The name of the pre-trained model which we should get from
      the get_dataset function from data.py
      
    - hidden_units: The number of hidden units to use in the classifier

returns: The pre trained model with an updated classifier.

'''

def model_classifier(model_name, hidden_units):
    
    
    model = models[model_name]
    
    if model == vgg16:
        
        features = 25088
        
    elif model == alexnet:
        
        features = 9216
        
    else:
        
        features = 512
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(nn.Linear(features, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim = 1))                        
    
    return model
                           
def train_classifier(epochs, data_directories, model, learning_rate, gpu):
    
    train_loader = data_loader(image_datasets(data_directories[0]), 64, True)
    test_loader = data_loader(image_datasets(data_directories[1]), 64, False)
    
#     Cheks if the user specifies the use or not of gpu
    if gpu == True and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model.to(device)
    
    criterion = nn.NLLLoss()
    
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
                           
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Moving inputs and labels to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

#             Testing our network accuracy and loss
            if steps % print_every == 0:
                # Turning our model into evaluation mode
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        output = model.forward(inputs)
                        batch_loss = criterion(output, labels)
                        test_loss += batch_loss.item()

                        # ACCURACY ( This accurary part of the code I got it from the previous classes in the nanodegree)
                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(test_loader):.3f}.. "
                    f"Test accuracy: {accuracy/len(test_loader):.3f}")

                running_loss = 0
                model.train()
                
    return model                      

def checkpoint(epochs, data_directories, model, learning_rate, model_name): 
    
    model.class_to_idx = image_datasets(data_directories[0]).class_to_idx
    
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    if model_name == 'vgg':
        arch = 'vgg16'
    elif model_name == 'resnet':
        arch = 'resnet18'
    else:
        arch = 'alexnet'
    
    checkpoint = {
        'input_size': 25088, # need to find a way to update this automatically
        'output_size': 102,
        'epochs': epochs,
        'arch': arch,
        'learning_rate': learning_rate,
        'classifier': model.classifier,
        'batch_size': 64,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    torch.save(checkpoint, 'checkpoint.pth')
    
    
def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["state_dict"])
#     optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint["class_to_idx"]
   
    
    
    return model 
    # Do we need to return the optimizer? 

'''
This function predicts the name of a given image. 

'''
def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if gpu == True and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    
    model.eval()
    
    # Importing the Image and proccesing it
    image = process_image(image_path)
    
    # Image to a Tensor
    image = torch.from_numpy(np.array([image])).float()
    
    image = image.to(device)
    
    # Calculating the output
    output = model.forward(image)
    
    # Probabilities of the image
    ps = torch.exp(output)
    
    
    # This last part of this code I got it from a guy named najeebhassan 
    # I found his way more efficient and effective
    probability = torch.topk(ps, topk)[0].tolist()[0] 
    index = torch.topk(ps, topk)[1].tolist()[0] 
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    # transfer index to label
    classes = []
    for i in range(topk):
        classes.append(ind[index[i]])

    return probability, classes

                          