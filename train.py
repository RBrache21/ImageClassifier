from input_args import input_args
import torch
from data import get_dataset, data_loader
from classes import model_classifier, train_classifier, checkpoint



def main():
    
    
    
    in_arg = input_args()
    
    ''' 
    Getting The train, valid and test datasets from the function get_dataset.
    Converting it into a list to get access to the differents dasasets by just 
    using list indexing.
    
    data_directories[0] = flowers/train
    data_directories[1] = flowers/valid
    data_directories[2] = flowers/test
    
    '''
    
    data_directories = list(get_dataset(in_arg.dataset))
                               

    # Getting the model and updating the model's classifier.

    trained_model = train_classifier(in_arg.epochs, data_directories, model_classifier(in_arg.arch, in_arg.hunits), in_arg.lrate, in_arg.gpu) 
    
    # Saving the model
    
    checkpoint(in_arg.epochs, data_directories, trained_model, in_arg.lrate, in_arg.arch)
    
    
if __name__ == "__main__":
    main()

    