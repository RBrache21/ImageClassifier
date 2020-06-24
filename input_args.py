import argparse

def input_args():
    
    parser = argparse.ArgumentParser()
    
    # Input args for training
    parser.add_argument('--dataset', type = str, default = 'flowers/', help = "Dataset to use for training")
    parser.add_argument('--arch', default ='vgg', help = 'The CNN model architecture to use')
    parser.add_argument('--hunits', default= 1024, help = 'Number of hidden units')
    parser.add_argument('--epochs', default = 5, help = 'Number of epochs')
    parser.add_argument('--lrate', default = 0.0001, help = 'Learning rate to use in the model')
    parser.add_argument('--gpu', default = True, help = 'Whether use gpu or cpu to train the model')
    
    # Input args for precting
    parser.add_argument('--filepath', type = str, default = 'checkpoint.pth', help = "Dataset to use for training")
    parser.add_argument('--top_k', default = 5, help = 'Return top K likely classes')
    parser.add_argument('--img_path', type = str, default = './flowers/test/1/image_06743.jpg', help = "Image to predict")
    parser.add_argument('--class_values', type = str, default = 'cat_to_name.json', help = "JSON file that maps the class values to other category names")
    
    return parser.parse_args()

