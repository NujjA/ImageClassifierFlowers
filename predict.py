import argparse
import os
import torch
from torch import nn, optim
from torchvision import models, transforms
import PIL.Image
import numpy as np
import json

parser = argparse.ArgumentParser(description ='Arguments used for predicting flower type')
parser.add_argument('path_to_img', help='Path to image')
parser.add_argument('checkpoint_name', help='Model used in prediction')
parser.add_argument('--save_dir', default=os.getcwd(),
                    help="Path to save location")
parser.add_argument('--category_names',
                    help="Path to JSON file with category names")
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    help="Use the GPU")
parser.add_argument('--top_k', type=int, default=5,
                    help="The number of predictions to return")
args = parser.parse_args()


def open_checkpoint(checkpoint_name, device):
    checkpoint = torch.load(checkpoint_name, map_location = device)
    return checkpoint

def load_checkpoint(checkpoint):

    arch = checkpoint["arch"]
    model = getattr(models, arch)(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = checkpoint['mapping']
    classifier_input = model.classifier[0].in_features
    classifier = nn.Sequential(nn.Linear(classifier_input, checkpoint['hidden_units']),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(checkpoint['hidden_units'], 1000),
                                 nn.LogSoftmax(dim=1))
    
    model.classifier = classifier

    criterion = nn.NLLLoss() 
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer']) # not needed but good to keep for future
    
    return model 

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    return img_transforms(image)

    # Implement the code to predict the class from an image file
def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    processed_img = process_image(PIL.Image.open(image_path))
    info = torch.exp(model(processed_img.unsqueeze(0)))
    probs, vals = torch.topk(info, args.top_k)
    idx_mapping = dict(map(reversed, checkpoint["mapping"].items()))
    classes = []
    for flower in vals[0]:
        classes += [idx_mapping[flower.item()]]
    
    return probs.data, classes

def convert_to_name(classes, cat_to_name):
    flowernames = []
    for flower in classes:
        flowernames += [cat_to_name[str(flower)]]
    return flowernames

def print_results(classes, probs):
    print("This flower is most likely {0} with a probability of {1:4f}".format(classes[0], probs[0]))
    print("Here are the top {0} results:".format(args.top_k))
    results = zip(classes, probs)
    for result in results:
        print("Class: {0} - Probability: {1:4f}".format(result[0], result[1]))
##testing
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
checkpoint = open_checkpoint(args.checkpoint_name, device)
model = load_checkpoint(checkpoint)
probs, classes = predict(args.path_to_img, model, args.top_k)
if(args.category_names):
    cat_to_name = {}
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    flowernames = convert_to_name(classes, cat_to_name)
    print_results(flowernames, probs[0])
else:
    print_results(classes, probs[0])
