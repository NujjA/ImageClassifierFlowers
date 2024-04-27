#Train a new network on a data set with train.py

#Basic usage: python train.py data_directory
#Prints out training loss, validation loss, and validation accuracy as the network trains

#Options: 
#* Set directory to save checkpoints: python train.py data_dir --save_dir save_directory 
#* Choose architecture: python train.py data_dir --arch "vgg13" 
#* Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 
#* Use GPU for training: python train.py data_dir --gpu

import argparse
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json

parser = argparse.ArgumentParser(description ='Arguments used for training network')
parser.add_argument('data_dir', help='Path to data', default=os.getcwd())
parser.add_argument('--save_dir', default=os.getcwd(),
                    help="Path to save location")
parser.add_argument('--arch', default="vgg13",
                    help="Network architecture")
parser.add_argument('--checkpoint', default="checkpoint.pth",
                    help="Name of file to store model information")
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help="Set the learning rate")
parser.add_argument('--hidden_units', type=int, default=512,
                    help="The number of hidden units")
parser.add_argument('--epochs', type=int, default=5,
                    help="The number of epochs to train")
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    help="Use the GPU")

args = parser.parse_args()


print_every = 30
b_size = 64
epochs = args.epochs
data_dir = args.data_dir

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.RandomRotation(50),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])



# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=validation_transforms)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = getattr(models, args.arch)(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

#classifier_input = model.classifier
classifier_input = model.classifier[0].in_features
classifier = nn.Sequential(nn.Linear(classifier_input, args.hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(args.hidden_units, 1000),
                                 nn.LogSoftmax(dim=1))
model.classifier = classifier

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
model.to(device)



def train_model(model, epochs, print_every, trainloader, testloader):
    running_loss = 0
    steps = 0
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            print("inputs and labels sent to " + str(device) + " step: " + str(steps))

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                print("calculating loss and accuracy")
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()
    return model, optimizer



# function for a more detailed version of a checkpoint
def save_model(model, checkpoint_name):
    checkpoint = {'arch': args.arch,
                  'hidden_units': args.hidden_units,
                  'model_state_dict': model.state_dict(),
                  'mapping': valid_data.class_to_idx,
                  'classifier_input': model.classifier[0].in_features,
                  'classifier_output': model.classifier[0].out_features,
                  'optimizer':optimizer.state_dict()
                 }

    torch.save(checkpoint, checkpoint_name)
    print("checkpoint saved")
    

########### model trained here
print_every = 30
b_size = 64

trainloader = torch.utils.data.DataLoader(train_data, batch_size=b_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=b_size)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=b_size)

model, optimizer = train_model(model, args.epochs, print_every, trainloader, testloader)
save_model(model, args.checkpoint)