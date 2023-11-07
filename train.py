import argparse
import torch
from torchvision import transforms, datasets, models
from torch import nn, optim
import futility
import fmodel
import json

parser = argparse.ArgumentParser(description='Parser for train.py')

parser.add_argument('data_dir', action="store", default='flowers/', type=str, help='Path to the data directory')
parser.add_argument('--save_dir', action="store", default='flower_classifier.pth', type=str, help='Path to save the trained model checkpoint')
parser.add_argument('--arch', action="store", default='vgg16', type=str, help='Pre-trained model architecture')
parser.add_argument('--learning_rate', action="store", default=0.001, type=float, help='Learning rate')
parser.add_argument('--hidden_units', action="store", default=512, type=int, help='Number of hidden units in the classifier')
parser.add_argument('--epochs', action="store", default=3, type=int, help='Number of training epochs')
parser.add_argument('--dropout', action="store", default=0.2, type=float, help='Dropout rate')
parser.add_argument('--gpu', action="store", default='gpu', type=str, help='Use GPU for training (if available)')

args = parser.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
dropout = args.dropout
gpu = args.gpu

if torch.cuda.is_available() and gpu == 'gpu':
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def main():
    # Load and preprocess data
    dataloaders, dataset_sizes, class_to_idx = futility.load_data(data_dir)
    
    # Build the model
    model = fmodel.build_model(arch, hidden_units, dropout)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Train the model
    futility.train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, epochs)
    
    # Save the model checkpoint
    model.class_to_idx = class_to_idx
    fmodel.save_checkpoint(model, save_dir, arch, hidden_units, dropout, learning_rate, epochs, class_to_idx)

if __name__ == '__main__':
    main()
