import argparse
import torch
import json
import numpy as np
from PIL import Image
from torchvision import models
from torchvision import transforms
import fmodel  # You may need to import your custom fmodel module for loading checkpoints and making predictions

parser = argparse.ArgumentParser(description='Parser for predict.py')

parser.add_argument('input', nargs='?', default='flowers/test/100/image_07896.jpg', type=str, help='Path to the image for prediction')
parser.add_argument('--checkpoint', default='flower_classifier.pth', type=str, help='Path to the model checkpoint')
parser.add_argument('--top_k', default=5, type=int, help='Top K predictions to display')
parser.add_argument('--category_names', default='cat_to_name.json', type=str, help='Path to the category names JSON file')
parser.add_argument('--gpu', default='gpu', type=str, help='Use GPU for inference (if available)')

args = parser.parse_args()
path_image = args.input
number_of_outputs = args.top_k
device = 'cuda' if args.gpu == 'gpu' and torch.cuda.is_available() else 'cpu'
json_name = args.category_names
path = args.checkpoint

def main():
    # Load the model checkpoint using your load_checkpoint function
    model = fmodel.load_checkpoint(path)
    
    # Load the class to index mapping from the checkpoint
    with open(json_name, 'r') as json_file:
        cat_to_name = json.load(json_file)
    
    # Load and process the image
    input_img = fmodel.process_image(path_image)
    
    # Convert the NumPy array to a PyTorch tensor
    input_img = torch.from_numpy(input_img).to(device)
    
    # Add a batch dimension
    input_img = input_img.unsqueeze(0)
    
    # Use the predict function to make predictions
    top_ps, top_classes = fmodel.predict(input_img, model, number_of_outputs)
    
    # Map class indices to class names
    class_names = [cat_to_name[str(cls)] for cls in top_classes]
    
    # Print the top K predictions and their class names
    for i in range(number_of_outputs):
        print(f"Prediction {i + 1}: {class_names[i]} with probability {top_ps[i]:.3f}")
    
if __name__ == "__main__":
    main()
