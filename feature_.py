from data import load_data
from torchvision import models
import numpy as np
import os
import torch
import torchvision.transforms as transforms

def __init__():
    pass


def load_features(number: int, data_type: str, part: str):
    """
    This function loads the features from the `extracted` feature folder that we have created
    """

    # Check if inputs are valid
    if number not in range(1, 11):
        raise ValueError("Number must be between 1 and 10.")
    if data_type not in ["eval", "train"]:
        raise ValueError("Type must be 'eval' or 'train'.")
    if part not in ["one", "two"]:
        raise ValueError("Part must be 'one' or 'two'.")

    # Construct the path
    path = f'extracted_feature/part_{part}_feature/{data_type}_feature/{number}_{data_type}_feature.tar.pth'
    
    # Load data
    data = torch.load(path)
    # features = data('data')  # Expected shape (2500, 512)
    
    return data.numpy()


def save_extracted_feature():
    """
    This function generates the extracted feature and save it in a new directory
    """
    part = ["one", "two"]
    types = ["eval", "train"]
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for p in part:
        for t in types:
            for n in numbers:
                images = load_data(n, t, p, True)
                images = get_from_gpu(extract_features(images))
                images_tensor = torch.from_numpy(images)

                # Define save path and ensure directory exists
                save_path = f'extracted_feature/part_{p}_feature/{t}_feature/{n}_{t}_feature.tar.pth'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                torch.save(images_tensor, save_path)


def extract_features(images, batch_size=5):
    """
    It automatically handles whether you are working on MAC or NVIDIA GPU
    Extract Features -> child functions extract_features_mac if mps availaible otherwise calls extract_features_cuda
    batch_size ( = 5)byDeafult. Batch Size for processing bulk images. 5 works well for 3050 etc.
    """
    if torch.backends.mps.is_available() :
        return get_from_gpu(extract_features_mac(images, batch_size=batch_size))
    return get_from_gpu(extract_features_cuda(images, batch_size=batch_size))


def get_from_gpu(df):
    """
    Get the data from GPU to CPU
    """
    res = [t.cpu().numpy() for t in df]
    return np.vstack(res)

def extract_features_cuda(images, batch_size=5):
    """
    Extract features from a batch of CIFAR-10 images using a pre-trained ResNet34 model.

    Args:
        images (numpy.ndarray): A 4D array of shape (N, 32, 32, 3), where N is the number of images.

    Returns:
        torch.Tensor: A 2D tensor of shape (N, 512) containing the extracted features.
    """
    
    # Step 1: Convert the input images to a tensor and apply transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert to PIL Image
        transforms.Resize(224),   # Resize to 224x224
        transforms.ToTensor(),     # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    # Transform each image and create a tensor
    transformed_images = torch.stack([transform(images[i]) for i in range(images.shape[0])])

    # Step 2: Load pre-trained ResNet34 model
    model = models.resnet34(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the classification layer
    model.eval()  # Set the model to evaluation mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    transformed_images = transformed_images.to(device)

    torch.cuda.set_per_process_memory_fraction(1.0)
    torch.cuda.empty_cache()

    # Step 3: Feature extraction
    features_list = []
    with torch.no_grad():
        torch.cuda.empty_cache()
        for i in range(0, transformed_images.size(0), batch_size):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            batch_images = transformed_images[i : i + batch_size]  # Get features
            batch_features = model(batch_images)  # Flatten the output
            features_list.append(batch_features.view(batch_features.size(0), -1))

    return features_list

def extract_features_mac(images, batch_size=5):
    """
    Extract features from a batch of CIFAR-10 images using a pre-trained ResNet34 model on MPS.

    Args:
        images (numpy.ndarray): A 4D array of shape (N, 32, 32, 3), where N is the number of images.

    Returns:
        torch.Tensor: A 2D tensor of shape (N, 512) containing the extracted features.
    """
    
    batch_size = 25
    # Step 1: Convert the input images to a tensor and apply transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert to PIL Image
        transforms.Resize(224),   # Resize to 224x224
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.ToTensor(),     # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    # Transform each image and create a tensor
    transformed_images = torch.stack([transform(images[i]) for i in range(images.shape[0])])

    # Step 2: Load pre-trained ResNet34 model
    model = models.resnet34(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the classification layer
    model.eval()  # Set the model to evaluation mode

    # Use MPS if available, otherwise fall back to CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(device, end=" ")
    transformed_images = transformed_images.to(device)

    # Step 3: Feature extraction
    features_list = []
    with torch.no_grad():
        for i in range(0, transformed_images.size(0), batch_size):
            batch_images = transformed_images[i : i + batch_size]  # Get batch of images
            batch_features = model(batch_images)  # Extract features
            features_list.append(batch_features.view(batch_features.size(0), -1))

    return features_list