import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image
from RandMix import RandMix

def load_data(number: int, data_type: str, part: str, X_only = False, y_only = False):
    """
    Loads image and label data based on the provided number, type, and part.
    Parameters:
        number (int): A number from 1 to 10, specifying the dataset part.
        data_type (str): A string, either 'eval' or 'train', specifying the dataset type.
        part (str): A string, either 'one' or 'two', specifying the dataset part.
    Returns:
        tuple: A tuple containing two arrays, images and labels if available, otherwise only images.
    Raises:
        ValueError: If `number` is not between 1 and 10, `data_type` is not 'eval' or 'train',
                    or `part` is not 'one' or 'two'.
    """

    # Check if inputs are valid
    if number not in range(1, 11):
        raise ValueError("Number must be between 1 and 10.")
    if data_type not in ["eval", "train"]:
        raise ValueError("Type must be 'eval' or 'train'.")
    if part not in ["one", "two"]:
        raise ValueError("Part must be 'one' or 'two'.")

    # Construct the path
    path = f'dataset/part_{part}_dataset/{data_type}_data/{number}_{data_type}_data.tar.pth'
    
    # Load data
    data = torch.load(path)
    images = data.get('data')  # Expected shape (2500, 32, 32, 3)
    
    if X_only == True:
        return images

    if 'targets' in data:
        labels = data['targets']  # Expected shape (2500,)
        if y_only == True: return labels
        return images, labels
    else:
        return images  # Return only images if labels are not present

# Initialize RandMix
noise_level = 1.0  # Adjust noise level
model = RandMix(noise_lv=noise_level).cuda() # No need for .cuda()

images, labels = load_data(1, 'train', 'one')

# Generate random input data (e.g., 4 RGB images of size 64x64)
# batch_size = 4
# image_size = (64, 64)  # Height and width
# x = torch.randn(batch_size, 3, *image_size).cuda()  # Random input on CPU
x = (torch.from_numpy(images[0]).permute(2, 0, 1).float() / 255.0).unsqueeze(0).cuda()
print(x.shape) # torch.Size([3, 32, 32])

# Generate samples
with torch.no_grad():  # No gradients needed for inference
    output = model.forward(x)
print(type(output))
print(output.shape)

def gray(image):
    return np.dot(image[:, :, :3], [0.2989, 0.5870, 0.1140])

# Convert and visualize samples
def visualize_images(images, title="Generated Images"):
    """Visualizes a batch of images."""
    plt.figure(figsize=(12, 6))
    # for i in range(images.shape[0]):
    #     plt.subplot(1, images.shape[0], i + 1)
    #     img = to_pil_image(images[i].clamp(0, 1))  # Clamp values and convert
    #     plt.imshow(img)
    #     plt.axis("off")
    for i in range(images.shape[0]):
        plt.subplot(1, images.shape[0], i + 1)
        # plt.imshow(gray(images[i]), cmap='gray')
        plt.imshow(images[i])
        # plt.title(f"Label: {labels[i]}")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

# Visualize input and output
visualize_images(np.array([x.cpu().squeeze(0).permute(1, 2, 0).numpy()]), title="Original Inputs")
out = output.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
# print(out.squeeze(0).shape)
visualize_images(np.array([out]), title="RandMix Outputs")