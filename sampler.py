from RandMix import RandMix
import torch
from torchvision import transforms

device=torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")) 

class sampler:
    """
    A class for data augmentation using randomized mixing with noise.

    Attributes:
        noise (float): The noise level to be applied in data augmentation.
        randomizer (RandMix): An instance of the RandMix class for applying random mixing.
        normalizer (transforms.Normalize): A normalization transform for standardizing image data.

    Methods:
        __init__(self, noise=1.0):
            Initializes the sampler with a specified noise level.
        
        generate(self, images, labels, ratio):
            Generates augmented data by applying random mixing and normalization to input images and labels.
    """

    def __init__(self, noise=1.0):
        """
        Initializes the sampler class.

        Parameters:
            noise (float): The noise level to apply during random mixing. Default is 1.0.
        """
        self.noise = noise
        self.randomizer = RandMix(noise_lv=self.noise).to(device)
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def generate(self, images, labels, ratio):
        """
        Generates augmented image and label datasets using random mixing and normalization.

        Parameters:
            images (torch.Tensor): A batch of input images (e.g., shape [N, C, H, W]).
            labels (torch.Tensor): Corresponding labels for the input images (e.g., shape [N]).
            ratio (float): The mixing ratio for randomizing the augmentation process.

        Returns:
            tuple:
                torch.Tensor: Augmented images after random mixing and normalization.
                torch.Tensor: Concatenated labels, repeated for the augmented images.
        """
        # Reset GPU memory stats and clear cache to ensure clean memory usage
        clear_cache()

        # Generate new augmented data by applying the randomizer and normalization
        new_data = self.normalizer(torch.sigmoid(self.randomizer(images, ratio=ratio)))

        # Concatenate original and augmented images and labels
        new_x = torch.cat([images, new_data])
        new_y = torch.cat([labels, labels])

        # Return the augmented images after further random mixing and the concatenated labels
        return RandMix(noise_lv=self.noise).to(device).forward(new_x), new_y
    

def clear_cache():
    """
    Clears GPU memory by resetting peak memory statistics and emptying the cache.
    """
    if device.type=="cuda":

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    elif device.type=="mps":
        torch.mps.empty_cache()

def generate_samples(images, labels):
    """
    Prepares and generates augmented samples from input images and labels.

    This function normalizes input images, adjusts their format for PyTorch operations, 
    and applies data augmentation using the `sampler` class. The output consists
    of augmented images and corresponding labels.

    Parameters:
        images (ndarray): Input images as a NumPy array of shape (N, H, W, C), where:
                          - N: Number of images
                          - H: Height of each image
                          - W: Width of each image
                          - C: Number of channels (e.g., 3 for RGB images).
        labels (ndarray): Corresponding labels for the images as a NumPy array of shape (N,).

    Returns:
        tuple:
            torch.Tensor: Augmented images of shape (2 * N, C, H, W).
            torch.Tensor: Augmented labels of shape (2 * N,).
    """
    X_new = ((torch.tensor(images)).float() / 255.0).permute(0, 3, 1, 2).to(device)
    X_gen, y_gen = sampler().generate(X_new, torch.tensor(labels).float().to(device), 1)
    return X_gen, y_gen
