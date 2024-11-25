import matplotlib.pyplot as plt
import numpy as np

def __init__():
    pass

def convert_to_grayscale(images):
    """
    This takes 4D array `images` and return 3D array of grayscaled_images
    """
    grayscale_images = (0.2989 * images[:, :, :, 0] + 
                        0.5870 * images[:, :, :, 1] + 
                        0.1140 * images[:, :, :, 2]).astype(np.uint8)

    return grayscale_images

def plot_class_frequencies(X, y):
    """
    Plots a bar graph of class frequencies.
    Parameters:
        X (ndarray): Data array of shape (N, d ). d can be features or d -> (32, 32, 3) array
        y (ndarray): Label array of shape (N,).
    """
    unique_labels, frequencies = np.unique(y, return_counts=True)

    plt.figure(figsize=(5, 3))
    plt.bar(unique_labels, frequencies, color='skyblue', edgecolor='black')

    plt.xlabel("Labels", fontsize=10)
    plt.ylabel("class frequency", fontsize=10)
    plt.xticks(unique_labels, fontsize=8)
    plt.yticks(fontsize=10)

    for i, freq in enumerate(frequencies):
        plt.text(unique_labels[i], freq + 0.5, str(freq), ha='center', fontsize=12)

    plt.tight_layout()
    plt.show()

def visualize_images(images, labels, num_images=30):
    """
    Visualizes the first `num_images` images and their corresponding labels in a grid.
    Parameters:
        images : ndarray, labels : ndarray.
        num_images (int): Number of images to display (default is 30).
    """
    # Set up the grid for displaying images
    num_images = min(num_images, len(images))
    cols = 6
    rows = (num_images + cols - 1) // cols 
    
    plt.figure(figsize=(10, rows * 2))  # Adjust figure size based on number of rows
    
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap= 'gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()