import torch
import math
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset without using weights_only
data = torch.load('dataset/part_one_dataset/eval_data/1_eval_data.tar.pth')
images = data['data'] # (2500, 32, 32, 3)
labels = data['targets'] # (2500,)

# if isinstance(images, torch.Tensor):
#     images = images.view(-1, 32 * 32).numpy()
# else:
#     images = np.array(images).reshape(-1, 32 * 32)

def mahalanobis_distance(x, y):
    """
    Frobenius Norm
    Not changing the name of the function because I don't want to change it in main
    """
    diff = x - y
    M = np.matmul(diff, diff.T)
    return np.sqrt(np.trace(M))

def greyscale(img):
    """
    img: x * y * 3
    """
    gs = [[0] * len(img[0]) for _ in range(len(img))]

    for i in range(len(img)):
        for j in range(len(img[0])):
            gs[i][j] = 0.3 * img[i][j][0] + 0.59 * img[i][j][1] + 0.11 * img[i][j][2]
    
    res = np.asanyarray(gs)
    return res

def learn_prototypes(images, labels, num_classes):
    prototypes = np.zeros((num_classes, 32, 32))
    for i in range(num_classes):
        class_samples = [images[j] for j in range(2500) if labels[j] == i]
        greyscales = []
        for sample in class_samples:
            sample_greyscale = greyscale(sample)
            greyscales.append(sample_greyscale)
        
        greyscales = np.asanyarray(greyscales)
        prototypes[i] = np.mean(class_samples)
    
    return prototypes

def compute_inverse_covariance(images):
    cov_matrix = np.cov(images.T)
    inv_cov = np.linalg.inv(cov_matrix)
    return inv_cov

def classify_with_prototypes(images, prototypes):
    predictions = []
    for img in images:
        transformed_img = greyscale(img)
        distances = [mahalanobis_distance(transformed_img, proto) for proto in prototypes]
        predictions.append(np.argmin(distances))
    return np.array(predictions)

def main():
    num_classes = 10
    prototypes = learn_prototypes(images, labels, num_classes)
    predictions = classify_with_prototypes(images, prototypes)
    accuracy = np.mean(predictions == np.asarray(labels))
    print(f'Accuracy: {accuracy * 100:.2f}%')
    visualize_predictions(images, predictions, labels)

def visualize_predictions(images, predictions, labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(greyscale(images[i]), cmap='gray')
        plt.title(f'Pred: {predictions[i]}, True: {labels[i]}')
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
