import torch
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
    diff = x - y
    M = np.matmul(diff, diff.T)
    return np.sqrt(M.trace)

def learn_prototypes(images, labels, num_classes):
    prototypes = np.zeros((num_classes, 32, 32))
    for i in range(num_classes):
        class_samples = [images[j] for j in range(2500) if labels[j] == i]
     #    prototypes[i] = np.mean(class_samples, axis=0)
     #    for idx, img in enumerate(class_samples):
     #      for x in range(32):
     #           for y in range(32):
                    
    prototypes[i] = (idx * prototypes[i] + img) / (idx + 1)
    return prototypes

def compute_inverse_covariance(images):
    cov_matrix = np.cov(images.T)
    inv_cov = np.linalg.inv(cov_matrix)
    return inv_cov

def classify_with_prototypes(images, prototypes):
    predictions = []
    for img in images:
        distances = [mahalanobis_distance(img, proto) for proto in prototypes]
        predictions.append(np.argmin(distances))
    return np.array(predictions)

def main():
    num_classes = 10
    prototypes = learn_prototypes(images, labels, num_classes)
#     inv_cov = compute_inverse_covariance(images)
    predictions = classify_with_prototypes(images, prototypes)
    accuracy = np.mean(predictions == labels.numpy())
    print(f'Accuracy: {accuracy * 100:.2f}%')
    visualize_predictions(images, predictions, labels)

def visualize_predictions(images, predictions, labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].reshape(32, 32), cmap='gray')
        plt.title(f'Pred: {predictions[i]}, True: {labels[i]}')
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
