import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = torch.load('dataset/part_one_dataset/eval_data/2_eval_data.tar.pth')

images = data['data']
labels = data['targets']

unique_elements = np.unique(labels)

print(images.shape, labels.shape, unique_elements)

# for i in range(len(images)):
#     img = images[i]  
#     label = labels[i] 

#     plt.imshow(img.squeeze(), cmap='gray')  
#     plt.title(f'Label: {label}')
#     plt.axis('off')
#     plt.show()

#     if i >= 10:  
#         break
