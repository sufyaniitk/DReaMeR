import torch

def __init__():
    pass

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