import numpy as np

class LwP:
    

    def __init__(self, n_prototypes):
        """
        LwP: A class for prototype-based classification using Mahalanobis and Euclidean distances.

        This model calculates prototypes (means) for each class, computes distances using various metrics.
        functions implemented -> fit, top_k, update, softmax_matrix, dist_matrix, cosine_sim, cosine_top_k, distance, predict

        Attributes:
            n_prototypes (int): Number of unique class labels.
            prototypes (ndarray): Mean vectors for each class.
            inv_cov_matrix (ndarray): Inverse of the covariance matrix.
            ...
        """
        
        self.n_prototypes = n_prototypes
        self.prototypes = None
        self.labels = None
        self.inv_cov_matrix = None
        self.class_counts = None  # Track the count of samples per class
        self.label_to_index = {}  # Maps each label to an index in the prototypes array

    def fit(self, X, y):
        """
        Trains the model by finding prototypes based on the training data.
        Parameters:
            X (ndarray) -> (N, d)
            y (ndarray): Labels -> (N,)
        """
        unique_labels = np.unique(y)
        self.labels = unique_labels
        self.prototypes = np.zeros((self.n_prototypes, X.shape[1]))  # Placeholder for prototypes
        self.class_counts = np.zeros(self.n_prototypes, dtype=int)

        # Create label-to-index mapping
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

        # Calculate prototypes for each label
        for label in unique_labels:
            class_samples = X[y == label]
            if len(class_samples) > 0:
                idx = self.label_to_index[label]
                self.prototypes[idx] = np.mean(class_samples, axis=0)
                self.class_counts[idx] = len(class_samples)

        # Compute the covariance matrix of the dataset and its inverse
        covariance_matrix = np.cov(X, rowvar=False)
        self.inv_cov_matrix = np.linalg.inv(covariance_matrix)
        self.eye = np.eye(X.shape[1])

    def distance(self, a, b, name ='mahalanobis'):
        """
        Parameters:
            a (ndarray): First vector.
            b (ndarray): Second vector.
            name : mahalanobis(Default) / Euclidean if specified
        Returns:
            float: Mahalanobis/Euclidean distance between a and b .
        """
        diff = a - b
        mat = self.inv_cov_matrix if name == 'mahalanobis' else self.eye
        return np.sqrt(np.dot(np.dot(diff, mat), diff.T))

    def update(self, X_new, y_new, update_inv_cov=True):
        """
        Updates the model with new training examples.
        i.e it updates the co-variance matrix (depends) and updates the self.prototypes
        Parameters:
            X_new (ndarray): New samples -> (M, d)
            y_new (ndarray): New labels -> (M,)
            update_inv_cov : Boolean : True(By_default) | False -> if you don't want to change the shape of class but 
            want to change the means we set it to False.
        """
        for label in np.unique(y_new):
            new_samples = X_new[y_new == label]
            n_new = len(new_samples)
            if n_new > 0:
                idx = self.label_to_index.get(label)
                if idx is None:
                    raise ValueError(f"Label {label} not found in the model. Ensure that all labels are initialized in fit.")

                current_count = self.class_counts[idx]
                total_count = current_count + n_new
                new_mean = np.mean(new_samples, axis=0)
                
                # Update prototype as a weighted mean
                self.prototypes[idx] = (current_count * self.prototypes[idx] + n_new * new_mean) / total_count
                self.class_counts[idx] = total_count

        # Update covariance matrix based on the combined data
        if update_inv_cov:
            combined_X = np.vstack([self.prototypes, X_new])
            covariance_matrix = np.cov(combined_X, rowvar=False)
            self.inv_cov_matrix = np.linalg.inv(covariance_matrix)

    def dist_matrix(self, X):
        """
        Calculates the distance between ith training point and jth prototype
        * dist_mat[i][j] = dist(i-th training point, j-th prototype)
        Parameter : X -> (N,d)
        Return : matrix -> (N, 10) if n(prototypes) = 10
        """

        n_samples, res = X.shape[0], []
        for i in range(n_samples):
            distances = np.zeros(len(self.labels))
            for j, prototype in enumerate(self.prototypes):
                distances[j] = self.distance(X[i], prototype)
            res.append(distances)
        
        return res

    def softmax_matrix(self, X):
        """
        parameter : X -> (N, d)
        returns : matrix -> (N,10)
        It basically converts X[i][j] -> probability of being jth label on the ith training example
        """
        dist_mat = self.dist_matrix(X)

        for i in dist_mat:
            total = 0
            for j in i: total += np.exp(j)
            for j in range(len(i)): i[j] = np.exp(i[j]) / total
        return dist_mat

    def top_k(self, X, k, indices=False):
        """
        Parameters:
        * X -> (N, d) training samples
        * k -> top kth elements with high confidence value or probability
        * indices : False(bydefault) if True it will return the indices of sample in the original Matrix (X)
        Note -> high confidence => less distance from mean => lower value of softmax score or inversly prop to probability
        """
        auxiliary, sfm = [], self.softmax_matrix(X)
        for i, X_i in enumerate(sfm):
            auxiliary.append([np.max(X_i), i])
        
        auxiliary = sorted(auxiliary)
        auxiliary = auxiliary[ -1 * k : ] # top_k predictions

        X_res, y_res = [], []
        for x in auxiliary:
            if not indices: X_res.append(X[x[1]]) # the i-th 'incoming' sample
            else: X_res.append(x[1]) # the index instead of the sample
            y_res.append(np.argmax(sfm[x[1]])) # the label of this incoming sample
        
        return np.asanyarray(X_res), np.asanyarray(y_res)

    def cosine_sim(self, incoming_X):
        """
        parameter:
        incoming_X : (N,d)
        return: matrix -> (N, 10) n(prototypes) = 10

        It computes the cosine similarity between incoming_X and self.prototypes of the model.
                            <incoming_X, self.prototypes>
           cos(theta) = -----------------------------------
                        ||incoming_X|| x ||self.prototypes||
        """
        proto = self.prototypes
        proto_norm = np.linalg.norm(proto, axis=1, keepdims=True)
        incoming_norm = np.linalg.norm(incoming_X, axis=1, keepdims=1)

        proto /= proto_norm
        incoming_X /= incoming_norm

        return np.dot(incoming_norm, proto_norm.T)
    
    def cosine_top_k(self, incoming_X, k, indices=False):
        """
        Parameters:
        * incoming_X -> (N, d) training samples
        * k -> top kth elements with high confidence/cosine similarity value
        * indices : False(bydefault) if True it will return the indices of sample in the original Matrix (X)
        """
        cosine_mat = self.cosine_sim(incoming_X)
        auxiliary = []

        for i, X_i in enumerate(cosine_mat):
            auxiliary.append([np.max(X_i), i])
        
        auxiliary = sorted(auxiliary)
        auxiliary = auxiliary[ -1 * k : ] # top_k cosine similar predictions

        X_res, y_res = [], []
        for x in auxiliary:
            if not indices: X_res.append(incoming_X[x[1]]) # the i-th 'incoming' sample
            else: X_res.append(x[1]) # index of the sample
            y_res.append(np.argmax(cosine_mat[x[1]])) # the label of this incoming sample
        
        return np.asanyarray(X_res), np.asanyarray(y_res)

    def predict(self, X, name='mahalanobis'):
        """
        parameter : X -> (N, d), name = "mahalanobis"
        return : y_pred -> (N,) predicted labels from the self.prototypes

        name = "mahalanobis" it predicts using covariance matrix otherwise euclidean distance
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            distances = np.zeros(len(self.labels))
            for j, prototype in enumerate(self.prototypes):
                distances[j] = self.distance(X[i], prototype, name)
            predictions[i] = self.labels[np.argmin(distances)]

        return predictions


class KNN:
    def __init__(self, k):
        """
        Initializes the KNN model.
        
        Parameters:
            k (int): The number of nearest neighbors to consider for classification.
        """
        self.k = k

    def fit(self, X_train, y_train):
        """
        Stores the training data and corresponding labels.

        Parameters:
            X_train (ndarray): Training data of shape (N, d), where N is the number of samples 
                               and d is the number of features.
            y_train (ndarray): Labels for the training data of shape (N,).
        """
        self.X_train = X_train
        self.y_train = y_train

    def distance(self, a, b):
        """
        Computes the Euclidean distance between two vectors.

        Parameters:
            a (ndarray): First vector of shape (d,).
            b (ndarray): Second vector of shape (d,).
        
        Returns:
            float: Euclidean distance between vectors `a` and `b`.
        """
        diff = a - b
        return np.sqrt(np.dot(diff.T, diff))

    def predict(self, X_test):
        """
        Predicts the labels for a given test dataset using the KNN algorithm.

        Parameters:
            X_test (ndarray): Test data of shape (M, d), where M is the number of test samples 
                              and d is the number of features.

        Returns:
            ndarray: Predicted labels for the test data of shape (M,).
        """
        y_pred = np.zeros(X_test.shape[0])  # Initialize an array to store predictions for M test samples

        for i, x in enumerate(X_test):  # Loop through each test sample
            distances = []  # List to store distances to all training samples

            # Compute distances from the current test point to all training points
            for idx, training_point in enumerate(self.X_train):
                distances.append([self.distance(x, training_point), idx])

            # Sort distances and select the k closest neighbors
            distances = sorted(distances)[:self.k]

            # Count occurrences of each label among the k neighbors
            counts = np.zeros(10)  # Assuming 10 possible labels (0-9)
            for point in distances:
                counts[self.y_train[point[1]]] += 1

            # Assign the label with the highest count (majority voting)
            label = np.argmax(counts)
            y_pred[i] = label

        return y_pred



class LwP_diff_cov:
    def __init__(self, n_prototypes):
        """
        Only difference between this class the `LwP` is that It creates different covariance matrix
        for each class.
        """

        self.n_prototypes = n_prototypes
        self.prototypes = None
        self.labels = None
        self.class_cov_matrices = {}  # Covariance matrices for each class
        self.class_inv_cov_matrices = {}  # Inverse covariance matrices for each class
        self.class_counts = None  # Track the count of samples per class
        self.label_to_index = {}  # Maps each label to an index in the prototypes array

    def fit(self, X, y):
        unique_labels = np.unique(y)
        self.labels = unique_labels
        self.prototypes = np.zeros((self.n_prototypes, X.shape[1]))  # Placeholder for prototypes
        self.class_counts = np.zeros(self.n_prototypes, dtype=int)

        # Create label-to-index mapping
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

        # Calculate prototypes and covariance matrices for each label
        for label in unique_labels:
            class_samples = X[y == label]
            if len(class_samples) > 0:
                idx = self.label_to_index[label]
                self.prototypes[idx] = np.mean(class_samples, axis=0)
                self.class_counts[idx] = len(class_samples)

                # Compute covariance and inverse covariance for this class
                cov_matrix = np.cov(class_samples, rowvar=False)
                self.class_cov_matrices[label] = cov_matrix
                self.class_inv_cov_matrices[label] = np.linalg.inv(cov_matrix)

    def distance(self, a, b, class_label, name='mahalanobis'):

        diff = a - b
        mat = None
        if name == 'mahalanobis':
            mat = self.class_inv_cov_matrices[class_label]
        else:
            mat = np.eye(a.shape[0])
        return np.sqrt(np.dot(np.dot(diff, mat), diff.T))

    def update(self, X_new, y_new, update_inv_cov=True):

        for label in np.unique(y_new):
            new_samples = X_new[y_new == label]
            n_new = len(new_samples)
            if n_new > 0:
                idx = self.label_to_index.get(label)
                if idx is None:
                    raise ValueError(f"Label {label} not found in the model. Ensure that all labels are initialized in fit.")

                current_count = self.class_counts[idx]
                total_count = current_count + n_new
                new_mean = np.mean(new_samples, axis=0)
                
                # Update prototype as a weighted mean
                self.prototypes[idx] = (current_count * self.prototypes[idx] + n_new * new_mean) / total_count
                self.class_counts[idx] = total_count

                # Update covariance and inverse covariance matrix for this class
                combined_samples = np.vstack([self.prototypes[idx].reshape(1, -1)] * current_count + [new_samples])
                cov_matrix = np.cov(combined_samples, rowvar=False)
                self.class_cov_matrices[label] = cov_matrix
                self.class_inv_cov_matrices[label] = np.linalg.inv(cov_matrix)

    def predict(self, X, name='mahalanobis'):
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            distances = np.zeros(len(self.labels))
            for j, prototype in enumerate(self.prototypes):
                class_label = self.labels[j]
                distances[j] = self.distance(X[i], prototype, class_label, name)
            predictions[i] = self.labels[np.argmin(distances)]

        return predictions
