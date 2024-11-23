import numpy as np

class LwP:
    def __init__(self, n_prototypes):
        """
        Parameters:
            n_prototypes (int): Number of distinct labels.
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
            X (ndarray) -> (N, 1024)
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

    def distance(self, a, b, name ='mahalanobis'):
        """
        Parameters:
            a (ndarray): First vector.
            b (ndarray): Second vector.
        Returns:
            float: Mahalanobis distance between a and b.
        """
        diff = a - b
        mat = self.inv_cov_matrix if name == 'mahalanobis' else np.eye(a.shape[0])
        return np.sqrt(np.dot(np.dot(diff, mat), diff.T))

    def update(self, X_new, y_new, update_inv_cov=True):
        """
        Updates the model with new training examples.
        Parameters:
            X_new (ndarray): New samples -> (M, 1024)
            y_new (ndarray): New labels -> (M,)
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

    def dist_mat(self, X):
        """
        dist_mat[i][j] = dist(i-th training pt, j-th prototype)
        """
        n_samples = X.shape[0]
        
        res = []
        for i in range(n_samples):
            distances = np.zeros(len(self.labels))
            for j, prototype in enumerate(self.prototypes):
                distances[j] = self.distance(X[i], prototype)
            res.append(distances)
        
        return res

    def softmax_mat(self, X):
        dist_mat = self.dist_mat(X)

        for i in dist_mat:
            total = 0
            for j in i:
                total += np.exp(j)
            
            for j in range(len(i)):
                i[j] = np.exp(i[j]) / total
        
        return dist_mat

    def next_gen(self, X, thres=0.8):
        sfm = self.softmax_mat(X)
        X_ret = []
        y_ret = []

        for i, x in enumerate(X):
            prob_idx = np.argmax(sfm[i])
            prob = sfm[i][prob_idx]
            if (prob < thres): continue

            X_ret.append(x)
            y_ret.append(prob_idx) # append class label
        
        return np.asanyarray(X_ret), np.asanyarray(y_ret)

    def top_k(self, X, k, indices=False):
        sfm = self.softmax_mat(X)
        X2 = []

        for i, Xi in enumerate(sfm):
            X2.append([np.min(Xi), i])
        
        X2 = sorted(X2)
        X2 = X2[ : k]

        if not indices:
            X_res = []
            y_res = []

            for x in X2:
                X_res.append(X[x[1]])
                y_res.append(np.argmin(sfm[x[1]]))
            
            return np.asanyarray(X_res), np.asanyarray(y_res)
        
        else:
            X_res = []
            y_res = []

            for x in X2:
                X_res.append(x[1])
                y_res.append(np.argmin(sfm[x[1]]))
            
            return np.asanyarray(X_res), np.asanyarray(y_res)

    def predict(self, X, name='mahalanobis'):
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            distances = np.zeros(len(self.labels))
            for j, prototype in enumerate(self.prototypes):
                distances[j] = self.distance(X[i], prototype, name)
            predictions[i] = self.labels[np.argmin(distances)]

        return predictions

class LwP_updated:
    def __init__(self, n_prototypes):
        """
        Parameters:
            n_prototypes (int): Number of distinct labels.
        """
        self.n_prototypes = n_prototypes
        self.prototypes = None
        self.labels = None
        self.class_cov_matrices = {}  # Covariance matrices for each class
        self.class_inv_cov_matrices = {}  # Inverse covariance matrices for each class
        self.class_counts = None  # Track the count of samples per class
        self.label_to_index = {}  # Maps each label to an index in the prototypes array

    def fit(self, X, y):
        """
        Trains the model by finding prototypes based on the training data.
        Parameters:
            X (ndarray) -> (N, 1024)
            y (ndarray): Labels -> (N,)
        """
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
        """
        Parameters:
            a (ndarray): First vector.
            b (ndarray): Second vector.
            class_label (int): Label of the class to use the respective covariance matrix.
        Returns:
            float: Mahalanobis distance between a and b for the given class label.
        """
        diff = a - b
        mat = None
        if name == 'mahalanobis':
            mat = self.class_inv_cov_matrices[class_label]
        else:
            mat = np.eye(a.shape[0])
        return np.sqrt(np.dot(np.dot(diff, mat), diff.T))

    def update(self, X_new, y_new, update_inv_cov=True):
        """
        Updates the model with new training examples.
        Parameters:
            X_new (ndarray): New samples -> (M, 1024)
            y_new (ndarray): New labels -> (M,)
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
