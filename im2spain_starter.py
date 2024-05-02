"""
The goal of this assignment is to predict GPS coordinates from image features using k-Nearest Neighbors.
Specifically, have featurized 28616 geo-tagged images taken in Spain split into training and test sets (27.6k and 1k).

The assignment walks students through:
    * visualizing the data
    * implementing and evaluating a kNN regression model
    * analyzing model performance as a function of dataset size
    * comparing kNN against linear regression

Images were filtered from Mousselly-Sergieh et al. 2014 (https://dl.acm.org/doi/10.1145/2557642.2563673)
and scraped from Flickr in 2024. The image features were extracted using CLIP ViT-L/14@336px (https://openai.com/clip/).
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def plot_data(train_feats, train_labels):
    """
    Input:
        train_feats: Training set image features
        train_labels: Training set GPS (lat, lon)

    Output:
        Displays plot of image locations, and first two PCA dimensions vs longitude
    """
    # Plot image locations (use marker='.' for better visibility)
    plt.scatter(train_labels[:, 1], train_labels[:, 0], marker=".")
    plt.title('Image Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    # Run PCA on training_feats
    ##### TODO(a): Your Code Here #####
    transformed_feats = StandardScaler().fit_transform(train_feats)
    transformed_feats = PCA(n_components=2).fit_transform(transformed_feats)

    # Plot images by first two PCA dimensions (use marker='.' for better visibility)
    plt.scatter(transformed_feats[:, 0],     # Select first column
                transformed_feats[:, 1],     # Select second column
                c=train_labels[:, 1],
                marker='.')
    plt.colorbar(label='Longitude')
    plt.title('Image Features by Longitude after PCA')
    plt.show()


def grid_search(train_features, train_labels, test_features, test_labels, is_weighted=False, verbose=True):
    """
    Input:
        train_features: Training set image features
        train_labels: Training set GPS (lat, lon) coords
        test_features: Test set image features
        test_labels: Test set GPS (lat, lon) coords
        is_weighted: Weight prediction by distances in feature space

    Output:
        Prints mean displacement error as a function of k
        Plots mean displacement error vs k

    Returns:
        Minimum mean displacement error
    """
    # Evaluate mean displacement error (in miles) of kNN regression for different values of k
    # Technically we are working with spherical coordinates and should be using spherical distances, but within a small
    # region like Spain we can get away with treating the coordinates as cartesian coordinates.
    knn = NearestNeighbors(n_neighbors=100).fit(train_features)

    if verbose:
        print(f'Running grid search for k (is_weighted={is_weighted})')

    ks = list(range(1, 11)) + [20, 30, 40, 50, 100]
    mean_errors = []
    for k in ks:
        distances, indices = knn.kneighbors(test_features, n_neighbors=k)

        errors = []
        for i, nearest in enumerate(indices):
            # Evaluate mean displacement error in miles for each test image
            # Assume 1 degree latitude is 69 miles and 1 degree longitude is 52 miles
            y = test_labels[i]

            ##### TODO(d): Your Code Here #####

            nearest_coordinates = train_labels[nearest]
            e = 0
            if is_weighted:
                weights = 1.0 / (distances[i] + 1e-9)
                weighted_coordinates = np.average(nearest_coordinates, axis=0, weights=weights)
                e = np.linalg.norm(y - weighted_coordinates)
            else:
                average_coordinates = np.mean(nearest_coordinates, axis=0)
                e = np.linalg.norm(y - average_coordinates)

            errors.append(e)
        
        e = np.mean(np.array(errors))
        mean_errors.append(e)
        if verbose:
            print(f'{k}-NN mean displacement error (miles): {e}')

    # Plot error vs k for k Nearest Neighbors
    if verbose:
        mean_errors = [x * 57 for x in mean_errors]
        plt.plot(ks, mean_errors)
        plt.xlabel('k')
        plt.ylabel('Mean Displacement Error (miles)')
        plt.title('Mean Displacement Error (miles) vs. k in kNN')
        plt.show()

    return min(mean_errors)


def main():
    print("Predicting GPS from CLIP image features\n")

    # Import Data
    print("Loading Data")
    data = np.load('im2spain_data.npz')

    train_features = data['train_features']  # [N_train, dim] array
    test_features = data['test_features']    # [N_test, dim] array
    
    data['train_labels'][:, 0] = data['train_labels'][:, 0] * 69
    data['train_labels'][:, 1] = data['train_labels'][:, 1] * 52
    train_labels = data['train_labels']      # [N_train, 2] array of (lat, lon) coords
    
    data['test_labels'][:, 0] = data['test_labels'][:, 0] * 69
    data['test_labels'][:, 1] = data['test_labels'][:, 1] * 52
    test_labels = data['test_labels']        # [N_test, 2] array of (lat, lon) coords
    
    train_files = data['train_files']        # [N_train] array of strings
    test_files = data['test_files']          # [N_test] array of strings

    # Data Information
    print('Train Data Count:', train_features.shape[0])

    # Part A: Feature and label visualization (modify plot_data method)
    plot_data(train_features, train_labels)
    print("Completed Part(a)")

    # Part C: Find the 5 nearest neighbors of test image 53633239060.jpg
    knn = NearestNeighbors(n_neighbors=3).fit(train_features)
    distances, indices = knn.kneighbors(test_features[test_files == '53633239060.jpg'])
    nearest_images = train_files[indices.squeeze()]

    # Get the coordinates of the nearest neighbors
    nearest_coordinates = train_labels[indices.squeeze()]

    print("Nearest neighbors of test image 53633239060.jpg:", nearest_images)
    print("Coordinates of nearest neighbors:", nearest_coordinates)
    print("Completed Part(b)")

    # Use knn to get the k nearest neighbors of the features of image 53633239060.jpg
    ##### TODO(c): Your Code Here #####
    c_latitude = np.mean(train_labels[:, 0])
    c_longitude = np.mean(train_labels[:, 1])
    c_coordinates = np.array([c_latitude, c_longitude])

    # repeat the centroid coordinates for all test points
    num_test_points = len(test_labels)
    repeated = np.tile(c_coordinates, (num_test_points, 1))

    # compute error between centroid coordinates and actual coordinates of the test points
    err = (test_labels - repeated) ** 2
    errors = np.sum(err)
    errors = errors ** 1/2
    errors = errors / num_test_points

    # calculate the mde
    baseline_mde = np.mean(errors)
    print("Constant Baseline MDE:", baseline_mde)
    print("Completed Part(c)")

    # Part D: establish a naive baseline of predicting the mean of the training set
    ##### TODO(d): Your Code Here #####
    min_error = grid_search(train_features, train_labels, test_features, test_labels)
    print(min_error)
    print("Completed Part(d)")

    # Part E: complete grid_search to find the best value of k
    weighted_error = grid_search(train_features, train_labels, test_features, test_labels, is_weighted=True)
    print("Best value of k is this far: " + str(weighted_error))
    print("Completed Part(e)")

    # Part F: compare to linear regression for different # of training points
    mean_errors_lin = []
    mean_errors_nn = []
    ratios = np.arange(0.1, 1.1, 0.1)
    for r in ratios:
        size = int(r * len(train_features))

        ##### TODO(f): Your Code Here #####
        linear_regression = LinearRegression().fit(train_features[:size], train_labels[:size])
        predictions = linear_regression.predict(test_features)
        error = np.linalg.norm(test_labels - predictions, axis=1)
        e_lin = np.mean(error)

        e_nn = grid_search(train_features[:size], train_labels[:size], test_features, test_labels)
        mean_errors_lin.append(e_lin)
        mean_errors_nn.append(e_nn)

    # Plot error vs training set size
    plt.plot(ratios, mean_errors_lin, label='lin. reg.')
    plt.plot(ratios, mean_errors_nn, label='kNN')
    plt.xlabel('Training Set Ratio')
    plt.ylabel('Mean Displacement Error (miles)')
    plt.title('Mean Displacement Error (miles) vs. Training Set Ratio')
    plt.legend()
    plt.show()
       

if __name__ == '__main__':
    main()
