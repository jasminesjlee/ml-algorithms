# k-nearest neighbors

def generate_train_points(count):
    # Generates data points to cluster
    x_coords = np.random.randint(low=0, high=100, size=count)
    y_coords = np.random.randint(low=0, high=100, size=count)
    X = np.array([[x, y] for x, y in zip(x_coords, y_coords)])
    return X


def generate_test_points(count):
    # Generates data points to cluster
    x_coords = np.random.randint(low=0, high=100, size=count)
    y_coords = np.random.randint(low=0, high=100, size=count)
    X = np.array([[x, y] for x, y in zip(x_coords, y_coords)])
    return X
