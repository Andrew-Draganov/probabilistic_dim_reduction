import numpy as np

def pairwise_euclidean(points):
    points_zero = np.expand_dims(points, axis=0)
    points_one = np.expand_dims(points, axis=1)
    return np.sum((points_zero - points_one)**2, axis=-1)

def remove_diag(x):
    x_no_diag = np.ndarray.flatten(x)
    x_no_diag = np.delete(x_no_diag, range(0, len(x_no_diag), len(x) + 1), 0)
    x_no_diag = x_no_diag.reshape(len(x), len(x) - 1)
    return x_no_diag

def high_dim_probs(points):
    probabilities = np.zeros((len(points), len(points)))
    pairwise_dists = pairwise_euclidean(points)
    # FIXME binary search for sigma
    sigma_i = np.ones((len(points), 1))
    numerator = np.exp(-1 * pairwise_dists / (2 * sigma_i)) 
    # FIXME should I remove one from the sums to account for the distance from a point to itself?
    denominator = np.sum(np.exp(-1 * pairwise_dists / (2 * sigma_i)), axis=1, keepdims=True) - 1
    p_j_given_i = numerator / denominator
    # Zero out probability from point to itself
    p_j_given_i -= np.eye(len(p_j_given_i)) * p_j_given_i
    P = (p_j_given_i + np.transpose(p_j_given_i)) / 2
    return P

def low_dim_probs(points):
    probabilities = np.zeros((len(points), len(points)))
    pairwise_dists = pairwise_euclidean(points)
    # FIXME binary search for sigma
    sigma_i = np.ones((len(points), 1))
    numerator = np.pow(1 + pairwise_dists / (2 * sigma_i), -1) 
    # FIXME should I remove one from the sums to account for the distance from a point to itself?
    denominator = np.sum(np.exp(-1 * pairwise_dists / (2 * sigma_i)), axis=1, keepdims=True) - 1
    q_j_given_i = numerator / denominator
    # Zero out probability from point to itself
    q_j_given_i -= np.eye(len(q_j_given_i)) * q_j_given_i
    Q = (q_j_given_i + np.transpose(q_j_given_i)) / 2
    return P


if __name__ == '__main__':
    points = np.random.rand(2000, 16)
    P = high_dim_probs(points)


