import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

def get_deltas(points):
    points = points.astype(np.float32)
    points_zero = np.expand_dims(points, axis=0)
    points_one = np.expand_dims(points, axis=1)
    return (points_zero - points_one)

def normalize_points(points):
    points = points / np.amax(points)
    return points

def pairwise_euclidean(points, keepdims=False):
    deltas = get_deltas(points)
    dists = np.sum(deltas**2, axis=-1, keepdims=keepdims)
    assert dists[0, 0] == 0
    return dists

def remove_diag(x):
    x_no_diag = np.ndarray.flatten(x)
    x_no_diag = np.delete(x_no_diag, range(0, len(x_no_diag), len(x) + 1), 0)
    x_no_diag = x_no_diag.reshape(len(x), len(x) - 1)
    return x_no_diag

def high_dim_probs(points, desired_perplexity=10, perp_thresh=1):
    points = normalize_points(points)
    probabilities = np.zeros((len(points), len(points)))
    pairwise_dists = pairwise_euclidean(points)
    max_sigma, min_sigma = 100, 0
    sigma = np.ones((len(points), 1))
    previous_sigma = np.ones((len(points), 1))
    perplexity_satisfied = False

    # binary search for perplexity
    for perp_step in range(1000):
        numerator = np.exp(-1 * pairwise_dists / (2 * sigma ** 2)) 
        # Subtract one so that we're not comparing point to itself in denominator sum
        denominator = np.sum(np.exp(-1 * pairwise_dists / (2 * sigma ** 2))) - len(points)
        assert denominator != 0
        p_j_given_i = numerator / denominator
        # Zero out probability from point to itself
        P = (p_j_given_i + np.transpose(p_j_given_i)) / 2
        entropy = -1 * np.sum(P * np.log2(P), axis=1)
        perplexity = np.power(2, entropy)
        perplexity_satisfied = (np.abs(np.mean(perplexity - np.array(desired_perplexity))) < perp_thresh)
        print(sigma)
        print(perplexity)
        if perplexity_satisfied:
            print('perplexity satisfied')
            break
        if(perp_step == 1):
            quit()
        temp = sigma[:]
        for i, s in enumerate(sigma):
            if perplexity[i] < desired_perplexity:
                sigma[i] = (previous_sigma[i] + max_sigma) / 2
            else:
                sigma[i] = (previous_sigma[i] + min_sigma) / 2

        previous_sigma = temp

    # Zero out the diagonal
    P -= np.eye(len(P)) * P
    return P

def low_dim_probs(points):
    probabilities = np.zeros((len(points), len(points)))
    pairwise_dists = pairwise_euclidean(points)
    numerator = np.power(1 + pairwise_dists, -1) 
    # Remove similarity from point to itself
    numerator -= np.eye(len(numerator))
    denominator = np.sum(np.power(1 + pairwise_dists, -1), axis=1, keepdims=True) - 1
    Q = numerator / denominator
    return Q


if __name__ == '__main__':
    mndata = MNIST('./python-mnist/data')
    images, labels = mndata.load_training()
    images = np.array(images)
    labels = np.array(labels)
    images = images[::100, :]
    labels = labels[::100]
    y = np.random.rand(len(images), 2)
    P = high_dim_probs(images)
    lr = 100
    momentum = 0.5
    old_y = y

    # Gradient descent loop
    for i in range(1000):
        Q = low_dim_probs(y)
        prob_diff = np.expand_dims(P - Q, axis=-1)
        deltas = get_deltas(y)
        Z = np.power(1 + pairwise_euclidean(y, keepdims=True), -1)
        all_grads = prob_diff * deltas * Z
        gradient = 4 * np.sum(all_grads, axis=1)

        temp = y
        y_delta = y - old_y
        y = y + lr * gradient + momentum * y_delta
        old_y = temp

    plt.scatter(y[:, 0], y[:, 1], c=labels)
    plt.show()
