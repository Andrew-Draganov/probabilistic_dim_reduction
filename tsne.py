import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mnist import MNIST

def get_deltas(points):
    points = points.astype(np.float32)
    points_zero = np.expand_dims(points, axis=0)
    points_one = np.expand_dims(points, axis=1)
    return (points_zero - points_one)

def normalize(points):
    points -= np.amin(points)
    points /= np.amax(points)
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

def high_dim_probs(points, desired_perplexity=10, n_steps=100, perplexity_tolerance=1e-5):
    pairwise_dists = normalize(pairwise_euclidean(points))
    n_samples, n_neighbors = pairwise_dists.shape[0], pairwise_dists.shape[1]
    using_neighbors = n_neighbors < n_samples

    P = np.zeros((n_samples, n_neighbors))
    print('Binary searching for variances...')
    for i in range(n_samples):
        beta_min, beta_max = -np.inf, np.inf
        beta = 1
        for l in range(n_steps):
            sum_Pi = 0.0
            for j in range(n_neighbors):
                if i != j or using_neighbors:
                    P[i, j] = math.exp(-pairwise_dists[i, j] * beta)
                    sum_Pi += P[i, j]

            if sum_Pi == 0.0:
                sum_Pi = 1e-8
            sum_disti_Pi = 0.0

            for j in range(n_neighbors):
                P[i, j] /= sum_Pi
                sum_disti_Pi += pairwise_dists[i, j] * P[i, j]

            entropy = math.log(sum_Pi) + beta * sum_disti_Pi
            entropy_diff = entropy - np.log2(desired_perplexity)

            if math.fabs(entropy_diff) <= perplexity_tolerance:
                break

            if entropy_diff > 0.0:
                beta_min = beta
                if beta_max == np.inf:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -np.inf:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

    # numerator = np.exp(-1 * pairwise_dists / (2 * sigma ** 2)) 
    # Subtract len(points) so that we're not comparing point to itself in denominator sum
    # denominator = np.sum(np.exp(-1 * pairwise_dists / (2 * sigma ** 2))) - len(points)
    # assert denominator != 0
    # p_j_given_i = numerator / denominator
    # Zero out probability from point to itself
    # p_ji_no_diag = remove_diag(p_j_given_i)
    # binary search for perplexity
    # for perp_step in range(100):
    #     numerator = np.exp(-1 * pairwise_dists / (2 * sigma ** 2)) 
    #     # Subtract len(points) so that we're not comparing point to itself in denominator sum
    #     denominator = np.sum(np.exp(-1 * pairwise_dists / (2 * sigma ** 2))) - len(points)
    #     assert denominator != 0
    #     p_j_given_i = numerator / denominator
    #     # Zero out probability from point to itself
    #     p_ji_no_diag = remove_diag(p_j_given_i)
    #     P = (p_j_given_i + np.transpose(p_j_given_i)) / (2 * len(points))
    #     entropy = -1 * np.sum(p_ji_no_diag * np.log2(p_ji_no_diag), axis=1)
    #     perplexity = np.power(2, entropy)
    #     perplexity_satisfied = (np.abs(np.mean(perplexity - np.array(desired_perplexity))) < perp_thresh)
    #     print(sigma)
    #     print(perplexity)
    #     if perp_step == 2:
    #         quit()
    #     if perplexity_satisfied:
    #         print('perplexity satisfied')
    #         break
    #     temp = sigma[:]
    #     for i, s in enumerate(sigma):
    #         if perplexity[i] < desired_perplexity:
    #             sigma[i] = (previous_sigma[i] + max_sigma) / 2
    #         else:
    #             sigma[i] = (previous_sigma[i] + min_sigma) / 2

    #     previous_sigma = temp

    P = P + P.T
    # The paper for t-SNE claims that they divide the symmetric probability by 2n.
    # However, the sklearn implementation instead divides by the sum over P
    P /= (np.sum(P) / 2)
    assert np.all(np.abs(P.data) <= 1.0)
    return P

def low_dim_probs(points):
    pairwise_dists = pairwise_euclidean(points)
    numerator = np.power(1 + pairwise_dists, -1) 
    # Remove similarity from point to itself
    numerator -= np.eye(len(numerator))
    denominator = np.sum(np.power(1 + pairwise_dists, -1)) - len(points)
    Q = numerator / denominator
    return Q


if __name__ == '__main__':
    mndata = MNIST('./python-mnist/data')
    images, labels = mndata.load_training()
    images = np.array(images)
    labels = np.array(labels)
    images = images[::40, :]
    labels = labels[::40]
    pca = PCA(n_components=2)
    y = normalize(pca.fit_transform(images))
    P = high_dim_probs(images)
    lr = 100
    l2_coefficient = 0.01
    old_y = y

    # Gradient descent loop
    for i in range(1000):
        if i % 10 == 0:
            print(i)

        Q = low_dim_probs(y)
        # assert symmetricity of probability distributions
        assert np.allclose(P, P.T)
        assert np.allclose(Q, Q.T)

        if i < 250:
            momentum = 0.5
        else:
            momentum = 0.8

        # Implement early exaggeration
        if i < 50:
            exaggeration_coefficient = 4
        else:
            exaggeration_coefficient = 1

        prob_diff = np.expand_dims(exaggeration_coefficient * P - Q, axis=-1)
        deltas = get_deltas(y)
        Z = np.power(1 + pairwise_euclidean(y, keepdims=True), -1)
        all_grads = prob_diff * deltas * Z
        gradient = 4 * np.sum(all_grads, axis=1)
        # L2 normalization for early compression implementation
        if i < 100:
            gradient -= y * l2_coefficient

        temp = y
        y_delta = y - old_y
        y = y + lr * gradient + momentum * y_delta
        old_y = temp

    plt.scatter(y[:, 0], y[:, 1], c=labels)
    plt.show()
