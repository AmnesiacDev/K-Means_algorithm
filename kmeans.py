import numpy as np
import matplotlib.pyplot as plt

def mk_folder(self, path):
    try:
        os.mkdir(path)
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        pass
    except PermissionError:
        print(f"Permission denied: Unable to create '{path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

class Kmeans:

    def __init__(self, n_samples, k, seed=63):
        self.labels = None
        self.n_samples = n_samples
        self.k = k
        np.random.seed(seed)
        self.samples = np.random.randn(n_samples, 2) * 0.75 + np.random.rand(n_samples, 2) * 5
        indices = np.random.choice(n_samples, k, replace=False)
        self.centroids = self.samples[indices]


    def update_clusters(self):
        tol = 1e-4
        while True:
            distances = np.linalg.norm(self.samples[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            new_centroids = np.array([self.samples[self.labels == i].mean(axis=0) for i in range(len(self.centroids))])

            if np.all(np.linalg.norm(new_centroids - self.centroids, axis=1) < tol):
                break

            self.centroids = new_centroids


    def plot_clusters(self, output_name = "kmeans", folder="", ext="jpg"):
        parent_folder = "kmeans_output"
        mk_folder(parent_folder)
        if folder:
            mk_folder(f"{parent_folder}/{folder}")

            path = f"{parent_folder}/{folder}/{output_name}"
        else:
            path = f"{parent_folder}/{output_name}"

        plt.figure(figsize=(8, 6))
        k = len(self.centroids)
        for i in range(k):
            plt.scatter(self.samples[self.labels == i, 0], self.samples[self.labels == i, 1], label=f'Cluster {i+1}')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='black', s=200, marker='X', label='Centroids')
        plt.title('K-Means Clustering (From Scratch)')
        plt.legend()

        plt.savefig(f"{path}.{ext}")


    def compare_seeds(self, iterations):
        for i in range(iterations):
            SEED = np.random.randint(0, 100)
            loop = Kmeans(self.n_samples, self.k, SEED)
            loop.update_clusters()
            loop.plot_clusters(f"iter-{i}", folder="kmeans_seeds")

