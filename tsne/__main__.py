from argparse import ArgumentParser, Namespace
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

from .tsne import TSNE


def parse_args() -> Namespace:
    parser = ArgumentParser("TSNE visualization algorithm.")
    parser.add_argument(
        "-o", "--output-file", type=str, required=False, default="output.jpg"
    )
    parser.add_argument(
        "-n", "--num-clusters", type=int, required=False, default=3
    )
    parser.add_argument(
        "--seed",
        required=False,
        type=int,
        default=42,
    )
    parser.add_argument(
        "--feature-size",
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument(
        "--data-per-cluster",
        type=int,
        required=False,
        default=50,
    )

    return parser.parse_args()


def generate_data(
    num_clusters: int,
    feature_size: int,
    data_per_cluster: int,
):
    logger.info("Creating data.")
    centers = np.random.randint(-20, 20, num_clusters)
    logger.info("Cluster centers: {}", centers)
    clusters = [np.ones(feature_size) * center for center in centers]

    X = np.concatenate(
        [
            np.random.rand(data_per_cluster, feature_size) + cluster
            for cluster in clusters
        ],
        axis=0,
    )
    return X


def main():
    args = parse_args()
    np.random.seed(args.seed)
    X = generate_data(
        args.num_clusters, args.feature_size, args.data_per_cluster
    )

    logger.info("Fitting TSNE.")
    X_transformed = TSNE(n_iter=2000).fit_transform(X)

    plt.figure(figsize=(16, 9))
    plt.scatter(
        X_transformed[:, 0],
        X_transformed[:, 1],
        c=np.ravel(
            [
                [i for _ in range(args.data_per_cluster)]
                for i in range(args.num_clusters)
            ]
        ),
    )
    plt.title("TSNE")
    plt.savefig(args.output_file)


if __name__ == "__main__":
    main()
