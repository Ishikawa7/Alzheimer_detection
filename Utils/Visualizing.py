import matplotlib.pyplot as plt
import numpy as np

def display(np_images_list : list[np.ndarray], labels: np.ndarray, n=10):
    """
    Displays n random images from each one of the supplied arrays.
    """

    indices = np.random.randint(len(np_images_list[0]), size=n)
    np_n_images = []
    for np_image in np_images_list:
        np_n_images.append(list(np_image[indices, :]))

    plt.figure(figsize=(20, 4))

    for col in range(len(np_n_images[0])):
        for row in range(len(np_n_images)):
            ax = plt.subplot(len(np_n_images), len(np_n_images[0]), row * len(np_n_images[0]) + col + 1)
            plt.imshow(np_n_images[row][col].reshape(208, 176))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.title(labels[col], fontsize=10)
    plt.show()