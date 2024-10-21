import torch
import matplotlib.pyplot as plt
import numpy as np
import string
import random



def get_random_string(length):
    # Random string with the combination of lower and upper case
    letters = string.ascii_letters
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def plot_class_distribution(dataloader, title):
    labels = [label for _, label, _ in dataloader.dataset]
    unique_labels, counts = torch.unique(torch.tensor(labels), return_counts=True)

    plt.bar(unique_labels, counts, align='center', alpha=0.7)
    plt.xticks(unique_labels)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()


def plot_angle_distribution(dataloader, title):
    angles = [angle for _, _, angle in dataloader.dataset]
    unique_angles, counts = torch.unique(torch.tensor(angles), return_counts=True)

    plt.bar(unique_angles, counts, align='center', alpha=0.7)
    plt.xticks(unique_angles)
    plt.xlabel('Angle')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()


def plot_rotated_mnist_images(dataloader, task, rotation_degree, num_images=5):
    plt.figure(figsize=(15, 3))

    b = next(iter(dataloader))[0]
    rng = np.random.default_rng()
    images = rng.choice(b, size=num_images, replace=False)

    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i, 0], cmap='gray')
        plt.title(f'Angle: {rotation_degree}°')
        plt.axis('off')

    plt.show()