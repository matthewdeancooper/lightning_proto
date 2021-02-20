import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF


def transforms(image, segmentation):
    transform_chance = 0.5

    if random.random() < transform_chance:
        image = TF.hflip(image)
        segmentation = TF.hflip(segmentation)

    if random.random() < transform_chance:
        angle = random.randint(-3, 3)
        # x, y translation
        translate = random.randint(-3, 3), random.randint(-3, 3)
        # x, y shear angle value
        shear = random.randint(-3, 3), random.randint(-3, 3)
        scale = 1
        image = TF.affine(image, angle, translate, scale, shear)
        segmentaion = TF.affine(segmentation, angle, translate, scale, shear)

    if random.random() < transform_chance:
        size = image.shape[-1]
        grid_size = random.randint(int(size - 0.1 * size), size)
        top = random.randint(0, size - grid_size)
        left = random.randint(0, size - grid_size)
        height = grid_size
        width = grid_size
        image = TF.resized_crop(image, top, left, height, width, size)
        segmentation = TF.resized_crop(segmentation, top, left, height, width,
                                       size)

    if random.random() < transform_chance:
        kernel_size = 3
        sigma = 0.1
        image = TF.gaussian_blur(image, kernel_size, sigma)
        segmentation = TF.gaussian_blur(segmentation, kernel_size, sigma)

    image = TF.normalize(image, torch.mean(image), torch.std(image))

    return image, segmentation


# if __name__ == "__main__":
#     img = np.load("../test_dataset/6375_cleaned/img/30.npy")
#     img = torch.from_numpy(img)

#     segmentation = np.load("../test_dataset/6375_cleaned/mask/30.npy")
#     sementation = torch.from_numpy(segmentation)

#     img, segmentation = transforms(img, img)
#     plt.imshow(img[0, ...])
#     plt.savefig("mygraph.png")
