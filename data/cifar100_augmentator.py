import numpy as np

class CIFAR100Augmentor:
    def __init__(self, crop_padding=4, flip_prob=0.5, noise_std=0.0):
        self.crop_padding = crop_padding
        self.flip_prob = flip_prob
        self.noise_std = noise_std

    def augment_batch(self, images):
        # images: [batch, channels, height, width]
        augmented = []
        for img in images:
            img = self.random_crop(img)
            img = self.random_flip(img)
            img = self.add_noise(img)
            augmented.append(img)
        return np.stack(augmented)

    def random_crop(self, img):
        c, h, w = img.shape
        padded = np.pad(img, ((0, 0), (self.crop_padding, self.crop_padding), (self.crop_padding, self.crop_padding)), mode='reflect')
        top = np.random.randint(0, 2 * self.crop_padding)
        left = np.random.randint(0, 2 * self.crop_padding)
        return padded[:, top:top+h, left:left+w]

    def random_flip(self, img):
        if np.random.rand() < self.flip_prob:
            return img[:, :, ::-1]  # horizontal flip
        return img

    def add_noise(self, img):
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, img.shape).astype(img.dtype)
            img = img + noise
            img = np.clip(img, 0.0, 1.0)
        return img