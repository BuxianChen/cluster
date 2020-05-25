from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Transformer(object):
    def __init__(self, data_name, bs, augment=True):
        # cifar
        cifar_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.18,
            height_shift_range=0.18,
            channel_shift_range=0.1,
            horizontal_flip=True,
            rescale=0.95,
            zoom_range=[0.85, 1.15])

        # mnist
        mnist_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            channel_shift_range=0.05,
            rescale=0.975,
            zoom_range=[0.95, 1.05])

        # usps
        usps_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1)

        gen_dict = {"mnist": usps_datagen, "cifar10": cifar_datagen, "usps": usps_datagen}
        self.augment = augment
        self.datagen = gen_dict.get(data_name, usps_datagen) if augment else None
        self.bs = bs

    def transform(self, images):
        # images: np.array: (self.bs, H, W, C)
        return next(self.datagen.flow(images, batch_size=self.bs, shuffle=False)) if self.augment else images
