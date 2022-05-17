import torchvision.transforms as transforms

IMG_HORIZONTAL_SHIFT = [0.1, 0.2, 0.3, 0.4, 0.5]
IMG_VERTICAL_SHIFT = [0.1, 0.2, 0.3, 0.4, 0.5]
IMG_ROTATION = [15, 30, 45, 60, 75]
IMG_BRIGHTNESS = [1.1, 1.2, 1.3, 1.4, 1.5]
IMG_LEVELS = [0, 1, 2, 3, 4]

class HorizontalTranslate(object):
    def __init__(self, distance, img_size):
        self.distance = distance
        self.img_size = img_size

    def __call__(self, image):
        img_size = self.img_size
        dx = float(self.distance * img_size[0])
        tx = int(round(dx))
        ty = 0
        translations = (tx, ty)
        return transforms.functional.affine(image, 0, translations, 1.0, 0, fill=0)

class VerticalTranslate(object):
    def __init__(self, distance, img_size):
        self.distance = distance
        self.img_size = img_size

    def __call__(self, image):
        img_size = self.img_size
        dy = float(self.distance * img_size[1])
        tx = 0
        ty = int(round(dy))
        translations = (tx, ty)
        return transforms.functional.affine(image, 0, translations, 1.0, 0, fill=0)