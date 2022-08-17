import cv2


class Enhancer(object):
    def __init__(self):
        pass

    def __call__(self, image, labels):
        return image, labels


if __name__ == "__main__":
    test_image_file = r"F:\PASCALVOC\VOC2012\JPEGImages\2007_000733.jpg"
    test_image = cv2.imread(test_image_file)
    test_bbox = [[48, 25, 273, 383], [103, 201, 448, 435]]
    enhancer = Enhancer()
    i, b = enhancer(test_image, test_bbox)
