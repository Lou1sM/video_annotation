from skimage import transform

def image_preprocessing(image):
    image = transform.resize(image, (256, 256), mode='reflect', anti_aliasing=True, preserve_range=True).astype('uint8')
    image = image[16:240, 16:240]
    image = image.astype('float32') / 255
    image = (image - 0.5) * 2.
    return image
