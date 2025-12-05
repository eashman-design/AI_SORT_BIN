# augment.py
# Eashan's AI_SORT_BIN data augmentation pipeline
# Works for: TensorFlow classification OR object detection

import tensorflow as tf

# ---------------------------------------------------------
# IMAGE-ONLY AUGMENTATIONS
# ---------------------------------------------------------

def random_flip(image, bboxes=None):
    """Horizontally flips the image (and bboxes if provided)."""
    flipped_image = tf.image.flip_left_right(image)

    # If doing DETECTION, adjust bounding boxes
    if bboxes is not None:
        ymin, xmin, ymax, xmax = tf.split(bboxes, 4, axis=1)
        new_xmin = 1.0 - xmax
        new_xmax = 1.0 - xmin
        bboxes = tf.concat([ymin, new_xmin, ymax, new_xmax], axis=1)
        return flipped_image, bboxes

    return flipped_image


def random_brightness(image):
    return tf.image.random_brightness(image, max_delta=0.15)


def random_contrast(image):
    return tf.image.random_contrast(image, lower=0.85, upper=1.15)


def random_saturation(image):
    return tf.image.random_saturation(image, lower=0.85, upper=1.20)


def random_hue(image):
    return tf.image.random_hue(image, max_delta=0.02)


def random_crop(image):
    """Optional — only for classification data."""
    return tf.image.random_crop(image, size=[224, 224, 3])


def random_jpeg_noise(image):
    """Simulates compression artifacts from webcams."""
    quality = tf.random.uniform([], 40, 100, dtype=tf.int32)
    jpeg = tf.image.adjust_jpeg_quality(image, quality)
    return jpeg


# ---------------------------------------------------------
# MASTER AUGMENT FUNCTION
# ---------------------------------------------------------

def augment(image, bboxes=None, is_detection=False):
    """
    Main augmentation function.
    image: TensorFlow image (H,W,3)
    bboxes: (N,4) in normalized [ymin, xmin, ymax, xmax] format
    is_detection: True if training SSD/YOLO-style detector
    """
    # Cast to float32
    image = tf.image.convert_image_dtype(image, tf.float32)

    # RANDOM FLIP
    if tf.random.uniform([]) < 0.5:
        if is_detection:
            image, bboxes = random_flip(image, bboxes)
        else:
            image = random_flip(image)

    # COLOR AUGMENTATIONS
    if tf.random.uniform([]) < 0.8:
        image = random_brightness(image)

    if tf.random.uniform([]) < 0.7:
        image = random_contrast(image)

    if tf.random.uniform([]) < 0.6:
        image = random_saturation(image)

    if tf.random.uniform([]) < 0.4:
        image = random_hue(image)

    # JPEG NOISE
    if tf.random.uniform([]) < 0.3:
        image = random_jpeg_noise(image)

    # Clip back to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)

    if is_detection:
        return image, bboxes

    # CLASSIFICATION: resize if needed
    image = tf.image.resize(image, (224, 224))
    return image


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------

def augment_detection_example(image, boxes):
    """Call this function inside your tf.data pipeline for SSD/YOLO."""
    return augment(image, bboxes=boxes, is_detection=True)


def augment_classification_example(image, label):
    """Call this for MobileNetV2 classification."""
    return augment(image, is_detection=False), label

