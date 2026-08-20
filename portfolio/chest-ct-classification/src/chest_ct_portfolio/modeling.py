"""Reference TensorFlow model for corrected transfer-learning methodology.

TensorFlow is an optional dependency so the data/evaluation utilities can be
reviewed and tested without installing the full ML runtime.
"""
from __future__ import annotations


def build_resnet50_classifier(*, num_classes: int = 4, image_size: int = 224):
    """Build a frozen-backbone ResNet50 classifier for an initial training phase."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "TensorFlow is optional. Install this project with the 'ml' extra."
        ) from exc

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.05)(x)
    x = tf.keras.applications.resnet50.preprocess_input(x)

    backbone = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="chest_ct_resnet50_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
