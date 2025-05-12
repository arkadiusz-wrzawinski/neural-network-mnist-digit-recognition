import errno
import os
import io
import numpy as np
from numpy import ndarray, dtype, float64
from tensorflow.keras import layers, models, losses


def ensure_data_exists() -> None:
    dirpath = os.path.abspath("data")

    if not os.path.exists(dirpath):
        raise FileNotFoundError(errno.ENOENT)
    if not os.path.isdir(dirpath):
        raise FileNotFoundError(errno.ENOTDIR)
    if not os.access(dirpath, os.R_OK):
        raise FileNotFoundError(errno.EACCES)

    for file in ["t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", "train-images.idx3-ubyte", "train-labels.idx1-ubyte"]:
        filepath = os.path.join(dirpath, file)

        if not os.path.exists(filepath):
            raise FileNotFoundError(errno.ENOENT)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(errno.EISDIR)
        if not os.access(filepath, os.R_OK):
            raise FileNotFoundError(errno.EACCES)


def load_images(filename: str) -> tuple[int, int, ndarray[tuple, dtype[float64]]]:
    with io.open(os.path.join(os.path.abspath("data"), filename), mode="rb") as file:
        int.from_bytes(file.read(4), byteorder="big")

        image_count = int.from_bytes(file.read(4), byteorder="big")

        row_count = int.from_bytes(file.read(4), byteorder="big")
        col_count = int.from_bytes(file.read(4), byteorder="big")

        images = np.frombuffer(file.read(), dtype=np.uint8).reshape((image_count, row_count, col_count))

        return row_count, col_count, images


def load_labels(filename: str) -> ndarray[tuple[int], dtype[float64]]:
    with io.open(os.path.join(os.path.abspath("data"), filename), mode="rb") as file:
        int.from_bytes(file.read(4), byteorder="big")

        label_count = int.from_bytes(file.read(4), byteorder="big")

        labels = np.frombuffer(file.read(), dtype=np.uint8).reshape(label_count)

        return labels


def load_all_data():
    train_row_count, train_col_count, train_data = load_images("train-images.idx3-ubyte")
    train_labels = load_labels("train-labels.idx1-ubyte")
    assert len(train_data) == len(train_labels), "Count of training images and labels needs to be equal."

    test_row_count, test_col_count, test_data = load_images("t10k-images.idx3-ubyte")
    test_labels = load_labels("t10k-labels.idx1-ubyte")
    assert len(test_data) == len(test_labels), "Count of test images and labels needs to be equal."

    assert train_col_count == test_col_count and train_row_count == test_row_count, "Size of training and test images needs to be equal."

    train_data = train_data / 255.0
    test_data = test_data / 255.0

    return train_data, train_labels, test_data, test_labels, train_row_count, train_col_count


def make_nn_model(columns, rows):
    model = models.Sequential([
        layers.Input((rows, columns)),
        layers.Flatten(),
        layers.Dense(int(rows * columns / 2), activation='relu'),
        layers.Dense(int(rows * columns / 8), activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10)
    ])
    loss = losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model


def main() -> None:
    try:
        ensure_data_exists()
    except FileNotFoundError:
        print(f"All four files from the original mnist dataset are required to exists in the directory /data: {",".join(["t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", "train-images.idx3-ubyte", "train-labels.idx1-ubyte"])}.")
        exit(1)

    train_x, train_y, test_x, test_y,  rows, columns = load_all_data()

    nn = make_nn_model(columns, rows)
    nn.fit(train_x, train_y, epochs=10)
    nn.evaluate(test_x, test_y, verbose=2)


if __name__ == "__main__":
    main()