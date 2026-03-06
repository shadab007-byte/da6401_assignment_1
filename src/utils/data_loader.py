import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset(name="fashion_mnist"):
    
    name = name.lower().replace("-", "_")

    if name == "mnist":
        from keras.datasets import mnist
        (X_full, y_full), (X_test, y_test) = mnist.load_data()
    elif name == "fashion_mnist":
        from keras.datasets import fashion_mnist
        (X_full, y_full), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose 'mnist' or 'fashion_mnist'.")

    #flatten 28x28 → 784 and normalize to [0,1]
    X_full = X_full.reshape(-1, 784).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0

    # 90% train, 10% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.1, random_state=42, stratify=y_full
    )

    print(f"[info] {name} → Train:{X_train.shape[0]}, Val:{X_val.shape[0]}, Test:{X_test.shape[0]}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def to_onehot(y, num_classes=10):
    oh = np.zeros((len(y), num_classes), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh


def get_batches(X, y, batch_size, shuffle=True):
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        yield X[idx[start:end]], y[idx[start:end]]


FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
MNIST_CLASSES = [str(i) for i in range(10)]


def get_class_names(dataset_name):
    return FASHION_MNIST_CLASSES if "fashion" in dataset_name.lower() else MNIST_CLASSES
