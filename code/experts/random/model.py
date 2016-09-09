import numpy as np

def predict(triplets):
    for _ in triplets:
        yield np.random.random()

def train(train_triplets, valid_triplets, model_path):
    pass
