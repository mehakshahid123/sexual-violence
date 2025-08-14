
import os

def test_tree_exists():
    assert os.path.exists("src/data/make_dataset.py")
    assert os.path.exists("src/features/build_features.py")
    assert os.path.exists("src/models/train_classification.py")
    assert os.path.exists("src/models/train_regression.py")
    assert os.path.exists("src/models/evaluate.py")
    assert os.path.exists("src/visualization/eda.py")
