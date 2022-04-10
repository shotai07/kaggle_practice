from kedro.pipeline import node, Pipeline
from .node import preprocess

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess,
                inputs=['titanic_train', 'titanic_test', 'parameters'],
                outputs='titanic_train_test_prepro',
                name='preprocess',
            ),
        ],
    )