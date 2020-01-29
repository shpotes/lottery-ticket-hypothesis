import tensorflow_datasets as tfds
from typing import Callable, Dict, Any, Union, Tuple

class Dataset:
    def __init__(self, name: str):
        data, info = tfds.load(name, with_info=True)

        self._data = data

        train_size = info.splits['train'].num_examples
        self.num_train_examples = tf.cast(train_size * 0.8, tf.int32)
        self.num_validation_examples = train_size - self.num_train_examples
        self.num_test_examples = info.splits['test'].num_examples
        
        self.input_shape = info.features['image'].shape
        self.num_classes = info.features['label'].num_classes

    def preprocessing(self, raw: tf.Tensor) -> tf.Tensor:
        image = raw['image'] / 256
        label = tf.one_hot(raw['label'], 10)
        return image, label
    
    def load(self, mode: str,
             batch_size: int = 32,
             overfit_mode: bool = False) -> tf.data.Dataset:

        if mode == 'val':
            data = self._data['train']
        else:
            data = self._data[mode]
        
        if mode == 'train':
            data = data.shuffle(10 * batch_size)
            if overfit_mode:
                data = data.take(2 * batch_size)
            else:
                data = data.take(self.num_train_examples)
        if mode == 'val':
            data = data.skip(self.num_train_examples)

        data = (data
                .repeat()
                .batch(batch)
                .map(preprocessing,
                     num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .cache()
                .prefetch(tf.data.experimental.AUTOTUNE))

        return data
