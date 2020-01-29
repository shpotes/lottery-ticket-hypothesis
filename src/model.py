import tensorflow as tf
from typing import Any, Callable, Dict

class LModel:
    def __init__(self, 
                 dataset_cls: type,
                 network_fn: Callable,
                 dataset_args: Dict[str, Any] = None
                 network_args: Dict[str, Any] = None,
                 debug_mode: bool = False):
        
        if dataset_args is None:
            dataset_args = {}
        if network_args is None:
            network_args = {}

        self.data = dataset_cls(**dataset_args)
        self.network = network_fn(
            self.data.input_shape,
            self.data.output_shape,
            **network_args
        )
        self.debug_mode = debug_mode

    @property
    def image_shape(self):
        return self.data.input_shape
        
    def get_current_weights(self) -> Dict[str, tf.Tensor]:
        return dict([(w.name, w)for w in self.network.get_weights()])

    def fit(self,
            batch_size: int = 32,
            epochs: int = 10,
            callbacks: List[tf.keras.callbacks.Callback] = None,
            **opt_args):
        
        if callbacks is None:
            callbacks = []

        self.network.compile(loss=self.loss,
                             optimizer=self.optimizer(**opt_args),
                             metrics=self.metrics)

        train_data = self.data.load('train', batch_size, self.debug_mode)
        val_data = self.data.load('val', batch_size)
        test_data = self.data.load('test', batch_size)
        
        step_per_epochs = self.data.num_train_examples // batch_size
        validation_steps = self.data.num_validation_examples // batch_size

        return self.network.fit(
            train_data,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=validation_data,
            steps_per_epoch=step_per_epochs,
            validation_steps=validation_steps
        )

        
    @property
    def loss(self):
        return 'categorical_crossentropy'

    @property
    def metrics(self):
        return ['accuracy']
    
    def optimizer(self, **kwargs):
        return tf.keras.optimizers.Adam(**kwargs)
