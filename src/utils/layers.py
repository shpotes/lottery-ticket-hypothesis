import tensorflow as tf
from tensorflow.keras import layers
from typing import Any, Callable, Dict, Tuple, Union

class LDense(layers.Layer):
    """
    TODO
    """
    def __init__(self, units: int,
                 activation: Union[str, Callable] = None,
                 use_bias: bool = True,
                 kernel_initalizer: tf.Tensor = None,
                 kernel_mask: tf.Tensor = None,
                 **kwargs):
      
        super(LDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        if kernel_initalizer is None:
            self.kernel_initalizer = tf.keras.initializers.GlorotNormal()
        else:
            self.kernel_initalizer = kernel_initalizer

        self.kernel_mask = kernel_mask
        self.masked = (kernel_mask is not None)
    
        
    def build(self, input_shape: Tuple[int]):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer=self.kernel_initalizer)
      
        if self.masked:
            self.m = self.add_weight(shape=(input_shape[-1], self.units),
                                     initializer=self.kernel_mask,
                                     trainable=False)
        
        if self.use_bias:
            self.b = self.add_weight(shape=(self.units),
                                     initializer=tf.zeros_initializer())
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.masked: 
            w = tf.multiply(self.m, self.w)
        else:
            w = self.w

        output = tf.matmul(inputs, w)

        if self.use_bias: 
            output = output + self.b
        
        if self.activation is not None:
            output = self.activation(output)
        
        return output  
