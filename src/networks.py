from utils.layers import LDense
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import Dropout, Flatten
from typing import Tuple

def mlp(input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        layer_size: List[int] = [128, 128, 128],
        dropout_rate: float = 0.2):

    num_classes = output_shape[0]
    
    model = Sequential([Flatten(input_shape=input_shape)])
    for units in layer_size:
        model.add(LDense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax')

    return model
