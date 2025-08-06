from tensorflow.keras import layers, models

def build_iotfecnn(input_shape, num_classes=2):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 1, activation='relu')(inputs)
    x1 = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(32, 1, activation='relu')(x1)
    x2 = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(16, 1, activation='relu')(x2)
    x3 = layers.MaxPooling1D(1)(x)

    f1 = layers.Flatten()(x1)
    f2 = layers.Flatten()(x2)
    f3 = layers.Flatten()(x3)
    f = layers.concatenate([f1, f2, f3])
    f = layers.Dense(512, activation='relu')(f)
    f = layers.BatchNormalization()(f)
    f = layers.Dropout(0.2)(f)
    f = layers.Dense(256, activation='relu')(f)
    f = layers.Dense(10, activation='relu', name='deep_feature')(f)

    if num_classes == 2:
        out = layers.Dense(1, activation='sigmoid')(f)
        loss_fn = 'binary_crossentropy'
    else:
        out = layers.Dense(num_classes, activation='softmax')(f)
        loss_fn = 'sparse_categorical_crossentropy'

    model = models.Model(inputs, out)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model

