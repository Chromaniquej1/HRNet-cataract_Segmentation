from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from model.hrnet_attention import seg_hrnet_attention
from data_preprocessing.data_loader import DataGenerator
import os

def train_model(train_imdir, train_maskdir, val_imdir, val_maskdir, weights_path, logpath, input_shape, num_classes, batch_size, epochs):
    # Prepare data generators
    train_generator = DataGenerator(train_imdir, train_maskdir, batch_size, num_classes, input_shape[0], input_shape[2])
    val_generator = DataGenerator(val_imdir, val_maskdir, batch_size, num_classes, input_shape[0], input_shape[2])

    # Create and compile model
    model = seg_hrnet_attention(input_shape, num_classes)
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Define callbacks
    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    csv_logger = CSVLogger(logpath, append=True, separator=';')

    # Train model
    model.fit(
        train_generator,
        epochs=epochs,
        verbose=1,
        callbacks=[model_checkpoint, csv_logger],
        validation_data=val_generator
    )

    model.save_weights(weights_path)
