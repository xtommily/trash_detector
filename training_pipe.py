import os
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class TrainingPipe:

    def __init__(self, image_shape=(384, 512, 3), batch_size=16):
        self.image_shape=image_shape
        self.batch_size=batch_size

        self.train_generator=None
        self.val_generator=None

    def prepare_data_generator(self, data_dir=None, val_size=0.2, preprocessing_function=None):
        train_datagen = ImageDataGenerator(rotation_range=30,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           brightness_range=(0.9, 1.1),
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           preprocessing_function=preprocessing_function,
                                           validation_split=val_size, 
                                           fill_mode='nearest')

        self.train_generator = train_datagen.flow_from_directory(directory=data_dir,
                                                    target_size=self.image_shape[:2],
                                                    batch_size=self.batch_size, 
                                                    class_mode="categorical",
                                                    subset="training")
        self.val_generator = train_datagen.flow_from_directory(directory=data_dir,
                                                    target_size=self.image_shape[:2],
                                                    batch_size=self.batch_size, 
                                                    class_mode="categorical",
                                                    subset="validation")

    def show_samples(self):
        # generate samples and plot
        fig, ax = plt.subplots(nrows=1, ncols=min(self.batch_size,4), figsize=(15,15))

        # generate batch of images
        for i in range(min(self.batch_size,4)):
            # convert to unsigned integers
            image = next(self.train_generator)[0].astype('uint8')
        
            # plot image
            ax[i].imshow(image[0])
            ax[i].axis('off')

    def train_model_auto(self, model_name, model=None, epochs=600, patience=30, save_freq=20, with_fine_tuning=True, epochs_ft=50, patience_ft=10, save_freq_ft=5):
        #Freeze filter layer
        model.layers[0].trainable = False

        #Compile and fit model
        model.compile(loss="categorical_crossentropy", 
                            optimizer="adam", 
                            metrics=["accuracy"])

        early_stopping = EarlyStopping(monitor="val_accuracy",
                                       patience=patience,
                                       restore_best_weights=True)

        model_checkpoint = ModelCheckpoint('training/checkpoints/' + model_name + '_weights_best.h5',
                                           monitor='val_accuracy',
                                           mode='max',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=True,
                                           save_freq=save_freq*self.train_generator.samples//self.batch_size)

        history1 = model.fit(x=self.train_generator,
                                 steps_per_epoch=self.train_generator.samples//self.batch_size,
                                 validation_data=self.val_generator,
                                 batch_size=self.batch_size,
                                 epochs=epochs,
                                 callbacks=[early_stopping, model_checkpoint])

        model.save(f'training/models/{model_name.replace(" ","")}')

        # Fine-tuning
        if with_fine_tuning:
            #Unfreeze filter fayer for fine tuning
            model.layers[0].trainable = True
            #Compile and fit model
            model.compile(loss="categorical_crossentropy", 
                               optimizer=Adam(1e-5), #low learning rate for fine-tuning
                               metrics=["accuracy"])
            
            early_stopping = EarlyStopping(monitor="val_accuracy",
                                       patience=patience_ft,
                                       restore_best_weights=True)

            model_checkpoint = ModelCheckpoint('training/checkpoints/' + model_name + '_fine_tuning_weights_best.h5',
                                                monitor='val_accuracy',
                                                mode='max',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=True,
                                                save_freq=save_freq_ft*self.train_generator.samples//self.batch_size)

            history2 = model.fit(x=self.train_generator,
                                     steps_per_epoch=self.train_generator.samples//self.batch_size,
                                     validation_data=self.val_generator,
                                     batch_size=self.batch_size,
                                     epochs=epochs_ft,
                                     callbacks=[early_stopping, model_checkpoint])
            
            model.save(f'training/models/{model_name.replace(" ","")}_fine_tuning')

        print('\n\n')
        self.plot_history(model_name, history1)
        if with_fine_tuning:   
            self.plot_history(model_name+' fine tuning', history2)

    def train_model(self, model_name, model=None, epochs=600, patience=30, save_freq=20, fine_tuning=True):
        #Freeze filter layer
        if not fine_tuning:
            model.layers[0].trainable = False
        else:
            model.layers[0].trainable = True
            model_name += "_fine_tuning"

        #Compile and fit model
        model.compile(loss="categorical_crossentropy", 
                            optimizer="adam" if fine_tuning else Adam(1e-5), 
                            metrics=["accuracy"])

        early_stopping = EarlyStopping(monitor="val_accuracy",
                                       patience=patience,
                                       restore_best_weights=True)

        model_checkpoint = ModelCheckpoint('training/checkpoints/' + model_name + '_weights_best.h5',
                                           monitor='val_accuracy',
                                           mode='max',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=True,
                                           save_freq=save_freq*self.train_generator.samples//self.batch_size)

        history = model.fit(x=self.train_generator,
                                 steps_per_epoch=self.train_generator.samples//self.batch_size,
                                 validation_data=self.val_generator,
                                 batch_size=self.batch_size,
                                 epochs=epochs,
                                 callbacks=[early_stopping, model_checkpoint])

        model.save(f'training/models/{model_name.replace(" ","")}')

        print('\n\n')
        self.plot_history(model_name, history)

    def plot_history(self, model_name, history):
        
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24,7))

        fig.suptitle(model_name, fontsize=16)

        # summarize history for accuracy
        ax[0].plot(history.history['accuracy'])
        ax[0].plot(history.history['val_accuracy'])
        ax[0].set_title('model accuracy')
        ax[0].set_ylabel('accuracy')
        ax[0].set_xlabel('epoch')
        ax[0].legend(['train', 'test'], loc='upper left')

        # summarize history for loss
        ax[1].plot(history.history['loss'])
        ax[1].plot(history.history['val_loss'])
        ax[1].set_title('model loss')
        ax[1].set_ylabel('loss')
        ax[1].set_xlabel('epoch')
        ax[1].legend(['train', 'test'], loc='upper left')

        plt.show()
