import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

class ModelEvaluator:

    def __init__(self, model, image_shape=(384, 512, 3)):
        self.model = model
        self.image_shape=image_shape
        self.test_generator=None
        self.y_pred=None

    def prepare_data_generator(self, data_dir=None, preprocessing_function=None):
        test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
        self.test_generator = test_datagen.flow_from_directory(directory=data_dir,
                                                               target_size=self.image_shape[:2],
                                                               batch_size=1,
                                                               shuffle=False,
                                                               class_mode="categorical")
 
    def get_generator(self):
      return self.test_generator

    def get_labels_and_predictions(self):
        if not isinstance(self.y_pred, np.ndarray):
          self.y_pred=np.argmax(self.model.predict(self.test_generator), axis=-1)
        y_true=self.test_generator.labels
        return y_true, self.y_pred

    def get_confusion_matrix(self):
        y_true, y_pred = self.get_labels_and_predictions()
        return confusion_matrix(y_true=y_true, y_pred=y_pred)

    def accuracy_score(self):
        y_true, y_pred = self.get_labels_and_predictions()
        score=accuracy_score(y_true, y_pred)
        print(f'Model accuracy score: {score.round(3)}')

    def confusion_matrix_categorical_accuracy(self, title='Model accuracy'):
        fig, ax = plt.subplots(figsize=(18, 7), nrows=1, ncols=2)
        fig.suptitle(title)
        sns.heatmap(self.get_confusion_matrix(), 
                    ax=ax[0],
                    yticklabels=list(self.test_generator.class_indices.keys()), 
                    xticklabels=list(self.test_generator.class_indices.keys()), 
                    robust=True)
        ax[1].barh(y=list(self.test_generator.class_indices.keys())[::-1],
                  width=(np.diag(self.get_confusion_matrix())/np.sum(self.get_confusion_matrix(), axis=1))[::-1],
                  height=0.6)
        ax[1].set_xlim([0, 1])

        plt.show()

    @staticmethod
    def predict_for_image(model=None, preprocessing_function=None, data_dir=None, img_path=None):
        datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
        generator = datagen.flow_from_directory(directory=data_dir)
        image = load_img(os.path.join(data_dir, img_path))
        plt.imshow(image)
        plt.show()

        image_arr = img_to_array(image)
        image_arr = image_arr.reshape((1, image_arr.shape[0], image_arr.shape[1], image_arr.shape[2]))
        image_arr = preprocessing_function(image_arr)

        pred = model.predict(image_arr)

        df = pd.DataFrame({'probability': pred[0]*100}, index=[k for k,v in generator.class_indices.items()])
        return df.round(2).sort_values(by=['probability'], ascending=False)

    @staticmethod
    def categorical_classification(evaluators):
        categories = [x for x in evaluators[0][1].get_generator().class_indices.keys()]
        n_categ = len(categories)
        indices=list(zip(np.array([categories]*n_categ).T.flatten(), categories*n_categ))
        
        data=dict()
        for name, evaluator in evaluators:
          amounts = np.sum(evaluator.get_confusion_matrix(), axis=1)
          data[name] = evaluator.get_confusion_matrix().flatten() / np.repeat(amounts, n_categ)

        df = pd.DataFrame(data=data, 
                          index=[a+'/'+b if a!=b else a
                                for a,b in indices])

        categ_set = set(categories)
        indx_set = set(df.index)

        return df.loc[categories].sort_index(), df.loc[indx_set-categ_set]
        