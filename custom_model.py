from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

class CustomModel():

    def __init__(self, base_model=None, top_hidden_layers=(20,10), n_classes=5, activation="sigmoid", dropout=-1):
        self.base_model=base_model
        self.top_model=Sequential()

        self.top_hidden_layers=top_hidden_layers
        self.activation=activation
        self.dropout=dropout

        self.n_classes=n_classes

    def get_model(self):
        model = Sequential()

        #Define top model 
        self.top_model.add(Flatten())
        for layer in self.top_hidden_layers:
            self.top_model.add(Dense(layer, self.activation))

            if 0<self.dropout<=1:
                self.top_model.add(Dropout(self.dropout))

        self.top_model.add(Dense(self.n_classes, "softmax"))

        #Merge base and top with functional approach
        model.add(self.base_model)
        model.add(self.top_model)

        return model


