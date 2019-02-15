
# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset_train = pd.read_csv('train.csv')
#X_train = dataset_train.iloc[1:,1:785].values
#y_train = dataset_train.iloc[1:,0:1].values
X = dataset_train.iloc[1:,1:785].values
y = dataset_train.iloc[1:,0:1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_main, X_test, y_train_main, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X_train_main, y_train_main, test_size = 0.1, random_state = 0)

# Testing on real data
dataset_test = pd.read_csv('test.csv')
X_test = dataset_test.iloc[:,0:784]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_val = sc_X.transform(X_val)
X_test = sc_X.transform(X_test)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features = [0])
y_train = ohe.fit_transform(y_train).toarray()
y_val = ohe.transform(y_val).toarray()


# Reshaping the input dataset for CNN
X_train = X_train.reshape(-1,28,28,1)
X_val = X_val.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the CNN
classifier = Sequential()

# Convolution
classifier.add(Convolution2D(32, 5, 5, input_shape = (28, 28, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding more convolutional layer
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dropout(0.5))

classifier.add(Dense(output_dim = 10, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Augumenting the data for better results in less data
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                   samplewise_center=False,  # set each sample mean to 0
                                   featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                   samplewise_std_normalization=False,  # divide each input by its std
                                   zca_whitening=False,  # apply ZCA whitening
                                   rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                                   zoom_range = 0.1, # Randomly zoom image 
                                   width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                                   height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                                   horizontal_flip=False,  # randomly flip images
                                   vertical_flip=False)

test_datagen = ImageDataGenerator()

train_datagen.fit(X_train)
train_datagen.fit(X_val)
test_datagen.fit(X_test)


# Fitting the classifier to dataset
classifier.fit_generator(train_datagen.flow(X_train,y_train, batch_size=64),
                              epochs = 20, validation_data=(X_val, y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0])

# Saving Models and Weights to a JSON and H5 file respectively
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights("model.h5")

# Loading Models and Weights from JSON and H5 file respectively
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred.tolist()
y_pred_main = []
for i in y_pred:
   y_pred_main.append((i.tolist().index(max(i.tolist()))))
y_pred_main = np.asarray(y_pred_main)
y_pred_output = y_pred_main.tolist()

# Writing the result in a csv file
import csv
with open("output.csv", 'a') as outcsv:
    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerow(['ImageId', 'Label'])
    i=1
    for item in y_pred_output:
        writer.writerow([i,item])
        i=i+1

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_main)
