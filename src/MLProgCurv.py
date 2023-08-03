import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold, cross_val_score
from scikeras.wrappers import KerasRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from GradAscent import AMRGrid
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.config.experimental.list_physical_devices('GPU'))


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_shape=(2,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation=None))  # final layer activation ReLu since T>=0
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def main():
    file_name = 'res/combGradCurv01930Data.pkl'
    print(f'Reading {file_name} ...')
    amr_data = pd.read_pickle(file_name)
    amr_data = amr_data.filter(['progress_variable', 'temp', 'MeanCurvature_progress_variable'])
    amr_train = amr_data.sample(frac=0.8, random_state=0)
    amr_test = amr_data.drop(amr_train.index)

    train_data = amr_train.copy()
    train_labels = train_data.pop('temp').values
    test_data = amr_test.copy()
    test_labels = test_data.pop('temp').values
    train_data = normalize(train_data.values, axis=0)
    test_data = normalize(test_data.values, axis=0)

    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(model=baseline_model, epochs=5, batch_size=64, verbose=1)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, train_data, train_labels, cv=kfold, scoring='neg_mean_squared_error')
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    print("Predicting temperature of point with c=0.8, K=300")
    print(f'T={pipeline.predict([0.8, 300])}')

    pipeline.named_steps['mlp'].model_save('res/test_model.h5')
    pipeline.named_steps['mlp'].model = None
    joblib.dump(pipeline, 'res/sklearn_pipeline.pkl')
    del pipeline


if __name__ == '__main__':
    # load model maybe?
    # pipeline = joblib.load('res/sklearn.pipeline.pkl')
    # pipeline.named_steps['mlp'].model = load_model('res/keras_model.h5')
    main()
