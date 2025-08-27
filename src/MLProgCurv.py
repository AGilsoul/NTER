import joblib
import keras.models
import numpy as np
import pandas as pd
import plotly.express as ex
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Normalization
import mpl_scatter_density
from GradAscent import AMRGrid
from ProbDistFunc import plot_against_2D, pdf_2D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def build_model(train_data, train_labels, num_inputs):
    # create model
    norm_layer = Normalization(axis=1, input_shape=(num_inputs,))
    print('Adapting normalizer ...')
    norm_layer.adapt(train_data, batch_size=128)
    print('Done adapting')
    model = Sequential()
    model.add(norm_layer)
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal', activation=None))  # final layer activation ReLu since T>=0
    # Compile model
    print('Compiling model ...')
    model.compile(loss='mean_squared_error', optimizer='adam')
    print('Fitting model ...')
    model.fit(train_data, train_labels, batch_size=64, epochs=5, validation_split=0.1, verbose=1)
    return model


def trainModel(inputs, outputs, file_name, file_out):
    print(f'Reading {file_name} ...')
    amr_data = pd.read_pickle(file_name)
    all_params = inputs.copy()
    all_params.extend(outputs)
    amr_data = amr_data.filter(all_params)
    amr_train = amr_data.sample(frac=0.8, random_state=0)

    train_data = amr_train[inputs].copy()
    train_labels = amr_train.drop(inputs, axis=1)
    model = build_model(train_data, train_labels, len(inputs))
    model.save(file_out)


def plotData2D(in_data, out_data, in_labels, out_label):
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    plot_against_2D(fig, ax, in_data[0], out_data[1], out_data, in_labels[0], in_labels[1], out_label, bins=[100, 100])
    plt.show()


def plotModelData(model, input_ranges, input_labels, output_label, grid_dim=100):
    x_range = np.linspace(input_ranges[0][0], input_ranges[0][1], grid_dim)
    y_range = np.linspace(input_ranges[1][0], input_ranges[1][1], grid_dim)
    x, y = np.meshgrid(x_range, y_range)

    x_plot, y_plot = x.ravel(), y.ravel()

    plot_data = np.vstack([x_plot, y_plot]).T
    res = model.predict(plot_data).T[0]
    plotData2D(plot_data, res, input_labels, output_label)
    return


def plotActualData(file_name, inputs, output, in_labels, out_label):
    data = pd.read_pickle(file_name)
    in_data = data[inputs]
    output = data[output]
    plotData2D(in_data, output, in_labels, out_label)
    return


def testModel(model, file_name):
    print(f'Reading {file_name} ...')
    amr_data = pd.read_pickle(file_name)
    amr_data = amr_data.filter(['progress_variable', 'MeanCurvature_progress_variable', 'Y(H)', 'Y(OH)'])
    amr_train = amr_data.sample(frac=0.8, random_state=0)
    amr_test = amr_data.drop(amr_train.index)
    test_data = amr_test[['progress_variable', 'MeanCurvature_progress_variable']].copy()
    test_labels = amr_test.drop(['progress_variable', 'MeanCurvature_progress_variable'], axis=1)
    model.evaluate(test_data, test_labels, verbose=1)


def prinCompAnalysis(data, params, n_components=2):
    column_low = [f'pc{i}' for i in range(1,n_components+1)]
    column_up = [f'PC{i}' for i in range(1,n_components+1)]
    data = data.filter(params)
    params.remove('temp')
    amr_data = data.filter(params)
    amr_data = StandardScaler().fit_transform(amr_data)
    amr_out = data['temp']
    print(f'Fitting PCA ...')
    pca = PCA(n_components=n_components)
    PC = pca.fit_transform(amr_data)
    principalDF = pd.DataFrame(data=PC, columns=column_low)
    finalDF = pd.concat([principalDF,data[['temp']]], axis=1)
    finalDF.head()
    PC_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    components = data.columns.tolist()
    components = components[:-1]
    loadingdf = pd.DataFrame(PC_loadings, columns=column_up)
    loadingdf['variable'] = components

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(loadingdf['PC1'], loadingdf['PC2'])
    print(f'{pca.get_feature_names_out()}')
    print(f'explained variance: {pca.explained_variance_}')
    print(f'explained variance ratio: {pca.explained_variance_ratio_}')
    print(f'covariance:\n{pca.get_covariance()}')
    variance_sums = [sum([abs(loadingdf['PC1'][i]) for i in range(len(loadingdf))]), sum([abs(loadingdf['PC2'][i]) for i in range(len(loadingdf))])]
    print(variance_sums)
    for i in range(len(loadingdf)):
        print(f'\tvar {loadingdf["variable"][i]} contribution to: PC1={loadingdf["PC1"][i]}, PC2={loadingdf["PC2"][i]}')
        print(f'\t var {loadingdf["variable"][i]} total contribution: {abs(loadingdf["PC1"][i] / variance_sums[0]) * pca.explained_variance_ratio_[0] + abs(loadingdf["PC2"][i] / variance_sums[1]) * pca.explained_variance_ratio_[1]}')
        print()

    if n_components == 2:
        fig = ex.scatter(x=loadingdf['PC1'], y=loadingdf['PC2'], text=loadingdf['variable'],)
        fig.update_layout(
            height=1000, width=1000,
            title_text='loadings plot')
        fig.update_traces(textposition='bottom center')
        fig.add_shape(type='line',
                      x0=-0, y0=-1, x1=0, y1=1,
                      line=dict(color='RoyalBlue', width=3))
        fig.add_shape(type='line',
                      x0=-1, y0=0, x1=1, y1=0,
                      line=dict(color='RoyalBlue', width=3))
        fig.show()
    return


if __name__ == '__main__':
    data_file = 'res/combGradCurv01930Data.pkl'
    model_file = 'res/Y_model.keras'
    # trainModel(['progress_variable', 'MeanCurvature_progress_variable', 'mixture_fraction'], ['Y(H)', 'Y(OH)'], data_file, model_file)
    # testModel()
    # combine gradient into magnitude
    # '''

    data = pd.read_pickle(data_file)
    data['progress_variable_g_mag'] = (data['progress_variable_gx']**2 + data['progress_variable_gy']**2 + data['progress_variable_gz']**2)**0.5
    prinCompAnalysis(data,
                      ["HeatRelease", "MeanCurvature_progress_variable", "StrainRate_progress_variable", "Y(H)", "Y(H2)",
                       "Y(H2O)", "Y(OH)", "density", "mixture_fraction", "progress_variable", "progress_variable_g_mag", "temp"],
                      n_components=5)
    '''
    prinCompAnalysis(data,
                     ["progress_variable", "MeanCurvature_progress_variable", 'temp'],
                     n_components=2)
    '''
