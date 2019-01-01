import sys
import math
import random
import pickle
import numpy

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

import validation

# Во сколько раз больше получаем данных на этапе зашумления
MULTIPLY_NUMBER = 100
# Интервал шума
# MULTIPLY_RANGE = (-1, 1)
MULTIPLY_RANGE = (-0.05, 0.05)

TEST_FILE_NAME = None
# TEST_FILE_NAME = './data.csv'

# SPLIT_SEED = 2
NOISING_SEED = 0

TRAIN_FILE_NAME = './data.csv'

TARGET = 'Depth' # 0.05 noise default
# TARGET = 'Velocity'

# data_columns - индексы колонок в файле из которых берутся данные
DATASET_SETTINGS = {'data_columns':[1, 3, 5]} # , 'addition_range':MULTIPLY_RANGE}

MODEL_SETTINGS = {
  'model_name':'NeuralNet',
  'learning_rate':0.20,
  # 'layers':[{'size':50, 'activation':'relu'}],
  # 'layers':[{'size':50, 'activation':'tanh'}],
  'layers':[{'size':50, 'activation':'relu'}, {'size':30, 'activation':'relu'}],
  'number_of_epochs':2000
}

# MODEL_SETTINGS = {
#   'model_name':'GradientBoostingRegressor',
#   'learning_rate':0.100,
#   'number_of_estimators':1000
#   }
# 
# 

# TRAIN_SETTINGS = {'train_type':'export_model'}
# 
# TRAIN_SETTINGS = {'train_type':'validation', 'validation_type':'simple_split',
#                   'validate_share':0.3,
#                   'split_seed':0}

TRAIN_SETTINGS = {'train_type':'validation', 'validation_type':'k_fold', 'number_of_splits':20,
                  'split_seed':0}


PREPROCESS_SETTINGS = {'data_columns':[1, 3, 5]}

# # POSTPROCESS_SETTINGS = {'function':'idfu'}
# # POSTPROCESS_SETTINGS = {'function':'tanh'}
# POSTPROCESS_SETTINGS = {'function':'tanh', 'min':None, 'max':None}

# valid_postprocess_settings = {
#                 'strict_function':code_to_inversion(POSTPROCESS_SETTINGS['function']),
#                 'invert_function':code_to_function(POSTPROCESS_SETTINGS['function']),
#                 'min':-0.999,
#                 'max':0.999}
POSTPROCESS_SETTINGS = {
                  'strict_function':lambda x: x * 1000,
                  'invert_function':lambda x: x / 1000,
                  'min':None,
                  'max':None}


def to_readable_settings(**kwargs):
  model_settings = kwargs['model_settings']
  if model_settings['model_name'] == 'NeuralNet':
    return 'NeuralNet\nlearning_rate: {0}\nnumber_of_epochs: {1}\nlayers: {2}'.format(
            model_settings['learning_rate'],
            model_settings['number_of_epochs'],
            model_settings['layers'])
  else:
    return model_settings


def code_to_function(function_code):
  if function_code == 'idfu': 
    return lambda x: x
  elif function_code == 'tanh': 
    return math.tanh
    # return lambda x: ((math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)))
  else:
    raise ValueError


def code_to_inversion(function_code):
  if function_code == 'idfu': 
    return lambda x: x
  elif function_code == 'tanh': 
    return math.atanh
    # return lambda x: 0.5 * (math.log(1 + x) - math.log(1 - x))
  else:
    raise ValueError


# Вычисление среднеквадратичного отклонения
def get_mse(np_first, np_second):
  return ((np_first - np_second) ** 2).mean()


# Вычисление среднего отклонения
def get_mae(np_first, np_second):
  return abs(np_first - np_second).mean()


# Функция для увеличения количества данных
def multiply_data(file_name='./data.csv', multiply_number=1,
                  multiply_range=(-1, 1), label='Velocity',
                  data_columns=[1, 3, 5]):
  random.seed(NOISING_SEED)
  result = []
  with open(file_name) as input_stream:
    header_line = input_stream.readline()
    for line in input_stream:
      values = list(map(float, line.split(',')))
      result.append(values)
      for i in range(multiply_number):
        new_values = []
        for i in range(len(values)):
          if i in data_columns:
            new_values.append(values[i] + random.uniform(*multiply_range))
          else:
            new_values.append(values[i])
        result.append(new_values)

  if label == 'Velocity':
    label_index = -1
  elif label == 'Depth':
    label_index = -2
  else:
    raise ValueError('Unknown label: {0}'.format(label))

  # return [values[:6] for values in result], [values[label_index] for values in result]
  return ([[values[1], values[3], values[5]] for values in result],
           [values[label_index] for values in result])


# Функция для загрузки данных
def load_data(file_name):
  list_of_samples = []
  list_of_labels = []
  with open(file_name) as input_stream:
    header_line = input_stream.readline()
    for line in input_stream:
      list_of_samples.append(list(map(float, line.split(',')[:6])))
      list_of_labels.append(float(line.split(',')[-1]))

  return list_of_samples, list_of_labels


# Функция для нормализации данных в препроцессинге
# Нормируются только столбцы с сильным разбросом
def normalize(np_train_samples, column_indices):
  for column in column_indices: # [1, 3, 5]:
    mean = numpy.mean(np_train_samples[:, column])
    std = numpy.std(np_train_samples[:, column])
    np_train_samples[:, column] -= mean
    np_train_samples[:, column] /= std
  return np_train_samples


class NeuralNet(object):
  # def __init__(self, learning_rate, hidden_layer_sizes, number_of_epochs):
  def __init__(self, learning_rate, layers, number_of_epochs):
    # Инициализируем модель
    self._model = Sequential()
    # Добавляем слои
    for layer in layers:
      self._model.add(Dense(units=layer['size'], activation=layer['activation'],
                            input_dim=len(PREPROCESS_SETTINGS['data_columns'])))
    # Добавляем выходной слой
    self._model.add(Dense(units=1, activation='linear'))
    # Инициализируем оптимизирующий алгоритм
    optimizer = SGD(lr=learning_rate)
    # Компилируем модель с заданными настройками
    self._model.compile(loss='mean_squared_error', optimizer=optimizer,
                        metrics=['accuracy', 'mae'])
    self.number_of_epochs = number_of_epochs
  def fit(self, np_samples, np_labels, **kwargs):
    return self._model.fit(np_samples, np_labels, epochs=self.number_of_epochs, **kwargs)
  def predict(self, np_samples):
    return self._model.predict(np_samples)


if __name__ == '__main__':
  # Загружаем зашумленные данные
  train_samples, train_labels = multiply_data(TRAIN_FILE_NAME, multiply_number=MULTIPLY_NUMBER,
                                              multiply_range=MULTIPLY_RANGE, label=TARGET,
                                              data_columns=DATASET_SETTINGS['data_columns'])
  # Переводим данные в форму numpy массивов (матриц)
  np_samples = numpy.array(train_samples)
  np_labels = numpy.array(train_labels)
  # np_labels = numpy.array(list(map(code_to_function(POSTPROCESSING_SETTINGS['function']),
  #                                  train_labels)))
  
  # Нормируем столбцы
  np_samples = normalize(np_samples, range(3))


  # TODO try KNN

  print('Model name: {0}'.format(MODEL_SETTINGS['model_name']))
  if MODEL_SETTINGS['model_name'] == 'GradientBoostingRegressor':
    print('Learning rate: {0}'.format(MODEL_SETTINGS['learning_rate']))
    print('Number of estimators: {0}'.format(MODEL_SETTINGS['number_of_estimators']))
    model_class = GradientBoostingRegressor
    model_settings = {'learning_rate':MODEL_SETTINGS['learning_rate'],
                      'n_estimators':MODEL_SETTINGS['number_of_estimators']}
  elif MODEL_SETTINGS['model_name'] == 'NeuralNet':
    print('Layers: {0}'.format(MODEL_SETTINGS['layers']))
    model_class = NeuralNet
    model_settings = {'learning_rate':MODEL_SETTINGS['learning_rate'],
                      # 'hidden_layer_sizes':MODEL_SETTINGS['hidden_layer_sizes'],
                      # 'hidden_layer_sizes':[MODEL_SETTINGS['layers'][0]['size']],
                      'layers':MODEL_SETTINGS['layers'],
                      'number_of_epochs':MODEL_SETTINGS['number_of_epochs']}
  else:
    raise ValueError('Unknown model {0}'.format(MODEL_SETTINGS['model_name']))

  if TRAIN_SETTINGS['train_type'] == 'validation':
    print('Validation type: {0}'.format(TRAIN_SETTINGS['validation_type']))
    print('Validation seed: {0}'.format(TRAIN_SETTINGS['split_seed']))
    if TRAIN_SETTINGS['validation_type'] == 'k_fold':
      print('Number of splits: {0}'.format(TRAIN_SETTINGS['number_of_splits']))
      print('Split seed: {0}'.format(TRAIN_SETTINGS['split_seed']))
    valid_results = validation.make_validation(model_class,
                                               model_settings,
                                               TRAIN_SETTINGS, POSTPROCESS_SETTINGS,
                                               error_rates=[get_mse, get_mae],
                                               np_samples=np_samples, np_labels=np_labels)
    # print(valid_results)
    mse = valid_results['final_rates'][0]
    mae = valid_results['final_rates'][1]
    #   model = GradientBoostingRegressor(learning_rate=MODEL_SETTINGS['learning_rate'],
    #                                     n_estimators=MODEL_SETTINGS['number_of_estimators'],
    #                                     ) # random_state=RANDOM_SEED)
    #   model.fit(x_train, y_train)

    # # from sklearn.linear_model import LinearRegression
    # # model = LinearRegression()
    # # model.fit(x_train, y_train)

    # # Получаем тестовые предсказания
    # predictions = model.predict(np_valid_samples)

    # # # Вычисляем среднеквадратичное отклонение. (среднее почему-то не работает в встроенной функции)
    # # mse, mae = model.evaluate(np_valid_samples, np_valid_labels, verbose=2)

    # # Вычисляем среднее отклонение
    # mse = get_mse(numpy.squeeze(predictions), np_valid_labels)
    # mae = get_mae(numpy.squeeze(predictions), np_valid_labels)
    print(TARGET)
    print(MULTIPLY_RANGE)
    print(TRAIN_SETTINGS)
    print(to_readable_settings(model_settings=MODEL_SETTINGS))
    print("Mean sqr error:", mse)
    print("Mean abs error:", mae)
    print(valid_results['model_rates'])

    if TEST_FILE_NAME != None:
      test_samples, test_labels = load_data(TEST_FILE_NAME)
      np_test_samples = numpy.array(test_samples)
      np_test_labels = numpy.array(test_labels)
      test_predictions = model.predict(np_test_samples)
      test_mse = get_mse(numpy.squeeze(test_predictions), np_test_labels)
      test_mae = get_mae(numpy.squeeze(test_predictions), np_test_labels)
      print('Test mse:', test_mse)
      print('Test mae:', test_mae)

  elif TRAIN_SETTINGS['train_type'] == 'export_model':
    model = model_class(**model_settings)
    model.fit(np_samples, np_labels)
    # predictions = model.predict(np_valid_samples)

    if MODEL_SETTINGS['model_name'] == 'NeuralNet':
      model._model.save('./data/models/velocity_0.h5')
    elif False:
      with open('./data/models/0.mdl', 'wb') as output_stream:
        output_stream.write(pickle.dumps(model))
    else:
      raise ValueError

