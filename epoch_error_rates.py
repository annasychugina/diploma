import os
import numpy
import matplotlib.pyplot as plotter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pipeline

DATASET_SETTINGS = {'data_columns':[1, 3, 5], 'target':'Depth', 'file_name':'./data.csv',
                    'multiply_number':100, 'multiply_range':(-0.05, 0.05)}

# DATASET_SETTINGS = {'data_columns':[1, 3, 5], 'target':'Velocity', 'file_name':'./data.csv',
#                     'multiply_number':100, 'multiply_range':(-1, 1)}
# 
MODEL_SETTINGS = {
  'model_name':'NeuralNet',
  'learning_rate':0.15,
  # 'layers':[{'size':30, 'activation':'relu'}],
  'layers':[{'size':50, 'activation':'relu'}, {'size':20, 'activation':'relu'}],
  # 'layers':[{'size':10, 'activation':'tanh'}, {'size':10, 'activation':'tanh'}],
  'number_of_epochs':1000
}

TRAIN_SETTINGS = {'train_type':'validation', 'validation_type':'k_fold', 'number_of_splits':20,
                  'split_seed':0}


POSTPROCESS_SETTINGS = {
                  'strict_function':lambda x: x * 1000,
                  'invert_function':lambda x: x / 1000,
                  'min':None,
                  'max':None}

if __name__ == '__main__':
  train_samples, train_labels = pipeline.multiply_data(DATASET_SETTINGS['file_name'],
                                              multiply_number=DATASET_SETTINGS['multiply_number'],
                                              multiply_range=DATASET_SETTINGS['multiply_range'],
                                              label=DATASET_SETTINGS['target'],
                                              data_columns=DATASET_SETTINGS['data_columns'])
  np_samples = numpy.array(train_samples)
  np_labels = numpy.array(train_labels)
  np_samples = pipeline.normalize(np_samples, range(3))

  print('Model name: {0}'.format(MODEL_SETTINGS['model_name']))
  if MODEL_SETTINGS['model_name'] == 'GradientBoostingRegressor':
    print('Learning rate: {0}'.format(MODEL_SETTINGS['learning_rate']))
    print('Number of estimators: {0}'.format(MODEL_SETTINGS['number_of_estimators']))
    model_class = GradientBoostingRegressor
    model_settings = {'learning_rate':MODEL_SETTINGS['learning_rate'],
                      'n_estimators':MODEL_SETTINGS['number_of_estimators']}
  elif MODEL_SETTINGS['model_name'] == 'NeuralNet':
    print('Layers: {0}'.format(MODEL_SETTINGS['layers']))
    model_class = pipeline.NeuralNet
    model_settings = {'learning_rate':MODEL_SETTINGS['learning_rate'],
                      'layers':MODEL_SETTINGS['layers'],
                      'number_of_epochs':MODEL_SETTINGS['number_of_epochs']}
  else:
    raise ValueError('Unknown model {0}'.format(MODEL_SETTINGS['model_name']))

  np_processed_labels = numpy.vectorize(POSTPROCESS_SETTINGS['invert_function'])(np_labels)
  # np_train_samples, np_valid_samples, np_train_labels, np_valid_labels = train_test_split(
  #                                                np_samples, np_processed_labels,
  #                                                test_size=VALIDATION_SETTINGS['validate_share'],
  #                                                random_state=VALIDATION_SETTINGS['split_seed'])
  # model = model_class(**model_settings)
  # out = model.fit(np_train_samples, np_train_labels)
  # print(out)

  crossval_iterator = KFold(n_splits=TRAIN_SETTINGS['number_of_splits'],
                            random_state=TRAIN_SETTINGS.get('split_seed', None),
                            shuffle=True)
  validation_losses = []
  mae_losses = []
  for train_index, valid_index in crossval_iterator.split(np_samples):
    np_train_samples = np_samples[train_index]
    np_valid_samples = np_samples[valid_index]
    np_train_labels = np_processed_labels[train_index]
    np_valid_labels = np_processed_labels[valid_index]
    model = model_class(**model_settings)
    out = model.fit(np_train_samples, np_train_labels,
                    validation_data=(np_valid_samples, np_valid_labels))
    # print(type(out))
    # print(out.epoch)
    # print(out.history)
    # input()
    validation_losses.append(out.history['val_loss'])
    mae_losses.append(out.history['mean_absolute_error'])

  print('Validation losses')
  print(validation_losses)
  print('Mean Average Error losses')
  print(mae_losses)

  mean_losses = []
  for i in range(len(validation_losses[0])):
    epoch_losses = [loss[i] for loss in validation_losses]
    mean_losses.append(sum(epoch_losses) / len(epoch_losses))

  mean_mae_losses = []
  for i in range(len(mae_losses[0])):
    epoch_losses = [loss[i] for loss in mae_losses]
    mean_mae_losses.append(sum(epoch_losses) / len(epoch_losses))

  print('==========================')
  try:
    print('DATASET_SETTINGS = {0}'.format(DATASET_SETTINGS))
    print('MODEL_SETTINGS = {0}'.format(MODEL_SETTINGS))
    print('TRAIN_SETTINGS = {0}'.format(TRAIN_SETTINGS))
    print('POSTPROCESS_SETTINGS = {0}'.format(POSTPROCESS_SETTINGS))
  except:
    pass
  print('Mean losses')
  print(mean_losses)
  print('Mean MAE losses')
  print(mean_mae_losses)

  # plotter.plot(mean_losses)
  # # plotter.plot(mean_mae_losses)
  # plotter.xlabel('epoch')
  # plotter.ylabel('error rate')
  # plotter.show()

  plotter.figure(1)
  layers_code = '-'.join([str(layer['size']) + '/' + layer['activation']
                          for layer in MODEL_SETTINGS['layers']])
  plotter.suptitle(DATASET_SETTINGS['target'] + ' learning graphs for ' + layers_code +
                   ', lr=' + str(MODEL_SETTINGS['learning_rate']) + ' Neural Net')
  plotter.subplot(211)
  plotter.plot(mean_losses)
  plotter.ylabel('MSE error rate')
  plotter.subplot(212)
  plotter.plot(mean_mae_losses)
  plotter.ylabel('MAE error rate')
  plotter.xlabel('epoch')
  counter = 0
  while os.path.exists('./data/learning_graph_' + str(counter) + '.png'):
    counter += 1
  plotter.savefig('./data/learning_graph_' + str(counter) + '.png')
  # plotter.show()

