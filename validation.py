"""
The :mod: grinlib.validation module includes functions to make validation.
"""



import sys
import statistics
import numpy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def make_validation(model_class, model_settings, validation_settings, postprocess_settings,
                    error_rates, np_samples, np_labels):
  np_processed_labels = numpy.vectorize(postprocess_settings['invert_function'])(np_labels)
  if validation_settings['validation_type'] == 'simple_split':
    result = {'model_rates':[], 'final_rates':[]}
    # print(np_labels)
    # print(np_processed_labels)
    np_train_samples, np_valid_samples, np_train_labels, np_valid_labels = train_test_split(
                                                   np_samples, np_processed_labels,
                                                   test_size=validation_settings['validate_share'],
                                                   random_state=validation_settings['split_seed'])
    # print(np_train_samples)
    # print(np_train_labels)
    # sys.exit()
    model = model_class(**model_settings)
    model.fit(np_train_samples, np_train_labels)
    predictions = model.predict(np_valid_samples)
    # print(predictions)
    # print(np_valid_labels)
    # sys.exit()
    for error_rate in error_rates:
      result['model_rates'].append(error_rate(numpy.squeeze(predictions), np_valid_labels))
      result['final_rates'].append(error_rate(
            numpy.vectorize(postprocess_settings['strict_function'])(numpy.squeeze(predictions)),
            numpy.vectorize(postprocess_settings['strict_function'])(np_valid_labels)))
      # if postprocess_settings['max'] == None:
      #   normalized_predictions = predictions
      # else:
      #   elementwise_max = numpy.ones(numpy.squeeze(predictions).shape) * postprocess_settings['max']
      #   normalized_predictions = numpy.minimum(elementwise_max, numpy.squeeze(predictions))
        # print(np_valid_labels.shape)
        # print(np_valid_labels)
        # for i in range(np_valid_labels.shape[0]):
        #   try:
        #     postprocess_settings['strict_function'](np_valid_labels[i])
        #   except:
        #     print(i)
        #     print(np_valid_labels[i])
        #     raise
      # result['final_rates'].append(error_rate(
      #       numpy.vectorize(postprocess_settings['strict_function'])(normalized_predictions),
      #       numpy.vectorize(postprocess_settings['strict_function'])(np_valid_labels)))
  elif validation_settings['validation_type'] == 'k_fold':
    result = {'fold_rates':[], 'model_rates':[], 'final_fold_rates':[], 'final_rates':[]}
    crossval_iterator = KFold(n_splits=validation_settings['number_of_splits'],
                              random_state=validation_settings.get('split_seed', None),
                              shuffle=True)
    for train_index, valid_index in crossval_iterator.split(np_samples):
      np_train_samples = np_samples[train_index]
      np_valid_samples = np_samples[valid_index]
      np_train_labels = np_processed_labels[train_index]
      np_valid_labels = np_processed_labels[valid_index]
      model = model_class(**model_settings)
      model.fit(np_train_samples, np_train_labels)
      predictions = model.predict(np_valid_samples)
      new_rates = []
      final_rates = []
      for error_rate in error_rates:
        new_rates.append(error_rate(numpy.squeeze(predictions), np_valid_labels))
        final_rates.append(error_rate(
            numpy.vectorize(postprocess_settings['strict_function'])(numpy.squeeze(predictions)),
            numpy.vectorize(postprocess_settings['strict_function'])(np_valid_labels)))
      result['fold_rates'].append(new_rates)
      result['final_fold_rates'].append(final_rates)
    result['model_rates'] = list(map(statistics.mean, zip(*result['fold_rates'])))
    result['final_rates'] = list(map(statistics.mean, zip(*result['final_fold_rates'])))
  else:
    raise ValueError('Unknown validation type {0}'.format(validation_settings['validation_type']))
  return result

