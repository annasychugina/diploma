import json
import pickle
import numpy
import keras

DEPTH_MODEL_FILE = './data/models/depth_0.h5'
VELOCITY_MODEL_FILE = './data/models/velocity_0.h5'
DATA_FILE = './data/test_1.json'
PREPROCESS = lambda x: [(x[0] - 2.395544621652496) / 1.3620799790658589,
                        (x[1] - 11.363975798596117) / 11.554441730305019,
                        (x[2] - 16.271663164182357) / 19.96327294006916]
POSTPROCESS = lambda x: x * 1000

if __name__ == '__main__':
  # with open(MODEL_FILE, 'rb') as input_stream:
  #   model = pickle.loads(input_stream.read())

  # data_sample = json.loads(DATA_FILE)
  # prediction = model.predict(data_sample)
  # print(prediction)

  depth_model = keras.models.load_model(DEPTH_MODEL_FILE)
  velocity_model = keras.models.load_model(VELOCITY_MODEL_FILE)

  with open(DATA_FILE) as input_stream:
    data_sample = json.loads(input_stream.read())

  depth_prediction = POSTPROCESS(depth_model.predict(
                                   numpy.array(PREPROCESS(data_sample)).reshape(1, -1))[0, 0])
  velocity_prediction = POSTPROCESS(velocity_model.predict(
                                   numpy.array(PREPROCESS(data_sample)).reshape(1, -1))[0, 0])

  print(DATA_FILE)
  print(data_sample)
  print('Depth: {0}'.format(depth_prediction))
  print('Velocity: {0}'.format(velocity_prediction))

