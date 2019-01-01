import json
import matplotlib.pyplot as plotter

if __name__ == '__main__':
  with open('arrays.json') as input_stream:
    json_data = json.loads(input_stream.read())
  x_array = json_data['neurons_variating_2']['neurons_num']
  y_array = json_data['neurons_variating_2']['abs_error_rates']
  plotter.plot(x_array, y_array)
  plotter.xlabel('number of neurons')
  plotter.ylabel('abs error rate')

  # array = json_data['1']
  # plotter.plot(array)

  # # plotter.title('This is title')
  # plotter.xlabel('epoch')
  # plotter.ylabel('abs error rate')
  # plotter.grid(True)

  plotter.show()
