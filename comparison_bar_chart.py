import numpy
import matplotlib.pyplot as plotter

actual = [66.0, 175.0, 96.0, 126.0, 103.0]
predicted = [66.07, 174.9, 95.7, 126.2, 103.1]

number_of_groups = len(actual)

figure, axes = plotter.subplots()
index = numpy.arange(number_of_groups)

bar_width = 0.35
opacity = 0.4


rects1 = plotter.bar(index, actual, bar_width,
                 alpha=opacity,
                 color='b',
                 label='actual')
 
rects2 = plotter.bar(index + bar_width, predicted, bar_width,
                 alpha=opacity,
                 color='g',
                 label='predicted')
 
plotter.xlabel('Experiment')
plotter.ylabel('Scores')
plotter.title('Prediction comparison')
plotter.xticks(index + bar_width, ('1', '4', '9', '12', '15'))
plotter.legend()
 
plotter.tight_layout()
plotter.show()
