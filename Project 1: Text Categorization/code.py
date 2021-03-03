import numpy
import csv
import random
from feature import Bag,Gram
from comparison_plot import alpha_gradient_plot

# Data Reading
with open('train.tsv') as f:
    tsvreader = csv.reader(f, delimiter='\t')
    temp = list(tsvreader)

# Initialization
data = temp[1:]
max_item=1000
random.seed(2021)
numpy.random.seed(2021)

# Feature Extraction
bag=Bag(data,max_item)
bag.get_words()
bag.get_matrix()

gram=Gram(data, dimension=2, max_item=max_item)
gram.get_words()
gram.get_matrix()

# Plot
alpha_gradient_plot(bag,gram,10000,10)
alpha_gradient_plot(bag,gram,100000,10)
