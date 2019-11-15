import numpy as np

from load import columns


# input variables
method = None
# method = 'cart'
# method = 'parametric'
# method = ['sample'] + [''] + ['cart' for _ in range(2, len(df.columns))]

visit_sequence = None
# visit_sequence = np.random.permutation(range(len(columns)))
# visit_sequence = [5, 2]
# visit_sequence = ['relationship', 'educational-num', 'race', 'age', 'workclass', 'income']

proper = False
# proper = True

smoothing = None
# smoothing = 'density'
# smoothing = {'capital-gain': 'density'}
# smoothing = {'capital-gain': 'density',
#              'workclass': None}

cont_na = None
# cont_na = {'workclass': ['?']}
# cont_na = {'age': [36, 38]}

numtocat = None
# numtocat = ['educational-num']

catgroups = None
# catgroups = {'educational-num': 5}
