import pandas as pd
from sklearn.model_selection import train_test_split


data_path = '/keybase/team/anon_ai/datasets/Spikes/Generator/others/adult/adult.csv'

dtypes = {'age': 'int',
          'workclass': 'category',
          'fnlwgt': 'int',
          'education': 'category',
          'educational-num': 'int',
          'marital-status': 'category',
          'occupation': 'category',
          'relationship': 'category',
          'race': 'category',
          'gender': 'category',
          'capital-gain': 'int',
          'capital-loss': 'int',
          'hours-per-week': 'int',
          'native-country': 'category',
          'income': 'category'
          }

df = pd.read_csv(data_path, dtype=dtypes)
df, test_df = train_test_split(df, test_size=.2, stratify=df['income'], random_state=2019)

columns = df.columns.tolist()  # in synthpop that's 'vars'

# temp
# import numpy as np
# df.loc[0:10, 'educational-num'] = np.nan

# df = df.loc[[i in [18, 20, 36, 38] for i in df['age']]]
# df.loc[df['age'] == 18, 'age'] = np.nan
