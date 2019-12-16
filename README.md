# synthpop
Python implementation of the R package synthpop


## Installation from sources

```bash
git clone git@github.com:hazy/synthpop.git
cd synthpop
pip install -r requirements.txt
python setup.py install
```

## [Project roadmap](ROADMAP.md)


## Examples

### [Adult dataset](datasets/README.md):
```
In [1]: from datasets.adult import df

In [2]: df.head()
Out[2]:
   age          workclass  fnlwgt   education  educational-num       marital-status          occupation    relationship    race   gender  capital-gain  capital-loss  hours-per-week  native-country  income
0   39          State-gov   77516   Bachelors               13        Never-married        Adm-clerical   Not-in-family   White     Male          2174             0              40   United-States   <=50K
1   50   Self-emp-not-inc   83311   Bachelors               13   Married-civ-spouse     Exec-managerial         Husband   White     Male             0             0              13   United-States   <=50K
2   38            Private  215646     HS-grad                9             Divorced   Handlers-cleaners   Not-in-family   White     Male             0             0              40   United-States   <=50K
3   53            Private  234721        11th                7   Married-civ-spouse   Handlers-cleaners         Husband   Black     Male             0             0              40   United-States   <=50K
4   28            Private  338409   Bachelors               13   Married-civ-spouse      Prof-specialty            Wife   Black   Female             0             0              40            Cuba   <=50K
```

### synthpop

1. Default parameters for the Adult dataset.
```
In [1]: from synthpop import Synthpop

In [2]: from datasets.adult import df, dtypes

In [3]: spop = Synthpop()

In [4]: spop.fit(df, dtypes)
train_age
train_workclass
train_fnlwgt
train_education
train_educational-num
train_marital-status
train_occupation
train_relationship
train_race
train_gender
train_capital-gain
train_capital-loss
train_hours-per-week
train_native-country
train_income

In [5]: synth_df = spop.generate(len(df))
generate_age
generate_workclass
generate_fnlwgt
generate_education
generate_educational-num
generate_marital-status
generate_occupation
generate_relationship
generate_race
generate_gender
generate_capital-gain
generate_capital-loss
generate_hours-per-week
generate_native-country
generate_income

In [6]: synth_df.head()
Out[6]:
   age   workclass  fnlwgt education  educational-num       marital-status      occupation    relationship    race   gender  capital-gain  capital-loss  hours-per-week  native-country  income
0   21           ?  213055      11th                7        Never-married               ?   Not-in-family   Other   Female             0             0              30   United-States   <=50K
1   23     Private  150683   HS-grad                9        Never-married    Adm-clerical   Not-in-family   White   Female             0             0              40   United-States   <=50K
2   61     Private  191417      10th                6              Widowed           Sales   Not-in-family   Black   Female             0             0              32   United-States   <=50K
3   50     Private  190762   HS-grad                9             Divorced           Sales   Not-in-family   White     Male             0             0              60   United-States   <=50K
4   42   Local-gov  255675   HS-grad                9   Married-civ-spouse   Other-service         Husband   Black     Male             0             0              40   United-States   <=50K

In [7]: spop.method
Out[7]:
age                sample
workclass            cart
fnlwgt               cart
education            cart
educational-num      cart
marital-status       cart
occupation           cart
relationship         cart
race                 cart
gender               cart
capital-gain         cart
capital-loss         cart
hours-per-week       cart
native-country       cart
income               cart
dtype: object

In [8]: spop.visit_sequence
Out[8]:
age                 0
workclass           1
fnlwgt              2
education           3
educational-num     4
marital-status      5
occupation          6
relationship        7
race                8
gender              9
capital-gain       10
capital-loss       11
hours-per-week     12
native-country     13
income             14
dtype: int64

In [9]: spop.predictor_matrix
Out[9]:
                 age  workclass  fnlwgt  education  educational-num  marital-status  occupation  relationship  race  gender  capital-gain  capital-loss  hours-per-week  native-country  income
age                0          0       0          0                0               0           0             0     0       0             0             0               0               0       0
workclass          1          0       0          0                0               0           0             0     0       0             0             0               0               0       0
fnlwgt             1          1       0          0                0               0           0             0     0       0             0             0               0               0       0
education          1          1       1          0                0               0           0             0     0       0             0             0               0               0       0
educational-num    1          1       1          1                0               0           0             0     0       0             0             0               0               0       0
marital-status     1          1       1          1                1               0           0             0     0       0             0             0               0               0       0
occupation         1          1       1          1                1               1           0             0     0       0             0             0               0               0       0
relationship       1          1       1          1                1               1           1             0     0       0             0             0               0               0       0
race               1          1       1          1                1               1           1             1     0       0             0             0               0               0       0
gender             1          1       1          1                1               1           1             1     1       0             0             0               0               0       0
capital-gain       1          1       1          1                1               1           1             1     1       1             0             0               0               0       0
capital-loss       1          1       1          1                1               1           1             1     1       1             1             0               0               0       0
hours-per-week     1          1       1          1                1               1           1             1     1       1             1             1               0               0       0
native-country     1          1       1          1                1               1           1             1     1       1             1             1               1               0       0
income             1          1       1          1                1               1           1             1     1       1             1             1               1               1       0
```

2. Visit sequence for the Adult dataset.
```
In [1]: from synthpop import Synthpop

In [2]: from datasets.adult import df, dtypes

In [3]: spop = Synthpop(visit_sequence=[0, 1, 5, 3, 2])

In [4]: spop.fit(df, dtypes)
train_age
train_workclass
train_marital-status
train_education
train_fnlwgt

In [5]: synth_df = spop.generate(len(df))
generate_age
generate_workclass
generate_marital-status
generate_education
generate_fnlwgt

In [6]: synth_df.head()
Out[6]:
   age          workclass  fnlwgt      education       marital-status
0   57   Self-emp-not-inc  327901    Prof-school   Married-civ-spouse
1   24            Private   34568      Assoc-voc        Never-married
2   50            Private  256861        HS-grad   Married-civ-spouse
3   28            Private  186239   Some-college        Never-married
4   38            Private  216129      Bachelors             Divorced

In [7]: spop.method
Out[7]:
age                sample
workclass            cart
fnlwgt               cart
education            cart
educational-num      cart
marital-status       cart
occupation           cart
relationship         cart
race                 cart
gender               cart
capital-gain         cart
capital-loss         cart
hours-per-week       cart
native-country       cart
income               cart
dtype: object

In [8]: spop.visit_sequence
Out[8]:
age               0
workclass         1
fnlwgt            4
education         3
marital-status    2
dtype: int64

In [9]: spop.predictor_matrix
Out[9]:
                age  workclass  fnlwgt  education  marital-status
age               0          0       0          0               0
workclass         1          0       0          0               0
fnlwgt            1          1       0          1               1
education         1          1       0          0               1
marital-status    1          1       0          0               0
```
