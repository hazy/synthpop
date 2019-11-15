import numpy as np
import pandas as pd

# classes
from synthpop.validator import Validator
from synthpop.processor import Processor
from synthpop.method import SampleMethod, CARTMethod, NormMethod, NormRankMethod, PolyregMethod
# global variables
from synthpop.processor import NAN_DICT
from synthpop.method import SAMPLE_METHOD, CART_METHOD, NORM_METHOD, NORMRANK_METHOD, POLYREG_METHOD
from synthpop.method import NA_METHODS


class Synthpop:
    def __init__(self,
                 method=None,
                 visit_sequence=None,
                 # predictor_matrix=None,
                 proper=False,
                 cont_na=None,
                 smoothing=None,
                 default_method=None,
                 numtocat=None,
                 catgroups=None,
                 seed=None):
        self.validator = Validator(self)
        self.processor = Processor(self)

        self.method = method
        self.visit_sequence = visit_sequence
        self.predictor_matrix = None
        self.proper = proper
        self.cont_na = cont_na
        self.smoothing = smoothing
        self.default_method = default_method
        self.numtocat = numtocat
        self.catgroups = catgroups
        self.seed = seed

        self.validator.check_init()

    def fit(self, df, dtypes=None):
        # TODO check df and check/EXTRACT dtypes
        # - all dtypes of df are correct ('int', 'float', 'datetime', 'category', 'bool'; no object)
        # - can map dtypes (if given) correctly to df
        # should create map col: dtype for future use (self.df_dtypes)

        self.df_columns = df.columns.tolist()
        self.n_df_rows, self.n_df_columns = np.shape(df)
        self.df_dtypes = dtypes

        # check processor
        self.validator.check_processor()
        # preprocess
        processed_df = self.processor.preprocess(df, self.df_dtypes)
        self.processed_df_columns = processed_df.columns.tolist()
        self.n_processed_df_columns = len(self.processed_df_columns)

        # check fit
        self.validator.check_fit()
        # fitting
        self._fit(processed_df)

    def _fit(self, df):
        self.saved_methods = {}

        # train
        self.predictor_matrix_columns = self.predictor_matrix.columns.to_numpy()
        for col, visit_step in self.visit_sequence.sort_values().iteritems():
            print('train_{}'.format(col))

            # TODO move that big block of code in a functiom with a map
            # TODO maybe normalise data before fitting and predicting
            # initialise the method
            if self.method[col] == SAMPLE_METHOD:
                col_method = SampleMethod(self.df_dtypes[col], smoothing=self.smoothing[col], proper=self.proper, random_state=self.seed)

            elif self.method[col] == CART_METHOD:
                col_method = CARTMethod(self.df_dtypes[col], smoothing=self.smoothing[col], proper=self.proper, random_state=self.seed)

            elif self.method[col] == NORM_METHOD:
                col_method = NormMethod(self.df_dtypes[col], smoothing=self.smoothing[col], proper=self.proper, random_state=self.seed)

            elif self.method[col] == NORMRANK_METHOD:
                col_method = NormRankMethod(self.df_dtypes[col], smoothing=self.smoothing[col], proper=self.proper, random_state=self.seed)

            elif self.method[col] == POLYREG_METHOD:
                col_method = PolyregMethod(self.df_dtypes[col], proper=self.proper, random_state=self.seed)

            # extract the relevant data and fit the method
            if self.method[col] == SAMPLE_METHOD:
                x = df[col]

                col_method.fit(x)

            elif self.method[col] in NA_METHODS:
                col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() != 0]
                X = df[col_predictors]
                y = df[col]

                # extract the non nan values and only use them for fitting the method
                if col in self.cont_na:
                    not_nan_indices = df[self.processor.processing_dict[NAN_DICT][col]['col_nan_name']] == 0
                    X = X.loc[not_nan_indices]
                    y = y.loc[not_nan_indices]

                col_method.fit(X, y)

            # save the method
            self.saved_methods[col] = col_method

    def generate(self, k=None):
        self.k = k

        # check generate
        self.validator.check_generate()
        # generating
        synth_df = self._generate()
        # postprocess
        processed_synth_df = self.processor.postprocess(synth_df)

        return processed_synth_df

    def _generate(self):
        synth_df = pd.DataFrame(columns=self.visit_sequence.index)
        for col in synth_df.columns:
            synth_df[col] = pd.Series(dtype=self.df_dtypes[col])

        for col, visit_step in self.visit_sequence.sort_values().iteritems():
            print('generate_{}'.format(col))

            # reload the method
            col_method = self.saved_methods[col]

            # extract the relevant data and use the method to predict
            if self.method[col] == SAMPLE_METHOD:
                synth_df[col] = col_method.predict(self.k)

            elif self.method[col] in NA_METHODS:
                col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() != 0]
                X = synth_df[col_predictors]

                synth_df[col] = col_method.predict(X)

                # change all missing values to 0
                if col in self.cont_na:
                    nan_indices = synth_df[self.processor.processing_dict[NAN_DICT][col]['col_nan_name']] != 0
                    synth_df.loc[nan_indices, col] = 0

            synth_df[col] = synth_df[col].astype(self.df_dtypes[col])

        return synth_df

    def save(self):
        pass

    def load(self):
        pass
