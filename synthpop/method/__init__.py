from synthpop.method.base import Method
from synthpop.method.helpers import proper, smooth
from synthpop.method.sample import SampleMethod
from synthpop.method.cart import CARTMethod
from synthpop.method.norm import NormMethod
from synthpop.method.normrank import NormRankMethod
from synthpop.method.polyreg import PolyregMethod


EMPTY_METHOD = ''
SAMPLE_METHOD = 'sample'
# non-parametric methods
CART_METHOD = 'cart'
# parametric methods
PARAMETRIC_METHOD = 'parametric'
NORM_METHOD = 'norm'
NORMRANK_METHOD = 'normrank'
POLYREG_METHOD = 'polyreg'


ALL_METHODS = set([EMPTY_METHOD, SAMPLE_METHOD, CART_METHOD, PARAMETRIC_METHOD, NORM_METHOD, NORMRANK_METHOD, POLYREG_METHOD])
DEFAULT_METHODS = set([CART_METHOD, PARAMETRIC_METHOD])
INIT_METHODS = set([SAMPLE_METHOD, CART_METHOD, PARAMETRIC_METHOD])
NA_METHODS = set([CART_METHOD, NORM_METHOD, NORMRANK_METHOD, POLYREG_METHOD])


PARAMETRIC_METHOD_MAP = {'int': NORMRANK_METHOD,
                         'float': NORMRANK_METHOD,
                         'datetime': NORMRANK_METHOD,
                         'bool': POLYREG_METHOD,
                         'category': POLYREG_METHOD
                         }

CART_METHOD_MAP = {'int': CART_METHOD,
                   'float': CART_METHOD,
                   'datetime': CART_METHOD,
                   'bool': CART_METHOD,
                   'category': CART_METHOD
                   }

SAMPLE_METHOD_MAP = {'int': SAMPLE_METHOD,
                     'float': SAMPLE_METHOD,
                     'datetime': SAMPLE_METHOD,
                     'bool': SAMPLE_METHOD,
                     'category': SAMPLE_METHOD
                     }

DEFAULT_METHODS_MAP = {CART_METHOD: CART_METHOD_MAP,
                       PARAMETRIC_METHOD: PARAMETRIC_METHOD_MAP
                       }


INIT_METHODS_MAP = DEFAULT_METHODS_MAP.copy()
INIT_METHODS_MAP[SAMPLE_METHOD] = SAMPLE_METHOD_MAP


CONT_NA_METHODS_MAP = {CART_METHOD: CART_METHOD,
                       NORM_METHOD: POLYREG_METHOD,
                       NORMRANK_METHOD: POLYREG_METHOD,
                       POLYREG_METHOD: POLYREG_METHOD
                       }
