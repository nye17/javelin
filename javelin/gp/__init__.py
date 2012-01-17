# Copyright (c) Anand Patil, 2007

# Where matrix evaluations can be done in chunks, the chunk size will be
# kept below this limit.
chunksize = 1e8

__modules__ = [ 'GPutils',
                'Mean',
                'Covariance',
                'BasisCovariance',
                'FullRankCovariance',
                'NearlyFullRankCovariance',
                'Realization',
                'cov_funs',
              ]


from GPutils import *
from Mean import *
from Covariance import *
from BasisCovariance import *
from FullRankCovariance import *
from NearlyFullRankCovariance import *
from Realization import *
from cov_funs import *

try:
    import SparseCovariance
except ImportError:
    pass
    
