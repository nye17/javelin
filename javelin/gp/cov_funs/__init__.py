from isotropic_cov_funs import *
from cov_utils import *
from bases import *
from wrapped_distances import *
import isotropic_cov_funs
from brownian import *
from nsmatern import *



extra_parameters = {
                'gaussian':   {'': ''},
                'pow_exp':    {'pow': 'The exponent in the exponential.'},
                'exponential':{'':''},
                'matern':     {'diff_degree': 'The degree of differentiability of realizations.'},
                'sphere':     {'': ''},
                'quadratic':  {'phi': 'The characteristic (scaled) distance of decorrelation.'},
                'pareto_exp': {'alpha': 'The pareto index.'},
                'pow_tail':   {'beta': 'the exponent in the exponential of the tail'},
                'kepler_exp': {'tcut': 'The cutoff time scale divided by tau_drw; pow=1.5'},
                'wkepler_exp': {'tcut': 'The cutoff time scale divided by tau_drw; pow=1.2'},
                }


for name in extra_parameters.iterkeys():
    locals()[name] = covariance_function_bundle(name, 'isotropic_cov_funs', extra_parameters[name], ampsq_is_diag=True)

nsmatern_extra_params = {'diff_degree': 'A function giving the local degree of differentiability.',
                         'h': 'A function giving the local relative amplitude.'}
nsmatern = covariance_function_bundle('nsmatern','nsmatern',nsmatern_extra_params,ampsq_is_diag=False,with_x=True)
for w in nsmatern.wrappers:
    w.diag_call = nsmatern_diag

