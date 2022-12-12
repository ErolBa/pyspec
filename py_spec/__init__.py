# import of all SPEC-related python scripts.
__version__ = "3.1.0"

from .ci import test
from .input.spec_namelist import SPECNamelist
from .output.spec import SPECout
from .slab.spec_slab import SPECslab, input_dict
from .slab.hmhd_slab import HMHDslab
