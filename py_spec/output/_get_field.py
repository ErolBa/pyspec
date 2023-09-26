## Calculate SPEC B field (and related quantities) purely in python
## attempt to remove pyoculus dependency (easier installation and more transparent calculation)
## TESTED against existing pyoculus-based implementation
## not tested for nonstell-symmetric cases

import numpy as np
from numba import njit, prange

## parameters ---------------------------------------------------------

run_with_numba = True

##---------------------------------------------------------------------

if(run_with_numba):
    run_decorator = njit(fastmath=True, cache=True, parallel=False)
else:
    run_decorator = lambda x: x


def get_field_cov(self, lvol, sarr, tarr, zarr):
    # not tested!!
    Bcontrav = get_field_contrav(self, lvol, sarr, tarr, zarr)
    metric = self.get_metric(sarr, tarr, zarr, lvol)

    Bcov = np.einsum('i...,ij...->j...', Bcontrav, metric, optimize=True)

    return Bcov

def get_field_cart(self, lvol, sarr, tarr, zarr):
    # not tested!!!
    
    geometry = self.input.physics.Igeometry
    rpol = self.input.physics.rpol
    rtor = self.input.physics.rtor
    Rac, Rbc = self.output.Rbc[lvol : lvol + 2]
    Zas, Zbs = self.output.Zbs[lvol : lvol + 2]
    # Ras, Rbs = self.output.Rbs[lvol : lvol + 2]
    # Zac, Zbc = self.output.Zbc[lvol : lvol + 2]
    im = np.array(self.output.im, dtype=np.int)
    in_ = np.array(self.output.in_, dtype=np.int)

    Rarr, Zarr = self.get_coord_transform(geometry, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol)

    Bcontrav = get_field_contrav(self, lvol, sarr, tarr, zarr)

    Bcart = np.empty_like(Bcontrav)

    Bcart[0] = Bcontrav[0] * Rarr[1] + Bcontrav[1] * Rarr[2] + Bcontrav[2] * Rarr[3]
    Bcart[1] = Bcontrav[0] * Zarr[1] + Bcontrav[1] * Zarr[2] + Bcontrav[2] * Zarr[3] 
    Bcart[2] = Bcontrav[2] * Rarr[0]

    return Bcart


def get_field_contrav(self, lvol, sarr, tarr, zarr):
    
    Aze = np.atleast_2d(self.vector_potential.Aze[lvol])
    Ate = np.atleast_2d(self.vector_potential.Ate[lvol])
    # Azo = np.atleast_2d(self.vector_potential.Azo[lvol])
    # Ato = np.atleast_2d(self.vector_potential.Ato[lvol])
    
    lrad = self.input.physics.Lrad[lvol]
    mpol = self.input.physics.Mpol
    im = self.output.im
    _in = self.output.in_
    coord_sing = self.input.physics.Igeometry > 1 and lvol == 0

    jac = self.get_jacobian(sarr, tarr, zarr, lvol)

    if(coord_sing):
        sbar = np.where(0.5*(sarr+1) > 0.0, 0.5*(sarr+1), 0)
        basis = get_zernike_basis(sbar, lrad, mpol)
    
        Bcontrav = get_field_sing(sarr, tarr, zarr, lrad, im, _in, basis, Aze, Ate, jac)
    
    else:
        basis = get_cheby_basis(sarr, lrad)

        Bcontrav = get_field_nonsing(sarr, tarr, zarr, lrad, im, _in, basis, Aze, Ate, jac)

    return Bcontrav


@run_decorator
def get_field_nonsing(sarr, tarr, zarr, lrad, im, _in, basis, Aze, Ate, jac):
    """Generates the contravariant magnetic field (volume without coordinate singularity)
    """    

    Bcontrav = np.zeros((3, len(sarr), len(tarr), len(zarr)))

    for t in range(len(tarr)):
        for z in range(len(zarr)):
            for ii in range(len(im)):
                
                mi = im[ii]
                ni = _in[ii]
                arg = mi * tarr[t] - ni * zarr[z]
                carg = np.cos(arg)
                sarg = np.sin(arg)
                
                for s in range(len(sarr)):
                    for l in range(lrad+1):
                        Bcontrav[0, s, t, z] += (- mi * Aze[ii, l] - ni * Ate[ii, l]) * basis[l, 0, s] * sarg
                        Bcontrav[1, s, t, z] += (- Aze[ii, l] * basis[l, 1, s]) * carg
                        Bcontrav[2, s, t, z] += (Ate[ii, l] * basis[l, 1, s]) * carg

                    # if(not stell_sym): # needs Ato, Azo
                    #     Bcontrav[0, s, t, z] += (- mi * Azo[ii, l+1] - ni * Ato[ii, l+1]) * basis[l, 0, s] * carg
                    #     Bcontrav[1, s, t, z] += (- Azo[ii, l+1] * basis[l, 1, s]) * sarg
                    #     Bcontrav[2, s, t, z] += (Ato[ii, l+1] * basis[l, 1, s]) * sarg

    Bcontrav /= jac
    
    return Bcontrav


@run_decorator
def get_field_sing(sarr, tarr, zarr, lrad, im, _in, basis, Aze, Ate, jac):
    """Generates the contravariant magnetic field (volume with coordinate singularity)
    """    

    Bcontrav = np.zeros((3, len(sarr), len(tarr), len(zarr)))

    for t in range(len(tarr)):
        for z in range(len(zarr)):
            for ii in range(len(im)):
                
                mi = im[ii]
                ni = _in[ii]
                arg = mi * tarr[t] - ni * zarr[z]
                carg = np.cos(arg)
                sarg = np.sin(arg)
                
                for s in range(len(sarr)):
                    for l in range(lrad+1):
                        Bcontrav[0, s, t, z] += (- mi * Aze[ii, l] - ni * Ate[ii, l]) * basis[l, mi, 0, s] * sarg
                        Bcontrav[1, s, t, z] += (- Aze[ii, l] * basis[l, mi, 1, s]) * 0.5 * carg
                        Bcontrav[2, s, t, z] += (Ate[ii, l] * basis[l, mi, 1, s]) * 0.5 * carg

                    # if(not stell_sym): # needs Ato, Azo
                    #     Bcontrav[0, s, t, z] += (- mi * Azo[ii, l+1] - ni * Ato[ii, l+1]) * basis[l, mi, 0, s] * carg
                    #     Bcontrav[1, s, t, z] += (- Azo[ii, l+1] * basis[l, mi, 1, s]) * 0.5 * sarg
                    #     Bcontrav[2, s, t, z] += (Ato[ii, l+1] * basis[l, mi, 1, s]) * 0.5 * sarg

    Bcontrav /= jac

    return Bcontrav


@run_decorator
def get_cheby_basis(sarr, lrad):

    # [radial mode (0->lrad), func/derivative (0/1), radial coord (sarr)]
    basis = np.zeros((lrad+1, 2, sarr.size)) 

    for s in range(len(sarr)):
        
        basis[0, 0, s] = 1.0
        basis[1, 0, s] = sarr[s]
        basis[1, 1, s] = 1.0

        for l in range(2, lrad+1):
            basis[l, 0, s] = 2.0 * sarr[s] * basis[l-1, 0, s] - basis[l-2, 0, s]
            basis[l, 1, s] = 2.0 * basis[l-1, 0, s] + 2.0 * sarr[s] * basis[l-1, 1, s] - basis[l-2, 1, s]

        # basis recombination
        basis[:, 0, s] -= (-1)**np.arange(lrad+1)

    # rescale for conditioning
    for l in range(0, lrad+1):
        basis[l] /= (l + 1.0)
    
    return basis


@run_decorator
def get_zernike_basis(sbararr, lrad, mpol):

    # [radial mode (0->lrad), poloidal mode (0->mpol), func/derivative (0/1), radial coord (sarr)]
    basis = np.zeros((lrad+1, mpol+1, 2, sbararr.size)) 

    for s in range(len(sbararr)):
        r = sbararr[s]
        rm = 1.0 # sbar^m
        rm1 = 0.0 # sbar^(m-1)
        for m in range(mpol+1):

            if(m <= lrad):
                basis[m, m, 0, s] = rm
                basis[m, m, 1, s] = m * rm1

            if(m+2 <= lrad):
                basis[m+2, m, 0, s] = (m+2) * rm * r**2.0 - (m+1) * rm
                basis[m+2, m, 1, s] = (m+2)**2.0 * rm * r - (m+1) * m * rm1 
            
            for l in range(m+4, lrad+1, 2):
                fac1 = l / (l**2.0 - m**2.0)
                fac2 = 4.0 * (l - 1)
                fac3 = (l - 2 + m) **2.0 / (l - 2) + (l - m)**2.0 / l
                fac4 = ((l - 2)**2.0 - m**2.0) / (l - 2.0)

                basis[l, m, 0, s] = fac1 * ( (fac2 * r**2.0 - fac3) * basis[l-2, m, 0, s] - fac4 * basis[l-4, m, 0, s])
                basis[l, m, 1, s] = fac1 * ( 2.0 * fac2 * r * basis[l-2, m, 0, s] 
                                            + (fac2 * r**2.0 - fac3) * basis[l-2, m, 1, s] - fac4 * basis[l-4, m, 1, s])
                
            rm1 = rm
            rm = rm * r
    
    # basis recombination
    for l in range(2, lrad+1, 2):
        basis[l, 0, 0] -= (-1)**(l/2.0)

    if(mpol >= 1):
        for s in range(len(sbararr)):
            for l in range(3, lrad+1, 2):
                basis[l, 1, 0, s] -= (-1)**((l-1)*0.5) * (l+1) * 0.5 * sbararr[s]
                basis[l, 1, 1, s] -= (-1)**((l-1)*0.5) * (l+1) * 0.5
    
    # rescaling
    for m in range(0, mpol+1):
        for l in range(m, lrad+1, 2):
            basis[l, m, :] /= (l + 1.0)

    return basis
