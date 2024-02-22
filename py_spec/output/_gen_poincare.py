### tools for custom evaluation of poincare plots

from numba import njit, objmode, prange
import numpy as np
from scipy.integrate import solve_ivp
# from nbkode import DOP853

from ._get_field import get_cheby_basis_single, get_zernike_basis_single

## parameters ---------------------------------------------------------

## numba signficantly speeds up calculations, can be installed with pip install ... numba
run_with_numba = True

##---------------------------------------------------------------------

if(run_with_numba):
    run_decorator = njit(fastmath=True, cache=True, parallel=False)
else:
    run_decorator = lambda x: x
    


def trace_field_lines(self, lvol, s_init, theta_init, zeta_init, num_torcycles):
    # tracing for an individual volume
    
    s_init = np.atleast_1d(s_init) + 0.0
    theta_init = np.atleast_1d(theta_init) + 0.0
    zeta_init = np.atleast_1d(zeta_init) + 0.0
    num_torcycles = int(num_torcycles)
    
    Aze = np.atleast_2d(self.vector_potential.Aze[lvol])
    Ate = np.atleast_2d(self.vector_potential.Ate[lvol])
    Azo = np.atleast_2d(self.vector_potential.Azo[lvol])
    Ato = np.atleast_2d(self.vector_potential.Ato[lvol])
    im = self.output.im
    _in = self.output.in_
    lrad = self.input.physics.Lrad[lvol]
    mpol = self.input.physics.Mpol
        
    s_data, theta_data =  trace_field_lines_helper(Aze, Ate, Azo, Ato, im, _in, lrad, mpol, s_init, theta_init, zeta_init, num_torcycles)
    # s_data, theta_data =  trace_field_lines_lsoda_helper(Aze, Ate, Azo, Ato, im, _in, lrad, mpol, s_init, theta_init, zeta_init, num_torcycles)

    R_data, Z_data = self.get_coord_transform_list(lvol, s_data, theta_data, np.zeros_like(s_data))
    R_data = R_data[0]
    Z_data = Z_data[0]
    
    return R_data, Z_data


# @njit(fastmath=True, cache=True, parallel=True)
def trace_field_lines_helper(Aze, Ate, Azo, Ato, im, _in, lrad, mpol, s_init, theta_init, zeta_init, num_torcycles):
    # tracing for an individual volume
    
    num_traj = len(s_init)
    
    for t in range(num_traj): # loop that iterates over trajectories
        
        s_data = np.empty(num_torcycles+1)
        theta_data = np.empty(num_torcycles+1)
        
        s_data[0] = s_init[t]
        theta_data[0] = theta_init[t]
        z_span = np.array([zeta_init[t], zeta_init[t] + 2*np.pi])

        for n in prange(num_torcycles): # number of toroidal turns to trace
            
            st0 = np.array([s_data[n], theta_data[n]])
            
            sol = solve_ivp(bfield_nonsing, z_span, st0, method='DOP853', rtol=1e-5, atol=1e-8, vectorized=False,
                            args=(Aze, Ate, Azo, Ato, im, _in, lrad, mpol))
            s_data[n+1] = sol.y[0, -1]
            theta_data[n+1] = sol.y[1, -1]
    
    return s_data, theta_data
   

@run_decorator
def bfield_nonsing(zeta, s_th, Aze, Ate, Azo, Ato, im, _in, lrad, mpol):
    # generates mag field in a non-singular volume
    # sdot = Bs / Bz, thetadot = Bt / Bz
    
    s = s_th[0]
    theta = s_th[1]
    
    Bs = 0.0
    Bt = 0.0
    Bz = 0.0
    
    basis = get_cheby_basis_single(s, lrad)

    for ii in range(mpol+1):
        mi = im[ii]
        ni = _in[ii]
        arg = mi * theta - ni * zeta
        carg = np.cos(arg)
        sarg = np.sin(arg)
        
        for l in range(lrad+1):
            Bs += (- mi * Aze[ii, l] - ni * Ate[ii, l]) * basis[l, 0] * sarg + (- mi * Azo[ii, l] - ni * Ato[ii, l]) * basis[l, 0] * carg
            Bt += (- Aze[ii, l] * basis[l, 1]) * carg + (- Azo[ii, l] * basis[l, 1]) * sarg
            Bz += (Ate[ii, l] * basis[l, 1]) * carg + (Ato[ii, l] * basis[l, 1]) * sarg   

    return np.array([Bs / Bz, Bt / Bz])


@run_decorator
def bfield_sing(zeta, s_th, Aze, Ate, Azo, Ato, im, _in, lrad, mpol):
    # generates mag field in a singular volume
    # sdot = Bs / Bz, thetadot = Bt / Bz
    
    s = s_th[0]
    theta = s_th[1]
    
    Bs = 0.0
    Bt = 0.0
    Bz = 0.0
    
    basis = get_zernike_basis_single(s, lrad, mpol)

    for ii in range(len(im)):
        mi = im[ii]
        ni = _in[ii]
        arg = mi * theta - ni * zeta
        carg = np.cos(arg)
        sarg = np.sin(arg)
        
        for l in range(lrad+1):
            Bs += (- mi * Aze[ii, l] - ni * Ate[ii, l]) * basis[l, mi, 0] * sarg + (- mi * Azo[ii, l] - ni * Ato[ii, l]) * basis[l, mi, 0] * carg
            Bt += (- Aze[ii, l] * basis[l, mi, 1]) * 0.5 * carg + (- Azo[ii, l] * basis[l, mi, 1]) * 0.5 * sarg
            Bz += (Ate[ii, l] * basis[l, mi, 1]) * 0.5 * carg + (Ato[ii, l] * basis[l, mi, 1]) * 0.5 * sarg
        
    return Bs / Bz, Bt / Bz



# ### OLD

# from numbalsoda import dop853, lsoda_sig
# from numba import cfunc, carray


# @run_decorator
# def pack_params(Aze, Ate, Azo, Ato, im, _in, lrad, mpol):
    
#     mpol = mpol + 1
    
#     params = np.empty(4*mpol*(lrad+1) + 2*mpol + 2, dtype=np.float64)
    
#     params[-1] = mpol - 1
#     params[-2] = lrad
#     len_A = mpol * (lrad+1)
    
#     params[-2-mpol:-2] = _in
#     params[-2-2*mpol:-2-mpol] = im
    
#     params[0:len_A] = Aze.flatten()
#     params[len_A:len_A*2] = Ate.flatten()
#     params[len_A*2:len_A*3] = Azo.flatten()
#     params[len_A*3:len_A*4] = Ato.flatten()
    
#     return params


# @run_decorator
# def unpack_params(params):
    
#     # print(params[-5:])
    
#     lrad = int(params[-2])
#     mpol = int(params[-1])
    
#     mpol = mpol + 1
    
#     _in = params[-2-mpol:-2].astype(np.int_)
#     im = params[-2-2*mpol:-2-mpol].astype(np.int_)
    
#     len_A = mpol * (lrad+1)
#     Aze = np.empty((mpol, lrad+1))
    
#     for i in range(mpol):
#         for j in range(lrad+1):
#             Aze[i, j] = params[i*(lrad+1):i*(lrad+1)+j]
    
#     # Aze = params[0:len_A].reshape(mpol, lrad+1)
#     # Ate = params[len_A:len_A*2].reshape(mpol, lrad+1)
#     # Azo = params[len_A*2:len_A*3].reshape(mpol, lrad+1)
#     # Ato = params[len_A*3:len_A*4].reshape(mpol, lrad+1)

#     # Aze = np.zeros((mpol, lrad+1))
#     Ate = np.zeros((mpol, lrad+1))
#     Azo = np.zeros((mpol, lrad+1))
#     Ato = np.zeros((mpol, lrad+1))
    
#     return Aze, Ate, Azo, Ato, im, _in, lrad, mpol-1


# @cfunc(lsoda_sig)
# def rhs(zeta, s_th, out, data):
#     s = s_th[0]
#     theta = s_th[1]

#     Bs = 0.0
#     Bt = 0.0
#     Bz = 0.1

#     params = carray(data, (1,))
#     Aze, Ate, Azo, Ato, im, _in, lrad, mpol = unpack_params(params)

#     basis = get_cheby_basis_single(s, lrad)

#     for ii in range(len(im)):
#         mi = im[ii]
#         ni = _in[ii]
#         arg = mi * theta - ni * zeta
#         carg = np.cos(arg)
#         sarg = np.sin(arg)
        
#         for l in range(lrad+1):
#             Bs += (- mi * Aze[ii, l] - ni * Ate[ii, l]) * basis[l, 0] * sarg + (- mi * Azo[ii, l] - ni * Ato[ii, l]) * basis[l, 0] * carg
#             Bt += (- Aze[ii, l] * basis[l, 1]) * carg + (- Azo[ii, l] * basis[l, 1]) * sarg
#             Bz += (Ate[ii, l] * basis[l, 1]) * carg + (Ato[ii, l] * basis[l, 1]) * sarg
    
#     out[0] = Bs / Bz
#     out[1] = Bt / Bz

# funcptr = rhs.address


# def trace_field_lines_lsoda_helper(Aze, Ate, Azo, Ato, im, _in, lrad, mpol, s_init, theta_init, zeta_init, num_torcycles):
#     # tracing for a volume
   
#     params = pack_params(Aze, Ate, Azo, Ato, im, _in, lrad, mpol)
    
#     num_traj = len(s_init)
    
#     for t in range(num_traj): # loop that iterates over trajectories
        
#         s_data = np.empty(num_torcycles+1)
#         theta_data = np.empty(num_torcycles+1)
        
#         s_data[0] = s_init[t]
#         theta_data[0] = theta_init[t]
#         z_span = np.array([zeta_init[t], zeta_init[t] + 2*np.pi])
        
#         st0 = np.array([s_data[0], theta_data[0]])
        
#         for n in range(num_torcycles): # number of toroidal turns to trace
#             sol, _ = dop853(funcptr, st0, z_span, params)

#             s_data[n+1] = sol[-1,0]
#             theta_data[n+1] = sol[-1,1]
            
#     return s_data, theta_data




# # @run_decorator
# # def bfield_nonsing_nbkode(zeta, s_th, params):
# #     # generates mag field in a non-singular volume
# #     # sdot = Bs / Bz, thetadot = Bt / Bz
    
# #     s = s_th[0]
# #     theta = s_th[1]
    
# #     Bs = 0.0
# #     Bt = 0.0
# #     Bz = 0.0
    
# #     # unpack params
# #     Aze, Ate, Azo, Ato, im, _in, lrad, mpol = unpack_params(params)
    
# #     basis = get_cheby_basis_single(s, lrad)

# #     for ii in range(len(im)):
# #         mi = im[ii]
# #         ni = _in[ii]
# #         arg = mi * theta - ni * zeta
# #         carg = np.cos(arg)
# #         sarg = np.sin(arg)
        
# #         for l in range(lrad+1):
# #             Bs += (- mi * Aze[ii, l] - ni * Ate[ii, l]) * basis[l, 0] * sarg + (- mi * Azo[ii, l] - ni * Ato[ii, l]) * basis[l, 0] * carg
# #             Bt += (- Aze[ii, l] * basis[l, 1]) * carg + (- Azo[ii, l] * basis[l, 1]) * sarg
# #             Bz += (Ate[ii, l] * basis[l, 1]) * carg + (Ato[ii, l] * basis[l, 1]) * sarg
        
# #     return np.array([Bs / Bz, Bt / Bz])
