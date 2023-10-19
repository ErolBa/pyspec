## Calculate geometric quantities for a SPEC equilibrium
## gives metric, jacobian, grid...

import numpy as np
from numba import njit

## parameters ---------------------------------------------------------

## numba signficantly speeds up calculations, can be installed with pip install ... numba
run_with_numba = True

##---------------------------------------------------------------------

if(run_with_numba):
    run_decorator = njit(fastmath=True, cache=True, parallel=False)
else:
    run_decorator = lambda x: x


def get_jacobian(self, sarr, tarr, zarr, lvol):

    geometry = self.input.physics.Igeometry
    rpol = self.input.physics.rpol
    rtor = self.input.physics.rtor
    im = np.array(self.output.im, dtype=int)
    in_ = np.array(self.output.in_, dtype=int)
    Rac, Rbc = self.output.Rbc[lvol : lvol + 2]
    Zas, Zbs = self.output.Zbs[lvol : lvol + 2]
    # Ras, Rbs = self.output.Rbs[lvol : lvol + 2]
    # Zac, Zbc = self.output.Zbc[lvol : lvol + 2]

    return get_jacobian_helper_v2(geometry, rpol, rtor, im, in_, Rac, Rbc, Zas, Zbs, sarr, tarr, zarr, lvol)


@run_decorator
def get_jacobian_helper_v1(geometry, rpol, rtor, im, in_, Rac, Rbc, Zas, Zbs, sarr, tarr, zarr, lvol):

    jac = np.zeros((len(sarr), len(tarr), len(zarr)))

    sbar = np.divide(np.add(sarr, 1.0), 2.0)
    
    if(geometry == 1): # slab/cartesian
        
        for t in range(len(tarr)):
            for z in range(len(zarr)):
                cos = np.cos(im * tarr[t] - in_ * zarr[z])
                Rarr1 = np.sum(0.5 * (Rbc - Rac) * cos, axis=0)
        
                jac[:,t,z] = Rarr1 * rpol * rtor

    elif(geometry == 2): # cylindrical
        pass
        for t in range(len(tarr)):
            for z in range(len(zarr)):
                cos = np.cos(im * tarr[t] - in_ * zarr[z])
                for s in range(len(sarr)):

                    if(lvol > 0):
                        fac0 = sbar[s] ** np.ones_like(im)
                        fac1 =  0.5 ** np.ones_like(im)
                    else:
                        fac0 = sbar[s] ** (im + 1.0)
                        fac1 =  (im + 1.0) / 2.0 * sbar[s] ** (im)
                        fac0[im==0] = sbar[s]
                        fac1[im==0] = 0.5

                    dR1 = Rac + fac0 * (Rbc - Rac)
                    Rarr0 = np.sum(dR1 * cos, axis=0)
                    Rarr1 = np.sum(fac1 * (Rbc - Rac) * cos, axis=0)

                    jac[s,t,z] = Rarr1 * Rarr0

    elif(geometry == 3):
        # pass
        for t in range(len(tarr)):
            for z in range(len(zarr)):
                cos = np.cos(im * tarr[t] - in_ * zarr[z])
                sin = np.sin(im * tarr[t] - in_ * zarr[z])
                for s in range(len(sarr)):

                    if(lvol > 0):
                        fac0 = sbar[s] ** np.ones_like(im)
                        fac1 =  0.5 ** np.ones_like(im)
                    else:
                        fac0 = sbar[s] ** im
                        fac1 =  (im / 2.0) * sbar[s] ** (im - 1.0)
                        fac0[im==0] = sbar[s]**2
                        fac1[im==0] = sbar[s]

                    dR1 = Rac + fac0 * (Rbc - Rac)
                    Rarr0 = np.sum(dR1 * cos, axis=0)
                    Rarr1 = np.sum(fac1 * (Rbc - Rac) * cos, axis=0)
                    Rarr2 = np.sum(-im * dR1 * sin, axis=0)

                    dZ1 = Zas + fac0 * (Zbs - Zas)
                    Zarr1 = np.sum(fac1 * (Zbs - Zas) * sin, axis=0)
                    Zarr2 = np.sum(im * dZ1 * cos, axis=0)

                    jac[s,t,z] = Rarr0 * (Rarr2 * Zarr1 - Rarr1 * Zarr2)       

    return jac


@run_decorator
def get_jacobian_helper_v2(geometry, rpol, rtor, im, in_, Rac, Rbc, Zas, Zbs, sarr, tarr, zarr, lvol):

    jac = np.empty((len(sarr), len(tarr), len(zarr)))

    Rarr, Zarr = get_coord_transform_helper(geometry, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol)

    if(geometry == 1): # slab/cartesian
        jac = Rarr[1] * (rpol * rtor)

    elif(geometry == 2): # cylindrical
        jac = Rarr[1] * Rarr[0]

    elif(geometry == 3): # toroidal
        jac = Rarr[0] * (Rarr[2] * Zarr[1] - Rarr[1] * Zarr[2])       

    return jac


def get_metric(self, sarr, tarr, zarr, lvol):
    """Computes the contravariant metric g_{ij} = e_i \dot e_j
    """    
    geometry = self.input.physics.Igeometry
    rpol = self.input.physics.rpol
    rtor = self.input.physics.rtor
    Rac, Rbc = self.output.Rbc[lvol : lvol + 2]
    Zas, Zbs = self.output.Zbs[lvol : lvol + 2]
    # Ras, Rbs = self.output.Rbs[lvol : lvol + 2]
    # Zac, Zbc = self.output.Zbc[lvol : lvol + 2]
    im = np.array(self.output.im, dtype=int)
    in_ = np.array(self.output.in_, dtype=int)

    # return get_metric_helper(geometry, rpol, rtor, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol)
    return get_metric_helper_v2(geometry, rpol, rtor, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol)


# slightly faster than v2
@run_decorator
def get_metric_helper_v1(geometry, rpol, rtor, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol):
    
    sbar = np.divide(np.add(sarr, 1.0), 2.0)

    metric = np.zeros((3, 3, len(sarr), len(tarr), len(zarr)))

    if(geometry == 1): # slab/cartesian
        
        for t in range(len(tarr)):
            for z in range(len(zarr)):
                cos = np.cos(im * tarr[t] - in_ * zarr[z])
                sin = np.sin(im * tarr[t] - in_ * zarr[z])
                for s in range(len(sarr)):

                    dR1 = Rac + sbar[s] * (Rbc - Rac)
                    Rarr = np.empty((3))
                    Rarr[0] = np.sum(0.5 * (Rbc - Rac) * cos, axis=0)
                    Rarr[1] = np.sum(-im * dR1 * sin, axis=0)
                    Rarr[2] = np.sum(in_ * dR1 * sin, axis=0)

                    for i in range(3):
                        for j in range(3):
                            metric[i,j,s,t,z] = Rarr[i] * Rarr[j]

        metric[1, 1] += rpol ** 2
        metric[2, 2] += rtor ** 2
    
    elif(geometry == 2): # cylindrical
        for t in range(len(tarr)):
            for z in range(len(zarr)):
                cos = np.cos(im * tarr[t] - in_ * zarr[z])
                sin = np.sin(im * tarr[t] - in_ * zarr[z])
                for s in range(len(sarr)):

                    if(lvol > 0):
                        fac0 = sbar[s] ** np.ones_like(im)
                        fac1 =  0.5 ** np.ones_like(im)
                    else:
                        fac0 = sbar[s] ** (im + 1.0)
                        fac1 =  (im + 1.0) / 2.0 * sbar[s] ** (im)
                        fac0[im==0] = sbar[s]
                        fac1[im==0] = 0.5
                    
                    dR1 = Rac + fac0 * (Rbc - Rac)
                    Rarr0 = np.sum(dR1 * cos, axis=0)
                    Rarr = np.empty((3))
                    Rarr[0] = np.sum(fac1 * (Rbc - Rac) * cos, axis=0)
                    Rarr[1] = np.sum(-im * dR1 * sin, axis=0)
                    Rarr[2] = np.sum(in_ * dR1 * sin, axis=0)

                    for i in range(3):
                        for j in range(3):
                            metric[i,j,s,t,z] = Rarr[i] * Rarr[j]

                    metric[1,1,s,t,z] += Rarr0 ** 2.0
        
        metric[2, 2] += 1.0

    elif(geometry == 3): # toroidal

        for t in range(len(tarr)):
            for z in range(len(zarr)):
                cos = np.cos(im * tarr[t] - in_ * zarr[z])
                sin = np.sin(im * tarr[t] - in_ * zarr[z])
                for s in range(len(sarr)):
                    
                    if(lvol > 0):
                        fac0 = sbar[s] ** np.ones_like(im)
                        fac1 =  0.5 ** np.ones_like(im)
                    else:
                        fac0 = sbar[s] ** im
                        fac1 =  (im / 2.0) * sbar[s] ** (im - 1.0)
                        fac0[im==0] = sbar[s]**2
                        fac1[im==0] = sbar[s]

                    dR1 = Rac + fac0 * (Rbc - Rac)
                    Rarr = np.empty((3))
                    Rarr0 = np.sum(dR1 * cos, axis=0)
                    Rarr[0] = np.sum(fac1 * (Rbc - Rac) * cos, axis=0)
                    Rarr[1] = np.sum(-im * dR1 * sin, axis=0)
                    Rarr[2] = np.sum(in_ * dR1 * sin, axis=0)
                    
                    dZ1 = Zas + fac0 * (Zbs - Zas)
                    Zarr0 = np.sum(dZ1 * sin, axis=0)
                    Zarr = np.empty((3))
                    Zarr[0] = np.sum(fac1 * (Zbs - Zas) * sin, axis=0)
                    Zarr[1] = np.sum(im * dZ1 * cos, axis=0)
                    Zarr[2] = np.sum(-in_ * dZ1 * cos, axis=0)

                    for i in range(3):
                        for j in range(3):
                            metric[i,j,s,t,z] = Rarr[i] * Rarr[j] + Zarr[i] * Zarr[j]

                    metric[2, 2, s, t, z] += Rarr0 ** 2.0

    return metric


@run_decorator
def get_metric_helper_v2(geometry, rpol, rtor, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol):
    
    metric = np.empty((3, 3, len(sarr), len(tarr), len(zarr)))

    Rarr, Zarr = get_coord_transform_helper(geometry, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol)

    for i in range(3):
        for j in range(3):
            metric[i,j] = Rarr[i+1] * Rarr[j+1]

    if(geometry == 1): # slab/cartesian
        metric[1, 1] += rpol ** 2
        metric[2, 2] += rtor ** 2
    
    elif(geometry == 2): # cylindrical
        metric[1, 1] += Rarr[0] ** 2
        metric[2, 2] += 1.0 

    elif(geometry == 3): # toroidal
        for i in range(3):
                for j in range(3):
                    metric[i,j] += Zarr[i+1] * Zarr[j+1]
        metric[2, 2] += Rarr[0] ** 2

    return metric


def get_coord_transform(self, lvol, sarr, tarr, zarr):

    geometry = self.input.physics.Igeometry
    Rac, Rbc = self.output.Rbc[lvol : lvol + 2]
    Zas, Zbs = self.output.Zbs[lvol : lvol + 2]
    im = np.array(self.output.im, dtype=int)
    in_ = np.array(self.output.in_, dtype=int)

    Rarr, Zarr = get_coord_transform_helper(geometry, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol)
    return Rarr, Zarr


@run_decorator
def get_coord_transform_helper(geometry, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol):
    """Return coordinate transformations: R, dR/ds, dR/dtheta, dR/dzeta 
        (and for toroidal geometry: Z, dZ/ds, dZ/dtheta, dZ/dzeta)
    Returns:
        Rarr 
        Zarr
    """    
 
    sbar = (sarr + 1.0) / 2.0
    ns = len(sarr)
    nt = len(tarr)
    nz = len(zarr)

    Rarr = np.empty((4, ns, nt, nz))
    Zarr = np.zeros((4, ns, nt, nz))

    fac0 = np.empty((len(im), ns))
    fac1 = np.empty((len(im), ns))
    
    if(lvol > 0 or geometry == 1): # no coordinate singularity
        fac0[:] = sbar[None,:]
        fac1[:] = 0.5
    
    else: # with coordinate singularity
        if(geometry == 2): # cylindrical
            for s in range(ns):
                fac0[:,s] = sbar[s] ** (im + 1.0)
                fac1[:,s] =  (im + 1.0) * 0.5 * sbar[s] ** (im)
            fac0[im==0] = sbar
            fac1[im==0] = 0.5
        elif(geometry == 3): # toroidal
            for s in range(ns):      
                fac0[:,s] = sbar[s] ** im
                fac1[:,s] =  im * 0.5 * sbar[s] ** (im - 1.0)
            fac0[im==0] = sbar**2
            fac1[im==0] = sbar

    if(geometry == 3):
        for s in range(ns):
            dR1 = Rac + fac0[:,s] * (Rbc - Rac)
            dZ1 = Zas + fac0[:,s] * (Zbs - Zas)
            for t in range(nt):
                for z in range(nz):
                    cos = np.cos(im * tarr[t] - in_ * zarr[z])
                    sin = np.sin(im * tarr[t] - in_ * zarr[z])
                             
                    Rarr[0,s,t,z] = np.sum(dR1 * cos)
                    Rarr[1,s,t,z] = np.sum(fac1[:,s] * (Rbc - Rac) * cos)
                    Rarr[2,s,t,z] = np.sum(-im * dR1 * sin)
                    Rarr[3,s,t,z] = np.sum(in_ * dR1 * sin)

                    Zarr[0,s,t,z] = np.sum(dZ1 * sin)
                    Zarr[1,s,t,z] = np.sum(fac1[:,s] * (Zbs - Zas) * sin)
                    Zarr[2,s,t,z] = np.sum(im * dZ1 * cos)
                    Zarr[3,s,t,z] = np.sum(-in_ * dZ1 * cos)
    else:
        for s in range(ns):
            dR1 = Rac + fac0[:,s] * (Rbc - Rac)
            for t in range(nt):
                for z in range(nz):
                    cos = np.cos(im * tarr[t] - in_ * zarr[z])
                    sin = np.sin(im * tarr[t] - in_ * zarr[z])
                             
                    Rarr[0,s,t,z] = np.sum(dR1 * cos)
                    Rarr[1,s,t,z] = np.sum(fac1[:,s] * (Rbc - Rac) * cos)
                    Rarr[2,s,t,z] = np.sum(-im * dR1 * sin)
                    Rarr[3,s,t,z] = np.sum(in_ * dR1 * sin)
    
    return Rarr, Zarr