#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import h5py
import py_spec
from scipy import integrate, interpolate
import subprocess
import os
import sys
from f90nml import Namelist
import numpy.linalg as linalg
from coilpy import FourSurf

mu0 = 4*np.pi*1e-7

## set up initial profiles
psi0_mag = 3 * np.sqrt(3) / 4
bz0_mag = 2.0
psi_w = 0.01
Nvol = 41
rslab = 1.5 # rslab = rtor = rpol
x_resolution = 2**14
lfindzero = 0

fname_template = 'template.sp'
fname_input = 'test.sp'

def psi0(x):
    return psi0_mag / np.cosh(x)**2
def by0(x):
    return -2 * psi0_mag * np.sinh(x) / np.cosh(x)**3
def jz0(x):
    #toroidal current
    return 4 * psi0_mag * np.sinh(x)**2 / np.cosh(x)**4 - 2 * psi0_mag / np.cosh(x)**4
def bz0(x):
    # toroidal field
    return np.sqrt(bz0_mag**2 - by0(x)**2)
def jy0(x):
    return -(4 * by0(x)**2 * np.tanh(x) - 4 * by0(x)**2 / np.sinh(2*x)) / (2*bz0(x))


def calc_delprime(k):
    return 2*(5-k**2)*(3+k**2)/(k**2*np.sqrt(4+k**2))

def torflux(x, field):
    integral = integrate.cumtrapz(field(x), x)
    return np.concatenate([[0], integral])

def norm_torflux(x, field):
    return torflux(x, field) /  integrate.trapz(field(x), x)

def iota_slab(x, by, bz, Ly, Lz):
    return (2*np.pi*Lz * by(x)) / (2*np.pi*Ly * bz(x))

def ind_resintf(x, psi_w):
    fluxt_norm = norm_torflux(x, bz0)
    ind_l = np.argmin(np.abs(fluxt_norm - 0.5 * (1 - psi_w)))
    ind_r = np.argmin(np.abs(fluxt_norm - 0.5 * (1 + psi_w)))
    return ind_l, ind_r

def line_picker(line, mouseevent):
    """
    Find the points within a certain distance from the mouseclick in
    data coords and attach some extra attributes, pickx and picky
    which are the data points that were picked.
    --closest point only
    """
    if mouseevent.xdata is None:
        return False, dict()
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    maxd = 0.03
    d = np.sqrt(
        (xdata - mouseevent.xdata)**2 + (ydata - mouseevent.ydata)**2)
    d[d>maxd] = 1e2
    if np.any(d < maxd):
        ind = np.argmin(d)
        pickx = xdata[ind]
        picky = ydata[ind]
        props = dict(ind=ind, pickx=pickx, picky=picky, label=line.get_label())
        return True, props
    else:
        return False, dict()

def onpick2(event):
    print(f"\t{event.label}:  x {event.pickx:.5e}  y {event.picky:.5e}")

def join_arrays(a1, a2):
    return np.concatenate([a1, a2])

def print_indentd(a):
    return '\n\t' + str(a).replace('\n', '\n\t')

def gen_ifaces_funcspaced(x, ind_Lresinface, ind_Rresinface, func):
    values = func(x)
    Linfaces_val = np.linspace(values[0], values[ind_Lresinface], (Nvol+1)/2, endpoint=True)
    Rinfaces_val = np.linspace(values[-1], values[ind_Rresinface], (Nvol+1)/2, endpoint=True)[::-1]
    Linfaces_ind = np.argmin(np.abs(values[:len(x)//2,None]-Linfaces_val[None,:]), axis=0)
    Rinfaces_ind = np.argmin(np.abs(values[len(x)//2:,None]-Rinfaces_val[None,:]), axis=0) + (len(x)//2)
    infaces_ind = join_arrays(Linfaces_ind, Rinfaces_ind)
    infaces_val = join_arrays(Linfaces_val, Rinfaces_val)
    infaces_x = x[infaces_ind]
    return infaces_ind, infaces_x, infaces_val

def mu(x, by, bz, jy, jz):
    return (jy(x)*by(x) + jz(x)*bz(x)) / (by(x)**2 + bz(x)**2)

def get_hessian(fname_hdf5):

    def read_int(fid):
        return np.fromfile(fid, 'int32',1)

    with open('.'+fname_hdf5[:-3]+'.DF', 'rb') as fid:
        read_int(fid)
        igeom = read_int(fid)
        isym = read_int(fid)
        lfreeb = read_int(fid)
        nvol = read_int(fid)
        mpol = read_int(fid)
        ntor = read_int(fid)
        ngdof = read_int(fid)
        read_int(fid)
        read_int(fid)

        mode_number = int((nvol-1) * (mpol + 1))
        H = np.fromfile(fid, 'float64').reshape((mode_number, mode_number)).T

    return H

def perturb_eq(fname_hdf5, psi_w):
    # takes an .end file (with the modes at the end)

    w, v = linalg.eig(get_hessian(fname_hdf5))
    ind_min_eigmode = np.argmin(w)
    min_eigmode = v[:, ind_min_eigmode]

    # print('min eigenvalue', np.min(w))
    # print('eigenvalues',w)

    # "adjust" the kick
    kick =  np.real(min_eigmode[1::2])
    kick /= np.max(np.abs(kick))
    kick *= (0.78 * np.pi * psi_w) #* .9
    kick *= np.sign(kick[0])
    kick[-1] = 0

    # create new .sp file and add perturbation (from previous END FILE)
    subprocess.run(f"cp {fname_hdf5[:-3]}.end pert_{fname_hdf5[:-3]}", shell=True)

    inputnml = py_spec.SPECNamelist(f"pert_{fname_hdf5[:-3]}")

    inputnml['physicslist']['Mpol'] = 1
    inputnml['physicslist']['Lconstraint'] = 3
    inputnml['locallist']['LBeltrami'] = 4
    inputnml['numericlist']['Linitialize'] = 0
    inputnml['globallist']['Lfindzero'] = lfindzero
    inputnml['diagnosticslist']['LHmatrix'] = False
    for i in range(len(kick)):
        inputnml.set_interface_guess(kick[i], 1, 0, i, 'Rbc')

    inputnml.write_simple(f"pert_{fname_hdf5[:-3]}")

# this works (was tested)
def get_spec_jacobian(data, lvol, sarr, theta, zeta):

    rtor = data.input.physics.rtor
    rpol = data.input.physics.rpol
    G = data.input.physics.Igeometry
    Rarr, Zarr = get_spec_R_derivatives(data, lvol, sarr, theta, zeta, 'R')

    if(G == 1):
        return Rarr[1] * rtor * rpol # fixed here
    elif(G == 2):
        return Rarr[0] * Rarr[1]
    elif(G == 3):
        return Rarr[0] * (Rarr[2]*Zarr[1] - Rarr[1]*Zarr[2])
    else:
        raise ValueError("Error: unsupported dimension")

# this works (was tested)
# the vol index is -1 compared to the matlab one
def get_spec_R_derivatives(data, vol, sarr, theta, zeta, RorZ):

    mn = data.output.mn
    Rmn = data.output.Rbc[vol,:]
    Rmn_p = data.output.Rbc[vol+1,:]
    Zmn = data.output.Zbs[vol,:]
    Zmn_p = data.output.Zbs[vol+1,:]
    im = data.output.im
    iN = data.output.in_
    ns = len(sarr)

    factor = get_spec_regularization_factor(data, vol, sarr, 'G')

    cosa = np.cos(im*theta-iN*zeta)
    sina = np.sin(im*theta-iN*zeta)

    R = np.zeros((4, ns))
    R[0] = np.einsum('m,ms->s', cosa * (Rmn + (Rmn_p - Rmn)), factor[:,0])
    R[1] = np.einsum('m,ms->s', cosa * (Rmn_p - Rmn), factor[:,1])
    R[2] = np.einsum('m,ms->s',-sina * (Rmn + (Rmn_p - Rmn)) * im, factor[:,0])
    R[3] = np.einsum('m,ms->s', sina * (Rmn_p - Rmn) * iN, factor[:,0])

    Z = np.zeros((4, ns))
    Z[0] = np.einsum('m,ms->s', sina * (Zmn + (Zmn_p - Zmn)), factor[:,0])
    Z[1] = np.einsum('m,ms->s', sina * (Zmn_p - Zmn), factor[:,1])
    Z[2] = np.einsum('m,ms->s', cosa * (Zmn + (Zmn_p - Zmn)) * im, factor[:,0])
    Z[3] = np.einsum('m,ms->s',-cosa * (Zmn_p - Zmn) * iN, factor[:,0])

    return R, Z


# this works (was tested)
def get_spec_polynomial_basis(data, lvol, sarr):

    ns = len(sarr)
    G = data.input.physics.Igeometry
    Mpol = data.input.physics.Mpol
    Lrad = data.input.physics.Lrad[lvol]

    Lzernike = False
    if(lvol == 0 and G != 1):
        if(data.version < 3.0):
            Lzernike = False
        else:
            Lzernike = True

    T = np.ones((Lrad+1,2,ns))
    T[0,1] = 0
    T[1,0] = sarr

    if(Lzernike):
        #NASTY STUFF
        zernike = np.zeros((Lrad+1,Mpol+1,2,ns))
        rm = np.ones(ns)
        rm1 = np.zeros(ns)

        sbar = (sarr+1)/2

        for m in range(-1,Mpol):
            if(Lrad >= m):
                zernike[m+1,m+1,0,:] = rm
                zernike[m+1,m+1,1,:] = (m+1) * rm1

            if(Lrad >= m+3):
                zernike[m+3,m+1,0,:] = (m+3) * rm * sbar**2 - (m+2) * rm
                zernike[m+3,m+1,1,:] = (m+3)**2 * rm * sbar - (m+2)*(m+1)*rm1

            for n in range(m+4-1,Lrad,2):
                factor1 = (n+1) / ((n+1)**2 - (m+1)**2)
                factor2 = 4 * (n)
                factor3 = (n-1+m+1)**2/(n-1) + (n-m)**2/(n+1)
                factor4 = ((n-1)**2-(m+1)**2)/(n-1)

                zernike[n+1,m+1,0,:] = factor1 * ((factor2*sbar**2-factor3)*zernike[n-1,m+1,0,:]) - factor4 * zernike[n-3,m+1,1,:]
                zernike[n+1,m+1,1,:] = factor1 * (2*factor2*sbar * zernike[n-1,m+1,1,:] + (factor2*sbar**2 - factor3)*zernike[n-1,m+1,1,:] - factor4 * zernike[n-3,m+1,1,:])

            rm1 = rm
            rm *= sbar

        for n in range(1,Lrad,2):
            zernike[n+1,0,0,:] -= (-1)**((n+1)/2)

        if(Mpol >= 1):
            for n in range(3,Lrad,2):
                zernike[n+1,1,0,:] -= (-1)**((n-1)/2) * (n+1)/2 * sbar
                zernike[n+1,1,1,:] -= (-1)**((n-1)/2) * (n+1)/2

        for m in range(-1,Mpol):
            for n in range(m,Lrad,2):
                zernike[n+1,m+1,:,:] /= (n)

        T[:,0,ns] = np.reshape(zernike[:,:,0,:],(Mpol+1,ns))
        T[:,1,ns] = np.reshape(zernike[:,:,1,:],(Mpol+1,ns))

    else:
        for l in range(2, Lrad+1):
            T[l,0] = 2*sarr*T[l-1,0] - T[l-2,0]
            T[l,1] = 2*T[l-1,0] + 2*sarr*T[l-1,1] - T[l-2,1]

        for l in range(Lrad):
            T[l+1,0] -= (-1)**(l+1)

        for l in range(-1,Lrad):
            T[l+1,0] /= (l+2)
            T[l+1,1] /= (l+2)

    return T

# this works (was tested)
def get_contra2cov(data, lvol, vec_contrav, sarr, theta, phi, norm):

    ns = len(sarr)
    g = get_spec_metric(data, lvol, sarr, theta, phi)

    vec_cov = np.einsum('xas,as->xs', g, vec_contrav)
    return vec_cov

# this works (was tested)
def get_spec_regularization_factor(data, lvol, sarr, ForG):

    G = data.input.physics.Igeometry
    mn = data.output.mn
    im = data.output.im
    iN = data.output.in_
    ns = len(sarr)

    sbar = np.array((1+sarr)/2)
    reumn = im/2
    Mregular  = data.input.numerics.Mregular
    fac = np.zeros((mn, 2, ns))

    if(Mregular > 1):
        ind = np.argmax(regumn>Mregular)
        regumn[ind] = Mregular / 2

    if(ForG == 'G'):
        if(G == 1):
            for j in range(mn):
                fac[j,0] = sbar
                fac[j,1] = 0.5*np.ones(ns)
        elif(G == 2):
            for j in range(mn):
                if(lvol == 0):
                    if(im[j] == 0):
                        fac[j,0] = sbar
                        fac[j,1] = 0.5*np.ones(ns)
                    else:
                        fac[j,0] = sbar**(im[j]+1)
                        fac[j,1] = 0.5 * (im[j]+1) * fac[j,0] / sbar
                else:
                    fac[j,0] = sbar
                    fac[j,1] = 0.5*np.ones(ns)
        elif(G == 3):
            for j in range(mn):
                if(lvol == 1):
                    if(im[j] == 0):
                        fac[j,0] = sbar**2
                        fac[j,1] = sbar
                    else:
                        fac[j,0] = sbar**im[j]
                        fac[j,1] = 0.5*im[j]*fac[j,0]/sbar
                else:
                    fac[j,0] = sbar
                    fac[j,1] = 0.5*np.ones(ns)
    elif(ForG == 'F'):
        fac[:,0,:] = np.ones((mn, ns))

    return fac


# this works (was tested)
def get_spec_metric(data, lvol, sarr, theta, zeta):

    ns = len(sarr)
    G = data.input.physics.Igeometry
    rtor = data.input.physics.rtor
    rpol = data.input.physics.rpol

    gmat = np.zeros((3,3,ns))
    Rarr, Zarr = get_spec_R_derivatives(data, lvol, sarr, theta, zeta, 'R')

    if(G == 1):
        gmat[0,0] = Rarr[1]**2
        gmat[1,1] = Rarr[2]**2 + rpol**2
        gmat[2,2] = Rarr[3]**2 + rtor**2
        gmat[0,1] = Rarr[1]*Rarr[2]
        gmat[0,2] = Rarr[1]*Rarr[3]
        gmat[1,2] = Rarr[2]*Rarr[3]
    elif(G == 2):
        gmat[0,0] = Rarr[1]**2
        gmat[1,1] = Rarr[2]**2 + Rarr[0]**2
        gmat[2,2] = Rarr[3]**2 + rtor**2
        gmat[0,1] = Rarr[1]*Rarr[2]
        gmat[0,2] = Rarr[1]*Rarr[3]
        gmat[1,2] = Rarr[2]*Rarr[3]
    elif(G == 3):
        gmat[0,0] = Rarr[1]**2 + Zarr[1]**2
        gmat[1,1] = Rarr[2]**2 + Zarr[2]**2
        gmat[2,2] = Rarr[0]**2 + Rarr[3]**2 + Zarr[3]**2
        gmat[0,1] = Rarr[1]*Rarr[2] + Zarr[1]*Zarr[2]
        gmat[0,2] = Rarr[1]*Rarr[3] + Zarr[1]*Zarr[3]
        gmat[1,2] = Rarr[2]*Rarr[3] + Zarr[2]*Zarr[3]
    else:
        raise ValueError("G (geometry setting) has to be 1,2, or 3")

    gmat[1,0] = gmat[0,1]
    gmat[2,1] = gmat[1,2]
    gmat[2,0] = gmat[0,2]

    return gmat

# this works (was tested)
def get_spec_radius(data, theta, zeta, vol):
    # Return the radial position of a KAM surface for a given theta, zeta and Nvol

    G = data.input.physics.Igeometry
    mn = data.output.mn
    Rmn = data.output.Rbc
    im = data.output.im
    iN = data.output.in_

    if(G == 1):
        r_out = 0
        for k in range(mn):
            r_out += Rmn[vol+1,k] * np.cos(im[k]*theta - iN[k]*zeta)
        z_out = 0
    elif(G == 2):
        r_out = np.dot(Rmn[vol+1,:], np.cos(im*theta - iN*zeta))
        z_out = 0
    elif(G == 3):
        r_out = np.dot(Rmn[vol+1,:], np.cos(im*theta - iN*zeta))
        z_out = np.dot(Zmn[vol+1,:], np.sin(im*theta - iN*zeta))

    return r_out, z_out

def plot_kamsurf(fname):

    myspec = py_spec.SPECout(fname)
    surfs = myspec.plot_kam_surface(marker='o',s=2.)

def get_vecpot(data, lvol, sarr, tarr, zarr):

    vecpot = data.vector_potential
    im = data.output.im
    iN = data.output.in_
    mn = data.output.mn
    ns = len(sarr)
    nt = len(tarr)
    nz = len(zarr)

    Lrad = data.input.physics.Lrad[lvol]
    Ate = np.array(vecpot.Ate[lvol]).T
    Aze = np.array(vecpot.Aze[lvol]).T
    Ato = np.array(vecpot.Ato[lvol]).T
    Azo = np.array(vecpot.Azo[lvol]).T

    At = np.zeros((ns,nt,nz))
    Az = np.zeros((ns,nt,nz))
    dAt = np.zeros((ns,nt,nz))
    dAz = np.zeros((ns,nt,nz))

    #chebyshev polynomials
    T = np.zeros((Lrad + 1, 2, ns))
    T[0,0] = np.ones(ns)
    T[0,1] = np.zeros(ns)
    T[1,0] = sarr
    T[1,1] = np.ones(ns)

    for l in range(2, Lrad+1):
        T[l,0] = 2 * sarr * T[l-1,0] - T[l-2,0]
        T[l,1] = 2 * T[l-1,0] + 2 * sarr * T[l-1,1] - T[l-2,1]

    # reg. factors
    fac = np.ones((ns, 2, ns))
    fac[:,1] = 0


    a = im[:,None,None] * tarr[None,:,None] - iN[:,None,None] * zarr[None,None,:]
    term_t =  Ate[:,:,None,None]*np.cos(a[None,...]) + Ato[:,:,None,None]*np.sin(a[None,...])
    term_z = Aze[:,:,None,None]*np.cos(a[None,...]) + Azo[:,:,None,None]*np.sin(a[None,...])
    At_dAt = np.einsum('js,li,ljtz->istz', fac[:,0], T, term_t)
    Az_dAz = np.einsum('js,li,ljtz->istz', fac[:,0], T, term_z)
    return At_dAt[0], Az_dAz[0], At_dAt[1], Az_dAz[1]

    for l in range(Lrad):
        for j in range(mn):
            for it in range(nt):
                for iz in range(nz):
                    cosa = np.cos(im[j]*tarr[it]-iN[j]*zarr[iz])
                    sina = np.sin(im[j]*tarr[it]-iN[j]*zarr[iz])
                    At[:,it,iz] += fac[j,0] * T[l,0] * (Ate[l,j]*cosa + Ato[l,j]*sina)
                    Az[:,it,iz] += fac[j,0] * T[l,0] * (Aze[l,j]*cosa + Azo[l,j]*sina)
                    dAt[:,it,iz] += fac[j,0] * T[l,1] * (Ate[l,j]*cosa + Ato[l,j]*sina)
                    dAz[:,it,iz] += fac[j,0] * T[l,1] * (Aze[l,j]*cosa + Azo[l,j]*sina)

    return At, Az, dAt, dAz

def get_rtarr(data, lvol, sarr, tarr, zarr0):

    Rac = np.array(data.output.Rbc[lvol,:])
    Rbc = np.array(data.output.Rbc[lvol+1,:])

    ns = len(sarr)
    nt = len(tarr)
    sbar = (np.array(sarr)+1)/2
    mn = data.output.mn
    im = data.output.im
    iN = data.output.in_

    Rarr = np.zeros((ns, nt))
    Tarr = np.zeros((ns, nt))
    dRarr = np.zeros((ns, nt))

    fac = np.ones((ns, 2, ns))
    fac[:,1] = 0

    if(data.input.physics.Igeometry == 1):
        for j in range(mn):
            fac[j,0] = sbar
            fac[j,1] = 0.5
    elif(data.input.physics.Igeometry == 2):
        for j in range(mn):
            if(lvol == 0):
                if(im(j) == 0):
                    fac[j,0] = np.sqrt(sbarr)
                    fac[j,1] = 0.25 / np.sqrt(sbarr)
                else:
                    fac[j,0] = sbar * im(j) / 2
                    fac[j,1] = im(j) * 0.25 * sbar**(im(j)/2-1)
        else:
            fac[j,0] = sbar
            fac[j,1] = 0.5

    for j in range(mn):
        for it in range(nt):
            cosa = np.cos(im[j]*tarr[it] - iN[j]*zarr0)
            sina = np.sin(im[j]*tarr[it] - iN[j]*zarr0)
            Rarr[:,it] += cosa * (Rac[j] + fac[j,0] * (Rbc[j]-Rac[j]))
            dRarr[:,it] += fac[j,1] * cosa * (Rbc[j] - Rac[j])
            Tarr[:,it] = tarr[it]

    return Rarr, Tarr, dRarr

def get_islandwidth(fname, ns, nt):

    data = py_spec.SPECout(fname)

    fdata = data.vector_potential
    lvol = 1 + data.input.physics.Nvol // 2
    sarr = np.linspace(-1, 1, ns)
    tarr = np.linspace(0, 2*np.pi, nt)

    z0   = np.array([0])
    r0   = data.output.Rbc[-1,0]/2

    # Get flux function
    At, Az, dAt, dAz = get_vecpot(data, lvol, sarr, tarr, z0)
    asep = Az[ns//2,0]

    # Find radial position of the separatrix at theta=pi
    I = np.argmin(np.abs(asep-Az[ns//2:,nt//2]))

    Rarr, Tarr, dRarr = get_rtarr(data, lvol, sarr, tarr, z0)

    rw = Rarr[ns//2+I-1,nt//2]

    Wisland = 2 * (rw - r0)

    print(f'Island width {Wisland}')
    return Wisland

def get_full_field(data, r, theta, zeta, nr, const_factor=None):

    nvol = data.output.Mvol
    G = data.input.physics.Igeometry
    mn = data.output.mn
    Rmn = data.output.Rbc
    im = data.output.im
    iN = data.output.in_
    rtor = data.input.physics.rtor
    rpol = data.input.physics.rpol

    iimin = 0
    iimax = 1

    r0, z0 = get_spec_radius(data, theta, zeta, -1)
    B = np.zeros((3, len(r)))

    for i in range(nvol):

        ri, zi = get_spec_radius(data, theta, zeta, i-1)
        rmin = np.sqrt((ri-r0)**2 + zi**2)

        ri, zi = get_spec_radius(data, theta, zeta, i)
        rmax = np.sqrt((ri-r0)**2 + zi**2)
        r_vol = np.linspace(rmin, rmax, nr+1)#[1:]

        sarr = 2 * (r_vol - rmin) / (rmax - rmin) - 1
        if(i == 0 and G != 1):
            sarr = 2 * ((r_vol - rmin) / (rmax - rmin))**2 - 1

        B_contrav = get_spec_magfield(data, i, sarr, theta, zeta)

        iimax = iimax + len(r_vol)

        B_cov = get_contra2cov(data, i, B_contrav, sarr, theta, zeta, 1)

        B_cart = np.zeros_like(B_cov)
        B_cart[0] = B_cov[0]
        B_cart[1] = B_cov[1] / rpol
        B_cart[2] = B_cov[2] / rtor


        if(const_factor is not None):
            B_cart *= const_factor[i] # multiply by mu to get current, or else

        iimin = iimin + len(r_vol)
        ind = np.logical_and(r <= rmax, r >= rmin)
        r_int = r[ind]

        for comp in range(3):
            f = interpolate.interp1d(r_vol, B_cart[comp,:], kind='cubic')
            B[comp, ind] = f(r_int)

    return B

# this works (was tested)
def get_spec_magfield(data, lvol, sarr, theta, zeta):

    jac = get_spec_jacobian(data, lvol, sarr, theta, zeta)

    vecpot = data.vector_potential
    im = data.output.im
    iN = data.output.in_
    mn = data.output.mn
    ns = len(sarr)
    G = data.input.physics.Igeometry

    Lrad = data.input.physics.Lrad[lvol]
    Ate = np.array(vecpot.Ate[lvol]).T
    Aze = np.array(vecpot.Aze[lvol]).T
    Ato = np.array(vecpot.Ato[lvol]).T
    Azo = np.array(vecpot.Azo[lvol]).T

    Bs = np.zeros((ns))
    Bt = np.zeros((ns))
    Bz = np.zeros((ns))

    T = get_spec_polynomial_basis(data, lvol, sarr)
    Lsingularity = False
    if(lvol == 0 and G != 1):
        Lsingularity = True

    for l in range(Lrad+1):
        for j in range(mn):
            if(Lsingularity):
                basis = np.swapaxes(T[l,0,im[j]+1,:],0,1)
                dbasis = np.swapaxes(T[l,1,im[j]+1,:],0,1)
            else:
                basis = T[l,0]
                dbasis = T[l,1]

            cosa = np.cos(im[j]*theta - iN[j]*zeta)
            sina = np.sin(im[j]*theta - iN[j]*zeta)
            Bs[:] += basis * ((im[j]*Azo[l,j] + iN[j]*Ato[l,j]) * cosa - (im[j]*Aze[l,j] + iN[j]*Ate[l,j]) * sina)
            Bt[:] -= dbasis * (Aze[l,j]*cosa + Azo[l,j]*sina)
            Bz[:] += dbasis * (Ate[l,j]*cosa + Ato[l,j]*sina) # fixed it here

    Bcontrav = np.array([Bs, Bt, Bz]) / jac
    return Bcontrav

def run_spec(fname, show_output=False):
    print(f"\n{('-'*80)}\nRunning SPEC with {fname}\n")
    if(show_output):
        subprocess.run(f'mpirun -n 4 xspec {fname}', shell=True)
    else:
        subprocess.run(f'mpirun -n 4 xspec {fname}', shell=True, stdout=subprocess.DEVNULL)
        data = py_spec.SPECout(fname+".h5")
        print(f"\n{('-'*80)}\nSPEC completed, force error {data.output.ForceErr:.4e}\n\n")

def check_isect():
    # TODO checks whether interfaces overlap (which means parameter combination is invalid)
    pass

def plot_infaces(x, ind_Lresinface, ind_Rresinface, infaces_x):

    # plot interfaces
    fig, ax = plt.subplots(1, 1)
    ln1 = ax.hlines(x[ind_Lresinface], -np.pi, np.pi, colors='r', linestyles='dashed', zorder=10, label='Resonant-volume interfaces')
    ax.hlines(x[ind_Rresinface], -np.pi, np.pi, colors='r', linestyles='dashed', zorder=10)
    ln2 = ax.hlines(infaces_x, -np.pi, np.pi, colors='k', linestyles='solid', label='Outer interfaces')
    ax.set_ylim([-np.pi*1.04,np.pi*1.04])
    ax.set_xlim([-np.pi,np.pi])
    ax.fill_between([-np.pi, np.pi], x[ind_Lresinface], x[ind_Rresinface], color='red', alpha=0.2)

    axb = ax.twiny()
    ln3 = axb.plot(psi0(x), x,'.-',label='psi0(x)')
    axb.set_xlim([-2,2])
    ax.set_xlabel('y')
    ax.set_ylabel('x')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = axb.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, framealpha=1.0)

    return fig

def check_profile_Bfield(fname):

    fig, ax = plt.subplots(1, 1)
    plt.title('Plot of magnetic field B')
    fig.canvas.set_window_title('Field B')

    data = py_spec.SPECout(fname)

    theta = np.pi
    phi = 0
    num_radpts = 200
    num_radpts_pervol = 10
    r = np.linspace(0, 2*np.pi, num_radpts, endpoint=True)

    B = get_full_field(data, r, theta, phi, num_radpts_pervol)

    # ax.plot(r, B[0], '.-', picker=line_picker, label='spec B_psi')
    ax.plot(r, B[1], '.-', picker=line_picker, label='spec B_theta')
    ax.plot(r, B[2], '.-', picker=line_picker, label='spec B_ phi')

    ax.plot(r, by0(r - np.pi), '--', picker=line_picker, label='B_y0')
    ax.plot(r, bz0(r-np.pi), '--', picker=line_picker, label='B_z0')

    ax.legend()
    ax.set_xlabel("Radial coordinate r / x [m]")
    ax.set_ylabel("Field strength [T]")

    fig.canvas.mpl_connect('pick_event', onpick2)
    fig.tight_layout()

def check_profile_curr(fname):

    fig, ax = plt.subplots(1, 1)
    plt.title('Plot of current J')
    fig.canvas.set_window_title('Current J')

    data = py_spec.SPECout(fname)

    theta = np.pi
    phi = 0
    num_radpts = 400
    num_radpts_pervol = 12
    r = np.linspace(0, 2*np.pi, num_radpts, endpoint=True)

    mu_spec = data.output.mu

    J = get_full_field(data, r, theta, phi, num_radpts_pervol, mu_spec)

    # ax.plot(r, J[0], '.-', picker=line_picker, label='SPEC J_psi')
    ax.plot(r, J[1], '.-', picker=line_picker, label='SPEC J_theta')
    ax.plot(r, J[2], '.-', picker=line_picker, label='SPEC J_phi')

    ax.plot(r, jy0(r - np.pi), '--', picker=line_picker, label='analytic J_y0')
    ax.plot(r, jz0(r - np.pi), '--', picker=line_picker, label='analytic J_z0')

    ax.legend()
    ax.set_xlabel("Radial coordinate r / x [m]")
    ax.set_ylabel("Current [$\mu_0$]")

    fig.canvas.mpl_connect('pick_event', onpick2)
    fig.tight_layout()

def plot_staired(ax, field, x, label):

    ax.hlines(field, x, x[1:], lw=2.0, label=label)
    l = ax.plot(0.5*(x[1:]+x[:-1]), field, alpha=0, picker=line_picker)
    l[0].set(label=label)

def check_profile_mu(fname):
    """
    Plots mu in each volume (piecewise constant)
    """
    fig, ax = plt.subplots(1, 1)
    plt.title('Plot of $\mu$')
    fig.canvas.set_window_title('Mu')

    data = py_spec.SPECout(fname)

    mu_spec = data.output.mu
    x_spec = data.output.Rbc[:,0]
    x_spec_vol = (x_spec[:-1]+x_spec[1:])*.5
    plot_staired(ax, mu_spec, x_spec, "SPEC mu")

    x_array =  np.linspace(np.min(x_spec), np.max(x_spec), 200) - np.pi
    mu_analytic = mu(x_array, by0, bz0, jy0, jz0)

    ax.plot(x_array + np.pi, mu_analytic, '--', lw=1.0, label="analytic mu", picker=line_picker)

    fig.canvas.mpl_connect('pick_event', onpick2)
    ax.legend()

    fig.tight_layout()

def check_profile_flux(fname):
    """
    Plots flux encloses by each interface (pflux, tflux); compares to analytic expressions
    """
    fig, ax = plt.subplots(1, 1)
    plt.title('Plot of poloidal and toroidal fluxes')
    fig.canvas.set_window_title('Flux')

    data = py_spec.SPECout(fname)

    phiedge_spec = data.input.physics.phiedge
    x_spec = data.output.Rbc[1:,0]
    pflux = data.output.pflux * phiedge_spec
    tflux = data.output.tflux

    ax.plot(x_spec, pflux, 'd', label='SPEC pflux', picker=line_picker)
    ax.plot(x_spec, tflux, 'd', label='SPEC tflux', picker=line_picker)

    x_array =  np.linspace(np.min(x_spec), np.max(x_spec), 200) - np.pi
    tflux_analytic = norm_torflux(np.concatenate([[-np.pi],x_array]), bz0)[1:]
    pflux_analytic = psi0(x_array) * (2 * np.pi * rslab)

    ax.plot(x_array + np.pi, pflux_analytic, '--', lw=2.0, label="analytic pflux", picker=line_picker)
    ax.plot(x_array + np.pi, tflux_analytic, '--', lw=2.0, label="analytic tflux", picker=line_picker)

    fig.canvas.mpl_connect('pick_event', onpick2)
    ax.legend()
    fig.tight_layout()

def check_profile_iota(fname):
    """
    Plots iota obtained from SPEC fieldline tracing; compares to analytic form
    """
    fig, ax = plt.subplots(1, 1)
    plt.title('Plot of iota')
    fig.canvas.set_window_title('Iota')

    data = py_spec.SPECout(fname)

    theta = np.pi
    phi = 0
    num_radpts = 200
    num_radpts_pervol = 10
    r = np.linspace(0, 2*np.pi, num_radpts, endpoint=True)

    B = get_full_field(data, r, theta, phi, num_radpts_pervol)

    iota_spec = B[1] / B[2] # in simple case here

    iota_analytic = iota_slab(r - np.pi, by0, bz0, rslab, rslab)
    ax.plot(r, iota_analytic, '--', lw=2.0, label="analytic iota", picker=line_picker)

    ax.plot(r, iota_spec, '.', label='SPEC iota', picker=line_picker)

    fiota = data.transform.fiota
    x_spec = data.output.Rbc[:,0]

    ## same as iota_spec
    # iota_in = fiota[1, fiota[0] < (-1+1e-8)]
    # iota_out = fiota[1, fiota[0] > (1-1e-8)]
    # ax.plot(x_spec[:-1], iota_in, '-.', label='SPEC iota in (tracing)', picker=line_picker)
    # ax.plot(x_spec[1:], iota_out, '-.', label='SPEC iota out (tracing)', picker=line_picker)

    fig.canvas.mpl_connect('pick_event', onpick2)
    ax.legend()
    fig.tight_layout()

if __name__ == "__main__":

    subprocess.run(f"rm test.sp.end test.sp.h5", shell=True)

    x_array = np.linspace(-np.pi, np.pi, x_resolution)

    ind_Lresinface, ind_Rresinface = ind_resintf(x_array, psi_w)
    infaces_ind, infaces_x, infaces_psi = gen_ifaces_funcspaced(x_array, ind_Lresinface, ind_Rresinface, lambda x: x) #uniform spacing
    # infaces_ind, infaces_x, infaces_psi = gen_ifaces_funcspaced(x_array, ind_Lresinface, ind_Rresinface, psi0)

    pre = np.zeros(Nvol)
    lrad = 8 * np.ones(Nvol)
    pcare_ntrj = np.ones(Nvol)
    resvol_ind = (Nvol+1)//2
    pcare_ntrj[resvol_ind] = 10
    phi_edge = 2 * np.pi * rslab * integrate.trapz(bz0(x_array), x_array)

    tflux = norm_torflux(x_array, bz0)[infaces_ind]
    pflux = psi0(infaces_x) * (2 * np.pi * rslab)

    mu_array = mu(x_array, by0, bz0, jy0, jz0)
    mu_vol = np.zeros(Nvol)
    for i in range(Nvol):
        mu_vol[i] = np.mean(mu_array[infaces_ind[i]:infaces_ind[i+1]])

    iota_vol = iota_slab(infaces_x, by0, bz0, rslab, rslab)

    curr_array = torflux(x_array, jz0) * (2*np.pi)
    curr_vol = curr_array[infaces_ind]

    ## create and edit the namelist file
    subprocess.run(f"cp {fname_template} {fname_input}", shell=True)
    inputnml = py_spec.SPECNamelist(fname_input)
    inputnml['physicslist']['pressure'] = pre
    inputnml['globallist']['Lfindzero'] = 0 # 0 for only field calculation, 2 for actually moving interfaces
    inputnml['physicslist']['Lrad'] = lrad
    inputnml['physicslist']['phiedge'] = phi_edge
    inputnml['physicslist']['Nvol'] = Nvol
    inputnml['diagnosticslist']['nPtrj'] = pcare_ntrj

    # inputnml['physicslist']['curtor'] = 2 * np.pi  * integrate.trapz(jz0(x_array), x_array)
    # inputnml['physicslist']['curpol'] = 0

    ## init geometry
    inputnml['numericlist']['Linitialize'] = 0 # 0 for custom Rbc, 1 for automatic interpolation
    inputnml['physicslist']['rpol'] = rslab
    inputnml['physicslist']['rtor'] = rslab
    inputnml.interface_guess[(0,0)]['Rbc'] = infaces_x[1:] + np.pi
    inputnml.interface_guess[(1,0)]['Rbc'] = np.zeros_like(infaces_x[1:])

    ## init profiles
    inputnml['physicslist']['Lconstraint'] = 0 # should be 0 for constant psi_p, psi_t, mu

    # if lconstraint is 0, the field has error in first volume
    if(inputnml['physicslist']['Lconstraint'] == 0):
        inputnml['physicslist']['tflux'] = tflux[1:]
        inputnml['physicslist']['pflux'] = pflux[1:] / phi_edge
        inputnml['physicslist']['mu'] = mu_vol

    # If lconstraint is 1 and iotas are set, then the field is good
    if(inputnml['physicslist']['Lconstraint'] == 1):
        inputnml['physicslist']['iota'] = iota_vol
        inputnml['physicslist']['oita'] = iota_vol
        inputnml['physicslist']['tflux'] = tflux[1:]
        inputnml['physicslist']['mu'] = mu_vol

    # constrain the currents
    # little const offset in Btheta for this case, and thus pflux
    if(inputnml['physicslist']['Lconstraint'] == 3):
        inputnml['physicslist']['isurf'] = np.zeros(Nvol)
        inputnml['physicslist']['ivolume'] = curr_vol[1:]
        inputnml['physicslist']['tflux'] = tflux[1:]

    inputnml.write_simple(fname_input)

    run_spec('test.sp', show_output=False)

    ## create and run a perturbed equilibrium
    # perturb_eq('test.sp.h5', psi_w)
    # run_spec('pert_test.sp')

    plotout = plot_kamsurf(fname='test.sp.h5')
    # get_islandwidth(fname='pert_test.sp.h5', ns=60, nt=100)
    # print('Del prime is ', calc_delprime(1/rslab))

    # the fluxes are good (both lfinzero=0 and 2, uneven/even spacing, many/little volumes, large/small psiw)
    check_profile_flux('test.sp.h5')

    # check_profile_curr('test.sp.h5')

    check_profile_mu('test.sp.h5')

    check_profile_Bfield('test.sp.h5')

    # check_profile_iota('test.sp.h5')


    ## weird magnetic field glitch when Lconstraint=1
    # check_profile_Bfield('test_lc0_new.sp.h5')

    plt.show()
