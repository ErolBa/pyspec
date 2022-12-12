## Written by Erol Balkovic
#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import h5py
from py_spec import SPECNamelist, SPECout
from scipy import integrate, interpolate, optimize
import subprocess
import sys
import numpy.linalg as linalg
import numba
import os
import warnings

class SPECslab():
    """
    Class for running the slab (G=1) case, including plotting, analyzing, and perturbing equilibrium
    """

    mu0 = 4*np.pi*1e-7

    def calc_delprime(k):
        return 2*(5-k**2)*(3+k**2)/(k**2*np.sqrt(4+k**2))

    def calc_rslab_for_delprime(target_delprime):
        target_k = optimize.fsolve(lambda x: SPECslab.calc_delprime(x)-target_delprime, 1.)
        target_rslab = 1/target_k[0]
        return target_rslab

    def torflux(x, field):
        integral = integrate.cumtrapz(field(x), x)
        return np.concatenate([[0], integral])

    def norm_torflux(x, field):
        return SPECslab.torflux(x, field) /  integrate.trapz(field(x), x)

    def iota_slab(x, by, bz, Ly, Lz):
        return (2*np.pi*Lz * by(x)) / (2*np.pi*Ly * bz(x))

    def infacesres_ind(x, psi_w, func_bz0):
        fluxt_norm = SPECslab.norm_torflux(x, func_bz0)
        ind_l = np.argmin(np.abs(fluxt_norm - 0.5 * (1 - psi_w)))
        ind_r = np.argmin(np.abs(fluxt_norm - 0.5 * (1 + psi_w)))
        return ind_l, ind_r

    def line_picker(line, mouseevent):
        """
        Find the points within a certain distance from the mouseclick in
        data coords and attach some extra attributes, pickx and picky
        which are the data points that were picked.
        --closest point only (from each line)
        """
        if mouseevent.xdata is None:
            return False, dict()
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        maxd = 0.02
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

    def gen_ifaces_funcspaced(x, ind_Lresinface, ind_Rresinface, func, Nvol):
        values = func(x)
        Linfaces_val = np.linspace(values[0], values[ind_Lresinface], (Nvol+1)/2, endpoint=True)
        Rinfaces_val = np.linspace(values[-1], values[ind_Rresinface], (Nvol+1)/2, endpoint=True)[::-1]
        Linfaces_ind = np.argmin(np.abs(values[:len(x)//2,None]-Linfaces_val[None,:]), axis=0)
        Rinfaces_ind = np.argmin(np.abs(values[len(x)//2:,None]-Rinfaces_val[None,:]), axis=0) + (len(x)//2)
        infaces_ind = SPECslab.join_arrays(Linfaces_ind, Rinfaces_ind)
        infaces_val = SPECslab.join_arrays(Linfaces_val, Rinfaces_val)
        infaces_x = x[infaces_ind]
        return infaces_ind, infaces_x, infaces_val

    def mu(x, by, bz, jy, jz):
        return (jy(x)*by(x) + jz(x)*bz(x)) / (by(x)**2 + bz(x)**2)

    def check_psi(x, psi):
        if(psi(x[0]) > 1e-16):
            raise ValueError("Psi at x=0 has to be 0 !!!\n")

    def get_hessian(fname):

        if(fname[-3:] == '.sp'):
            fname = fname+".DF"
        elif(fname[-3:] == '.h5'):
            fname = fname[:-3]+".DF"
        elif(fname[-4:] == '.end'):
            fname = fname[:-4]+".DF"
        else:
            raise ValueError("Invalid file given to get_hessian()")

        def read_int(fid):
            return np.fromfile(fid, 'int32',1)

        with open('.'+fname, 'rb') as fid:
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

    def get_eigenstuff(fname):

        h = SPECslab.get_hessian(fname)
        w, v = linalg.eig(h)

        ## there are two ways of determining the smallest eigenvalue (choose one)
        # first, looking at the smallest real part eigenvalue (numpy defualt btw)
        indmin = np.argmin(w)
        # second, looking at the largest magnitude of the complex number (default in MATLAB)
        indmin2 = np.argmin(np.abs(w))

        minvec = v[:, indmin]

        # minvec_m1 = minvec[1::2]
        # print("minvec",minvec_m1[np.abs(minvec_m1.imag)>1e-15])
        # print(indmin,indmin2)
        # print("indmin1",w[indmin])
        # print("indmin2",w[indmin2])
        # print("negative w",w[w.real<0])
        # print("imag w",w[np.abs(w.imag)>0])
        # print('h is symmetric',np.allclose(h,h.T, rtol=1e-02, atol=1e-4))
        # plt.figure()
        # plt.plot(minvec[1::3].real)
        # plt.plot(minvec[1::3].imag)
        # print(w[np.abs(w.imag)>1e-10])
        return w, v, indmin, minvec

    def perturb_eq(fname_hdf5, psi_w):
        # takes an .end file (with the modes at the end)

        w, v = linalg.eig(get_hessian(fname_hdf5))
        ind_min_eigmode = np.argmin(w)
        min_eigmode = v[:, ind_min_eigmode]

        # "adjust" the kick
        kick =  np.real(min_eigmode[1::2])
        kick /= np.max(np.abs(kick))
        kick *= (0.78 * np.pi * psi_w)
        kick *= np.sign(kick[0])
        kick[-1] = 0

        # create new .sp file and add perturbation (from previous END FILE)
        subprocess.run(f"cp {fname_hdf5[:-3]}.end pert_{fname_hdf5[:-3]}", shell=True)

        inputnml = SPECNamelist(f"pert_{fname_hdf5[:-3]}")

        inputnml['physicslist']['Mpol'] = 1
        inputnml['physicslist']['Lconstraint'] = 3
        inputnml['locallist']['LBeltrami'] = 4
        inputnml['numericlist']['Linitialize'] = 0
        inputnml['globallist']['Lfindzero'] = 0
        inputnml['diagnosticslist']['LHmatrix'] = False
        for i in range(len(kick)):
            inputnml.set_interface_guess(kick[i], 1, 0, i, 'Rbc')

        inputnml.write_simple(f"pert_{fname_hdf5[:-3]}")

    def get_spec_energy(fname):
        data = SPECout(fname)
        energy_iters = np.array([data.iterations[i][1] for i in range(len(data.iterations))])
        # print(f"SPEC energy {SPECslab.print_indentd(energy_iters)}\n")
        return energy_iters

    def get_spec_force_err(fname):
        data = SPECout(fname)
        return data.output.ForceErr

    def get_resinface_width(fname):
        data = SPECout(fname)
        x = data.output.Rbc - np.pi
        x_min = np.max(np.where(x < 0, x, -100.0))
        x_max = np.min(np.where(x > 0, x, 100.0))
        width = x_max - x_min
        return width

    def get_spec_jacobian(data, lvol, sarr, theta, zeta):

        rtor = data.input.physics.rtor
        rpol = data.input.physics.rpol
        G = data.input.physics.Igeometry
        Rarr, Zarr = SPECslab.get_spec_R_derivatives(data, lvol, sarr, theta, zeta, 'R')

        if(G == 1):
            return Rarr[1] * rtor * rpol # fixed here
        elif(G == 2):
            return Rarr[0] * Rarr[1]
        elif(G == 3):
            return Rarr[0] * (Rarr[2]*Zarr[1] - Rarr[1]*Zarr[2])
        else:
            raise ValueError("Error: unsupported dimension")

    def get_spec_R_derivatives(data, vol, sarr, theta, zeta, RorZ):
        # the vol index is -1 compared to the matlab one

        mn = data.output.mn
        Rmn = data.output.Rbc[vol,:]
        Rmn_p = data.output.Rbc[vol+1,:]
        Zmn = data.output.Zbs[vol,:]
        Zmn_p = data.output.Zbs[vol+1,:]
        im = data.output.im
        iN = data.output.in_
        ns = len(sarr)

        factor = SPECslab.get_spec_reg_factor(data, vol, sarr, 'G')

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

    def get_spec_poly_basis(data, lvol, sarr):

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

    def get_contra2cov(data, lvol, vec_contrav, sarr, theta, phi, norm):

        ns = len(sarr)
        g = SPECslab.get_spec_metric(data, lvol, sarr, theta, phi)

        vec_cov = np.einsum('xas,as->xs', g, vec_contrav)
        return vec_cov

    def get_spec_reg_factor(data, lvol, sarr, ForG):

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

    def get_spec_metric(data, lvol, sarr, theta, zeta):

        ns = len(sarr)
        G = data.input.physics.Igeometry
        rtor = data.input.physics.rtor
        rpol = data.input.physics.rpol

        gmat = np.zeros((3,3,ns))
        Rarr, Zarr = SPECslab.get_spec_R_derivatives(data, lvol, sarr, theta, zeta, 'R')

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

    def get_spec_radius(data, theta, zeta, vol):
        # Return the radial position of a KAM surface for a given theta, zeta and Nvol

        G = data.input.physics.Igeometry
        Rmn = data.output.Rbc
        Zmn = data.output.Zbs
        im = data.output.im
        iN = data.output.in_

        if(G == 1):
            r_out = np.sum(Rmn[vol+1] * np.cos(im*theta - iN*zeta))
            z_out = 0
        elif(G == 2):
            r_out = np.dot(Rmn[vol+1,:], np.cos(im*theta - iN*zeta))
            z_out = 0
        elif(G == 3):
            r_out = np.dot(Rmn[vol+1,:], np.cos(im*theta - iN*zeta))
            z_out = np.dot(Zmn[vol+1,:], np.sin(im*theta - iN*zeta))

        return r_out, z_out

    def plot_kamsurf(fname, title=None):

        if(fname[-3:] == '.sp'):
            fname = fname+".h5"
        elif(fname[-4:] == '.end'):
            fname = fname[:-4]+".h5"

        if(not os.path.isfile(fname)):
            print(f"File '{fname}' does not exist (plot_kamsurf())")
            return

        if(title is None):
            title = fname

        fig, ax = plt.subplots(1,1)
        myspec = SPECout(fname)
        surfs = myspec.plot_kam_surface(marker='o', s=2., ax=ax)

        ax.set_ylim([0-0.2,2*np.pi+0.2])
        ax.set_xlim([0-0.1,2*np.pi+0.1])

        fig.canvas.set_window_title(title)
        plt.title(title)
        fig.set_size_inches(7,9)
        fig.tight_layout()

        # thetas = np.linspace(0, 2*np.pi, 200)
        # x_four = myspec.output.Rbc[:,:]
        # x = np.zeros((x_four.shape[0], len(thetas)))
        # for i in range(x_four.shape[1]):
        #     x += np.cos(i*thetas)[None,:] * (x_four[:,i])[:,None]
        #
        # plt.figure()
        # for vol in range(x_four.shape[0]):
        #     plt.plot(thetas, x[vol])

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

        T = np.zeros((Lrad + 1, 2, ns))
        T[0,0] = np.ones(ns)
        T[0,1] = np.zeros(ns)
        T[1,0] = sarr
        T[1,1] = np.ones(ns)

        for l in range(2, Lrad+1):
            T[l,0] = 2 * sarr * T[l-1,0] - T[l-2,0]
            T[l,1] = 2 * T[l-1,0] + 2 * sarr * T[l-1,1] - T[l-2,1]

        fac = np.ones((ns, 2, ns))
        fac[:,1] = 0

        a = im[:,None,None] * tarr[None,:,None] - iN[:,None,None] * zarr[None,None,:]
        term_t =  Ate[:,:,None,None]*np.cos(a[None,...]) + Ato[:,:,None,None]*np.sin(a[None,...])
        term_z = Aze[:,:,None,None]*np.cos(a[None,...]) + Azo[:,:,None,None]*np.sin(a[None,...])

        At_dAt = np.einsum('sa,lia,ljtz->istz', fac[:,0], T, term_t)
        Az_dAz = np.einsum('sa,lia,ljtz->istz', fac[:,0], T, term_z)

        Az = np.zeros((ns,nt,nz))
        for j in range(mn):
            for l in range(Lrad+1):
                for t in range(nt):
                    for z in range(nz):
                        a = (im[j]*tarr[t]-iN[j]*zarr[z])
                        Az[:,t,z] += fac[j,0]*T[l,0] * (np.cos(a)*Aze[l,j]+np.sin(a)*Azo[l,j])

        return At_dAt[0], Az, At_dAt[1], Az_dAz[1]

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


        fac = np.ones((mn, 2, ns))
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

    def get_islandwidth(fname, ns, nt, plot=False, plot_title=None):

        if(not os.path.isfile(fname)):
            print(f"File '{fname}' does not exist (get_islandwidth())")
            return

        data = SPECout(fname)

        fdata = data.vector_potential
        lvol =  data.input.physics.Nvol // 2
        sarr = np.linspace(-1, 1, ns)
        tarr = np.linspace(0, 2*np.pi, nt)

        z0   = np.array([0])
        r0   = data.output.Rbc[-1,0]/2

        # std approach
        At, Az, dAt, dAz = SPECslab.get_vecpot(data, lvol, sarr, tarr, z0)
        Rarr, Tarr, dRarr = SPECslab.get_rtarr(data, lvol, sarr, tarr, z0)

        o_point = np.unravel_index((Az[:,:,0]).argmin(), Az[:,:,0].shape)
        x_point = np.unravel_index(((Az[:,:,0])**2).argmin(), Az[:,:,0].shape)

        c = plt.contour(Tarr, Rarr, Az[:,:,0], levels=[Az[x_point][0]],colors='red',linestyles='solid')
        plt.close()

        cont_pts = []
        for i in range(len(c.collections[0].get_paths())):
            if(np.std(c.collections[0].get_paths()[i].vertices[:,1]) > 0.001):
                cont_pts.append(c.collections[0].get_paths()[i].vertices)

        if(plot):
            fig = plt.figure()
            plt.contourf(Tarr, Rarr, Az[:,:,0], levels=40)
            plt.colorbar()

        if(len(cont_pts) < 1):
            island_w = 0.0
        else:
            cont_pts = np.concatenate(cont_pts)
            island_w = np.max(cont_pts[:, 1]) - np.min(cont_pts[:, 1])

            if(plot):
                for i in range(len(c.collections[0].get_paths())):
                    if(np.std(c.collections[0].get_paths()[i].vertices[:,1]) > 0.05):
                        plt.plot(c.collections[0].get_paths()[i].vertices[:,0], c.collections[0].get_paths()[i].vertices[:,1], 'k-.')
                plt.axhline(np.min(cont_pts[:, 1]), color='red', linestyle='dashed')
                plt.axhline(np.max(cont_pts[:, 1]), color='red', linestyle='dashed')
                plt.plot(Tarr[x_point],Rarr[x_point],'rX', ms=13)
                plt.plot(Tarr[o_point],Rarr[o_point],'bo', ms=13)

        if(plot):
            if(plot_title is None):
                plot_title = "A_z resonant volume"
            plt.title(plot_title + f" w={island_w:.3f}")
            fig.canvas.set_window_title(plot_title + f" w={island_w:.3f}")

            plt.ylim([np.pi-1.,np.pi+1.])
            # plt.ylim([0, 2*np.pi])

            plt.xlim([0, 2*np.pi])
            # plt.xlim([0, np.pi+0.1])

        print(f"Island width is {island_w:.4f}")

        return island_w


    def plot_vecpot_volumes(fname, lvol_list, ns, nt, title=None):

        if(not os.path.isfile(fname)):
            print(f"File '{fname}' does not exist (plot_vecpot_volumes())")
            return

        if(title is None):
            title = 'Plot of A_z vector potential'

        data = SPECout(fname)

        sarr = np.linspace(-1, 1, ns)
        tarr = np.linspace(0, 2*np.pi, nt)
        z0 = np.array([0])

        fig, ax = plt.subplots(1, 1)
        plt.title(title)
        fig.canvas.set_window_title('A_z')

        for lvol in lvol_list:
            Rarr, Tarr, dRarr = SPECslab.get_rtarr(data, lvol, sarr, tarr, z0)
            At, Az, dAt, dAz = SPECslab.get_vecpot(data, lvol, sarr, tarr, z0)
            ax.contourf(Tarr,Rarr,Az[:,:,0], levels=40)

        return ax

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

        r0, z0 = SPECslab.get_spec_radius(data, theta, zeta, -1)
        B = np.zeros((3, len(r)))
        for i in range(nvol):

            ri, zi = SPECslab.get_spec_radius(data, theta, zeta, i-1)
            rmin = np.sqrt((ri-r0)**2 + zi**2)

            ri, zi = SPECslab.get_spec_radius(data, theta, zeta, i)
            rmax = np.sqrt((ri-r0)**2 + zi**2)
            r_vol = np.linspace(rmin, rmax, nr+1)#[1:]

            sarr = 2 * (r_vol - rmin) / (rmax - rmin) - 1
            if(i == 0 and G != 1):
                sarr = 2 * ((r_vol - rmin) / (rmax - rmin))**2 - 1

            B_contrav = SPECslab.get_spec_magfield(data, i, sarr, theta, zeta)

            iimax = iimax + len(r_vol)

            B_cov = SPECslab.get_contra2cov(data, i, B_contrav, sarr, theta, zeta, 1)

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

    def get_spec_magfield(data, lvol, sarr, theta, zeta):

        jac = SPECslab.get_spec_jacobian(data, lvol, sarr, theta, zeta)

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

        T = SPECslab.get_spec_poly_basis(data, lvol, sarr)

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

    def run_spec_master(fname, show_output=False):

        print(f"\n{('-'*80)}\nRunning SPEC with {fname}")

        # # if running from jupyer, redirect output appropriately
        # from IPython import get_ipython
        # if(get_ipython() is not None):
        #     print("Using jupyer nb")
        #
        #     with subprocess.Popen(f'mpirun -n 6 ~/SPEC/xspec {fname}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
        #         for line in p.stdout:
        #             print(line, end='') # process line here
        #
        #     if p.returncode != 0:
        #         raise CalledProcessError(p.returncode, p.args)
        #
        #     return True

        try:
            if(show_output):
                subprocess.run(f'mpirun -n 6 ~/SPEC/xspec {fname}', shell=True)
            else:
                subprocess.run(f'mpirun -n 6 ~/SPEC/xspec {fname}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            data = SPECout(fname+".h5")
            print(f"\nSPEC completed")
            print(f"force error {data.output.ForceErr:.4e}")
            # print("ForceErr iterations",SPECslab.print_indentd(np.array([data.iterations[i][2] for i in range(len(data.iterations))])))
            print("\n")

            return True

        except Exception:
            print("Running SPEC failed!!!")
            return False

    def run_spec_descent(fname, show_output=False):

        print(f"\n{('-'*80)}\nRunning SPEC with {fname}")

        try:
            if(show_output):
                subprocess.run(f'mpirun -n 6 ~/spec_descent/SPEC/xspec {fname}', shell=True)
            else:
                subprocess.run(f'mpirun -n 6 ~/spec_descent/SPEC/xspec {fname}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            data = SPECout(fname+".h5")
            print(f"\nSPEC completed")
            print(f"force error {data.output.ForceErr:.4e}")
            # print("ForceErr iterations",SPECslab.print_indentd(np.array([data.iterations[i][2] for i in range(len(data.iterations))])))
            print("\n")

            return True

        except Exception:
            print("Running SPEC failed!!!")
            return False


    def get_spec_error(fname):
        data = SPECout(fname)
        return data.output.ForceErr

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

    def check_profile_Bfield(fname, func_by0, func_bz0):

        fig, ax = plt.subplots(1, 1)
        plt.title('Plot of magnetic field B')
        fig.canvas.set_window_title('Field B')

        data = SPECout(fname)

        theta = np.pi
        phi = 0
        num_radpts = 300
        num_radpts_pervol = 10
        r = np.linspace(0, 2*np.pi, num_radpts, endpoint=True)

        B = SPECslab.get_full_field(data, r, theta, phi, num_radpts_pervol)

        ax.plot(r, B[0], '.-', picker=SPECslab.line_picker, label='spec B_psi')
        ax.plot(r, B[1], '.-', picker=SPECslab.line_picker, label='spec B_theta')
        ax.plot(r, B[2], '.-', picker=SPECslab.line_picker, label='spec B_ phi')

        ax.plot(r, func_by0(r - np.pi), '--', picker=SPECslab.line_picker, label='B_y0')
        ax.plot(r, func_bz0(r-np.pi), '--', picker=SPECslab.line_picker, label='B_z0')

        ax.legend()
        ax.set_xlabel("Radial coordinate r / x [m]")
        ax.set_ylabel("Field strength [T]")

        fig.canvas.mpl_connect('pick_event', SPECslab.onpick2)
        fig.tight_layout()

    def check_profile_curr(fname, func_jy0, func_jz0):

        fig, ax = plt.subplots(1, 1)
        plt.title('Plot of current J')
        fig.canvas.set_window_title('Current J')

        data = SPECout(fname)

        theta = np.pi
        phi = 0
        num_radpts = 1000
        num_radpts_pervol = 20
        r = np.linspace(0, 2*np.pi, num_radpts, endpoint=True)

        mu_spec = data.output.mu
        J = SPECslab.get_full_field(data, r, theta, phi, num_radpts_pervol, mu_spec)

        # ax.plot(r, J[0], '.-', picker=SPECslab.line_picker, label='SPEC J_psi')
        # ax.plot(r, J[1], '.-', picker=SPECslab.line_picker, label='SPEC J_theta')
        # ax.plot(r, J[2], '.-', picker=SPECslab.line_picker, label='SPEC J_phi', alpha=0.8)

        ax.plot(r, func_jy0(r - np.pi), '--', picker=SPECslab.line_picker, label='analytic J_y0')
        ax.plot(r, func_jz0(r - np.pi), '--', picker=SPECslab.line_picker, label='analytic J_z0')

        rslab = data.input.physics.rtor
        print(data.input.physics.Ivolume)
        ivol = np.diff(data.input.physics.Ivolume) / (2*np.pi*rslab)
        x_spec = data.output.Rbc[:,0]
        SPECslab.taired(ax, ivol, x_spec, "SPEC ivol")

        ax.legend()
        ax.set_xlabel("Radial coordinate r / x [m]")
        ax.set_ylabel("Current [$\mu_0$]")

        fig.canvas.mpl_connect('pick_event', SPECslab.onpick2)
        fig.tight_layout()

    def plot_staired(ax, field, x, label):

        ax.hlines(field, x, x[1:], lw=2.0, label=label)
        l = ax.plot(0.5*(x[1:]+x[:-1]), field, alpha=0, picker=SPECslab.line_picker)
        l[0].set(label=label)

    def check_profile_mu(fname, func_by0, func_bz0, func_jy0, func_jz0):
        """
        Plots mu in each volume (piecewise constant)
        """
        fig, ax = plt.subplots(1, 1)
        plt.title('Plot of $\mu$')
        fig.canvas.set_window_title('Mu')

        data = SPECout(fname)

        mu_spec = data.output.mu
        x_spec = data.output.Rbc[:,0]
        x_spec_vol = (x_spec[:-1]+x_spec[1:])*.5
        SPECslab.plot_staired(ax, mu_spec, x_spec, "SPEC mu")

        x_array =  np.linspace(np.min(x_spec), np.max(x_spec), 200) - np.pi
        mu_analytic = SPECslab.mu(x_array, func_by0, func_bz0, func_jy0, func_jz0)

        ax.plot(x_array + np.pi, mu_analytic, '--', lw=1.0, label="analytic mu", picker=SPECslab.line_picker)

        fig.canvas.mpl_connect('pick_event', SPECslab.onpick2)
        ax.legend()

        fig.tight_layout()

    def check_profile_flux(fname, func_psi0=None, func_bz0=None):
        """
        Plots flux encloses by each interface (pflux, tflux); compares to analytic expressions
        """
        fig, ax = plt.subplots(1, 1)
        plt.title('Plot of poloidal and toroidal fluxes')
        fig.canvas.set_window_title('Flux')

        psi_flag = bz_flag = True
        if(func_psi0 is None):
            psi_flag = False
        if(func_bz0 is None):
            bz_flag = False

        data = SPECout(fname)

        phiedge_spec = data.input.physics.phiedge
        x_spec = data.output.Rbc[1:,0]

        pflux = data.output.pflux * phiedge_spec
        tflux = data.output.tflux
        rslab = data.input.physics.rtor

        ax.plot(SPECslab.join_arrays([0],x_spec), SPECslab.join_arrays([0],pflux), 'd', label='SPEC pflux', picker=SPECslab.line_picker)
        ax.plot(SPECslab.join_arrays([0],x_spec), SPECslab.join_arrays([0],tflux), 'd', label='SPEC tflux', picker=SPECslab.line_picker)

        x_array =  np.linspace(np.min(x_spec), np.max(x_spec), 200) - np.pi
        x_array = SPECslab.join_arrays([-np.pi], x_array)

        if(psi_flag):
            pflux_analytic = func_psi0(x_array) * (2 * np.pi * rslab)
            ax.plot(x_array + np.pi, pflux_analytic, '.--', lw=2.0, label="analytic pflux", picker=SPECslab.line_picker)

        if(bz_flag):
            tflux_analytic = SPECslab.norm_torflux(x_array, func_bz0)
            ax.plot(x_array + np.pi, tflux_analytic, '.--', lw=2.0, label="analytic tflux", picker=SPECslab.line_picker)

        fig.canvas.mpl_connect('pick_event', SPECslab.onpick2)
        ax.legend()
        fig.tight_layout()

    def check_profile_iota(fname, func_by0, func_bz0):
        """
        Plots iota obtained from SPEC fieldline tracing; compares to analytic form
        """
        fig, ax = plt.subplots(1, 1)
        plt.title('Plot of iota')
        fig.canvas.set_window_title('Iota')

        data = SPECout(fname)

        rtor = data.input.physics.rtor

        theta = np.pi
        phi = 0
        num_radpts = 200
        num_radpts_pervol = 10
        r = np.linspace(0, 2*np.pi, num_radpts, endpoint=True)

        B = SPECslab.get_full_field(data, r, theta, phi, num_radpts_pervol)

        iota_spec = B[1] / B[2] # in simple case here

        iota_analytic = SPECslab.iota_slab(r - np.pi, func_by0, func_bz0, rtor, rtor)
        ax.plot(r, iota_analytic, '--', lw=2.0, label="analytic iota", picker=SPECslab.line_picker)

        ax.plot(r, iota_spec, '.', label='SPEC iota', picker=SPECslab.line_picker)

        ## same as iota_spec
        # fiota = data.transform.fiota
        # x_spec = data.output.Rbc[:,0]
        # iota_in = fiota[1, fiota[0] < (-1+1e-8)]
        # iota_out = fiota[1, fiota[0] > (1-1e-8)]
        # ax.plot(x_spec[:-1], iota_in, '-.', label='SPEC iota in (tracing)', picker=SPECslab.line_picker)
        # ax.plot(x_spec[1:], iota_out, '-.', label='SPEC iota out (tracing)', picker=SPECslab.line_picker)

        fig.canvas.mpl_connect('pick_event', SPECslab.onpick2)
        ax.legend()
        fig.tight_layout()

    def get_b0b0hat_vol1(fname):

        data = SPECout(fname)
        mu = data.output.mu[1]
        delta = data.output.Rbc[1,0]
        pflux = data.output.pflux[0] * data.input.physics.phiedge
        tflux = data.output.tflux[0] * data.input.physics.phiedge

        mubar = mu * delta / 2
        b0hat = pflux * mubar / (2*np.pi*delta*np.sin(mubar))
        b0 = tflux * mubar / (2*np.pi*delta*np.sin(mubar))

    def quick_plot(ydata, xdata=None):

        fig, ax = plt.subplots(1, 1)

        if(xdata is None):
            ax.plot(ydata, 'd', picker=SPECslab.line_picker)
        else:
            ax.plot(xdata, ydata, 'd', picker=SPECslab.line_picker)

        fig.canvas.mpl_connect('pick_event', SPECslab.onpick2)
        fig.tight_layout()

        return fig

    def run_spec_slab(inpdict, show_output=False):

        inpdict.set_if_none('fname_template', 'template.sp')
        inpdict.set_if_none('fname_outh5', inpdict.fname_input+'.h5')
        inpdict.set_if_none('pre', np.zeros(inpdict.Nvol))
        inpdict.set_if_none('linitialize', 0)
        inpdict.set_if_none('lbeltrami', 4)
        inpdict.set_if_none('psi0_mag', 3 * np.sqrt(3) / 4)
        inpdict.set_if_none('bz0_mag', 10.0)
        inpdict.set_if_none('lrad', 8 * np.ones(inpdict.Nvol, dtype=int))

        x_resolution = 2**14
        x_array = np.linspace(-np.pi, np.pi, x_resolution)

        ind_Lresinface, ind_Rresinface = SPECslab.infacesres_ind(x_array, inpdict.psi_w, inpdict.bz0)
        if(inpdict.infaces_spacing == 'uniform'):
            infaces_ind, infaces_x, infaces_psi = SPECslab.gen_ifaces_funcspaced(x_array, ind_Lresinface, ind_Rresinface, lambda x: x, inpdict.Nvol) #uniform spacing
        elif(inpdict.infaces_spacing == 'psi'):
            infaces_ind, infaces_x, infaces_psi = SPECslab.gen_ifaces_funcspaced(x_array, ind_Lresinface, ind_Rresinface, inpdict.psi0, inpdict.Nvol)
        else:
            raise ValueError("infaces_spacing has to be either uniform or psi")

        # phi_edge = 2 * np.pi * inpdict.rslab * integrate.trapz(inpdict.bz0(x_array), x_array)
        # tflux = SPECslab.norm_torflux(x_array, inpdict.bz0)[infaces_ind]

        tflux = np.zeros(len(infaces_x))
        for t in range(len(infaces_x)):
            tflux[t] = integrate.quad(inpdict.bz0, -np.pi, infaces_x[t])[0]
        phi_edge = tflux[-1] * 2 * np.pi * inpdict.rslab
        tflux /= tflux[-1]

        pflux = inpdict.psi0(infaces_x) * (2 * np.pi * inpdict.rslab)

        mu_vol = np.zeros(inpdict.Nvol)
        mu_func = lambda x: SPECslab.mu(x, inpdict.by0, inpdict.bz0, inpdict.jy0, inpdict.jz0)
        for t in range(0, inpdict.Nvol):
            mu_vol[t] = integrate.quad(mu_func, infaces_x[t], infaces_x[t+1])[0] / (infaces_x[t+1] - infaces_x[t])

        iota_vol = SPECslab.iota_slab(infaces_x, inpdict.by0, inpdict.bz0, inpdict.rslab, inpdict.rslab)

        curr_array = SPECslab.torflux(x_array, inpdict.jz0) * (2 * np.pi * inpdict.rslab)
        curr_vol = curr_array[infaces_ind]


        # vecpot_y = SPECslab.join_arrays([0.0], integrate.cumtrapz(inpdict.bz0(x_array), x_array))
        # vecpot_z = SPECslab.join_arrays([0.0], -integrate.cumtrapz(inpdict.by0(x_array), x_array))
        # helicity = np.zeros(inpdict.Nvol)
        # for i in range(inpdict.Nvol):
        #     x_mask = np.logical_and(x_array < infaces_x[i+1], x_array > infaces_x[i])
        #
        #     vecpot_y_curr = vecpot_y[x_mask]
        #     vecpot_y_curr -= vecpot_y_curr[0]
        #     vecpot_z_curr = vecpot_z[x_mask]
        #     vecpot_z_curr -= vecpot_z_curr[0]
        #     vecpot_dot_B = vecpot_y_curr * inpdict.by0(x_array[x_mask]) + vecpot_z_curr * inpdict.bz0(x_array[x_mask])
        #
        #     # helicity[i] = np.mean(vecpot_dot_B) * (infaces_x[i+1] - infaces_x[i])
        #     helicity[i] = np.trapz(vecpot_dot_B, x_array[x_mask])
        # helicity *= (2 * np.pi * inpdict.rslab) ** 2

        # another method to calculate H
        helicity = np.zeros(inpdict.Nvol)
        def vecpot_y(x):
            return integrate.quad(inpdict.bz0, -np.pi, x)[0]
        for i in range(inpdict.Nvol):
            helicity[i] = (2*np.pi*inpdict.rslab)**2 * integrate.quad(
                lambda x: (vecpot_y(x)-vecpot_y(infaces_x[i])) * inpdict.by0(x) -
                (inpdict.psi0(x)-inpdict.psi0(infaces_x[i])) * inpdict.bz0(x),
            infaces_x[i], infaces_x[i+1])[0]

        ## create and edit the namelist file
        subprocess.run(f"cp {inpdict.fname_template} {inpdict.fname_input}", shell=True)
        inputnml = SPECNamelist(inpdict.fname_input)
        inputnml['physicslist']['pressure'] = inpdict.pre
        inputnml['globallist']['Lfindzero'] = inpdict.lfindzero # 0 for only field calculation, 2 for actually moving interfaces
        inputnml['physicslist']['Lrad'] = inpdict.lrad
        inputnml['physicslist']['phiedge'] = phi_edge
        inputnml['physicslist']['Nvol'] = inpdict.Nvol
        inputnml['physicslist']['Mpol'] = inpdict.Mpol

        ## init geometry
        inputnml['numericlist']['Linitialize'] = 0 # 0 for custom Rbc, 1 for automatic interpolation
        inputnml['physicslist']['rpol'] = inpdict.rslab
        inputnml['physicslist']['rtor'] = inpdict.rslab
        inputnml.interface_guess[(0,0)]['Rbc'] = infaces_x[1:] + np.pi
        inputnml.interface_guess[(1,0)]['Rbc'] = np.zeros_like(infaces_x[1:])

        ## init profiles
        inputnml['physicslist']['Lconstraint'] = inpdict.lconstraint # should be 0 for constant psi_p, psi_t, mu

        if(inputnml['physicslist']['Lconstraint'] == 0):
            SPECslab.check_psi(x_array, inpdict.psi0)
            inputnml['physicslist']['tflux'] = tflux[1:]
            inputnml['physicslist']['pflux'] = pflux[1:] / phi_edge
            inputnml['physicslist']['mu'] = mu_vol

        elif(inputnml['physicslist']['Lconstraint'] == 1):
            inputnml['physicslist']['iota'] = iota_vol
            inputnml['physicslist']['oita'] = iota_vol
            inputnml['physicslist']['tflux'] = tflux[1:]
            inputnml['physicslist']['mu'] = mu_vol

        elif(inputnml['physicslist']['Lconstraint'] == 2):
            inputnml['physicslist']['pflux'] = pflux[1:] / phi_edge
            inputnml['physicslist']['tflux'] = tflux[1:]
            inputnml['physicslist']['helicity'] = helicity

        elif(inputnml['physicslist']['Lconstraint'] == 3):
            SPECslab.check_psi(x_array, inpdict.psi0)
            inputnml['physicslist']['isurf'] = np.zeros(inpdict.Nvol)
            inputnml['physicslist']['ivolume'] = curr_vol[1:]#*1.03
            inputnml['physicslist']['tflux'] = tflux[1:]
            inputnml['physicslist']['pflux'] = pflux[1:] / phi_edge
        else:
            raise ValueError("Lconstraint has to be 0,1,2, or 3")

        # Make sure that the simple, direct solver is used
        # Bcs gmres will fail for mpol>3
        inputnml['locallist']['Lmatsolver'] = 1

        inputnml.write_simple(inpdict.fname_input)

        SPECslab.run_spec_master(inpdict.fname_input, show_output=show_output)


    def save_figures_pdf():
        with matplotlib.backends.backend_pdf.PdfPages("output.pdf") as pdf:
            figs = list(map(plt.figure, plt.get_fignums()))
            for f in figs:
                pdf.savefig(f)

    def check_intersect(fname):
        x = SPECslab.get_infaces_pts(fname)
        insect_flag = check_intersect_helper(x, x.shape[0])
        return insect_flag

    def check_intersect_initial(inputnml, plot_flag=False):
        infaces_fourier = inputnml.interface_guess
        thetas = np.linspace(0, 2*np.pi, 200)
        x = np.zeros((inputnml['physicslist']['Nvol'], len(thetas)))

        for i in range(inputnml['physicslist']['Mpol']+1):
            # print(np.array(infaces_fourier[(i,0)]['Rbc']))
            x += np.cos(i*thetas)[None,:] * (np.array(infaces_fourier[(i,0)]['Rbc']))[:,None]

        x = np.vstack([np.zeros_like(thetas), x])

        if(plot_flag):
            plt.figure()
            for vol in range(inputnml['physicslist']['Nvol']+1):
                plt.plot(thetas, x[vol])

        insect_flag = check_intersect_helper(x, x.shape[0])
        return insect_flag

    def get_infaces_pts(fname):
        data = SPECout(fname)
        thetas = np.linspace(0, 2*np.pi, 200)
        x_four = data.output.Rbc[:,:]
        x = np.zeros((x_four.shape[0], len(thetas)))
        for i in range(x_four.shape[1]):
            x += np.cos(i*thetas)[None,:] * (x_four[:,i])[:,None]

        # plt.figure()
        # for vol in range(x_four.shape[0]):
        #     plt.plot(thetas, x[vol])

        return x

    def plot_descent_iters(fname):

        if(not os.path.isfile(fname)):
            print(f"File '{fname}' does not exist (plot_descent_iters())")
            return

        data = SPECout(fname)

        if(data.input.global1.Lfindzero != 3):
            return

        plt.figure("Force error")
        plt.semilogy(np.array([data.iterations[i][2] for i in range(len(data.iterations))]))
        plt.title("Force error for descent iterations")

        plt.figure("Energy")
        plt.plot(np.array([data.iterations[i][2] for i in range(len(data.iterations))])-data.iterations[0][2])
        plt.title("Energy for descent iterations")

        return

    def run_slab_loureiro(inpdict, show_output=False):

        a = 0.35

        inpdict.fname_outh5 = inpdict.fname_input+'.h5'

        def psi0(x, psi0_mag=inpdict.psi0_mag):
            # initial flux func
            return psi0_mag / np.cosh(x)**2 - psi0_mag / np.cosh(-np.pi)**2
        def by0(x, psi0_mag=inpdict.psi0_mag):
            # poloidal field
            return -2 * psi0_mag * np.sinh(x) / np.cosh(x)**3
        def jz0(x, psi0_mag=inpdict.psi0_mag):
            # toroidal current
            return 4 * psi0_mag * np.sinh(x)**2 / np.cosh(x)**4 - 2 * psi0_mag / np.cosh(x)**4
        def bz0(x, bz0_mag=inpdict.bz0_mag):
            # toroidal field
            return np.sqrt(bz0_mag**2 - by0(x)**2)
        def jy0(x):
            # poloidal current
            return -(4 * by0(x)**2 * np.tanh(x) - 4 * by0(x)**2 / np.sinh(2*x)) / (2*bz0(x))

        inpdict.psi0 = psi0
        inpdict.by0 = by0
        inpdict.bz0 = bz0
        inpdict.jy0 = jy0
        inpdict.jz0 = jz0

        SPECslab.run_spec_slab(inpdict, show_output)

        print(f"Del\' * a {SPECslab.calc_delprime(1/inpdict.rslab)*a:.3f} rslab {inpdict.rslab:.3f} psi_w {inpdict.psi_w:.3f}\n")

        SPECslab.get_spec_energy(inpdict.fname_outh5)
        # SPECslab.plot_kamsurf(inpdict.fname_outh5, f"init kam surfaces psi_w={inpdict.psi_w:.3f}")

    def add_perturbation(input_fname, ouput_fname, kick_amplitude=0.8, max_num_iters=200):

        subprocess.run(f"cp {input_fname} {ouput_fname}", shell=True)
        inputnml = SPECNamelist(ouput_fname)

        eigval, eigvec, min_eigval_ind, min_eigvec = SPECslab.get_eigenstuff(input_fname)
        if(eigval[min_eigval_ind] > 0.0):
            print("Smallest eigenvalue of the force-gradient matrix is positive, system is stable!")
            return False

        perturbation = np.real(min_eigvec[1::inputnml._Mpol+1])
        perturbation /= np.max(np.abs(perturbation))
        perturbation *= np.sign(perturbation[1*len(perturbation)//4])
        perturbation *= kick_amplitude

        # perturbation[-1] = 0
        # perturbation[inpdict.Nvol//2:] *= 10
        # perturbation[:inpdict.Nvol//2] *= 10
        #     # perturbation *= 0

        print("Finding good pertubation amplitude... ",end='')

        for n in range(max_num_iters):
            # print(n, end=' ')
            for i in range(inputnml._Nvol-1):
                inputnml.set_interface_guess(perturbation[i], 1, 0, i, key="Rbc")

            isect_flag = SPECslab.check_intersect_initial(inputnml, False)
            if(isect_flag):
                if(n == max_num_iters-1):
                    print("\nERROR: Couldn't find pertubation amplitude!")
                    return False
                perturbation *= 0.98
            else:
                break
        print(n)

        inputnml.write_simple(ouput_fname)
        SPECslab.check_intersect_initial(inputnml, False)

        return True


class input_dict(dict):
    """input dictionary class
        -> with dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def set_if_none(self, key, val):
        if(key not in self.keys()):
            self[key] = val

    def has_key(self, key):
            if(key in self.keys()):
                return True
            else:
                return False

@numba.njit(cache=True)
def check_intersect_helper(x, nvol):
    for v in range(nvol):
        for sv in range(v-1, v+1):
            diff = x[v] - x[sv]
            for i in range(len(diff)-1):
                if((diff[i]*diff[i+1]) < 0):
                    return True
    return False

if __name__ == "__main__":

    print("SPECslab contains tools for running and analyzing SPEC in slab geometry (G=1)")
