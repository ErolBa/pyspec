## Written by Erol Balkovic

import numpy as np
import matplotlib.pyplot as plt
import h5py
from py_spec import SPECNamelist
from . import SPECslab
from scipy import integrate, interpolate, optimize
import subprocess
import sys
import numba
import warnings
import sympy as sym
import glob
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline
import matplotlib
from scipy.fftpack import fft
import matplotlib.animation as anim
from IPython.display import HTML
import os
import contextlib
from tqdm.auto import tqdm

class HMHDslab():

    def set_inputfile_var(file, varname, varvalue):
        subprocess.run(f"sed -i s/^{varname}=.*/{varname}="+f"{varvalue:.2e}".replace('e','D')+f"\ \ \ \ !auto\ insert\ with\ py/ i{file}", shell=True)

    def get_hmhd_profiles(hmhd_fname, num_conts, mpol, plot_flag):

        numba.set_num_threads(4)

        if(plot_flag):
            fig, ax = plt.subplots()
            ax.set_title("Plot of $\psi(x,y)$")
            ax.set_xlabel("y")
            ax.set_ylabel("x")

        with h5py.File(hmhd_fname) as h5file:
            psi = h5file['psi'][2:-2, 2:-1]
            bz = h5file['bz'][2:-2, 2:-1]
            by = h5file['by'][2:-2, 2:-1]
            bx = h5file['bx'][2:-2, 2:-1]
            jz = h5file['jz'][2:-2, 2:-1]
            jy = h5file['jy'][2:-2, 2:-1]
            jx = h5file['jx'][2:-2, 2:-1]
            x = h5file["x"][2:-1]
            z = h5file["z"][2:-2]

        # roll the fields
        ind = np.arange(psi.shape[1])
        ind = np.roll(ind, len(ind)//2)
        ind[:len(ind)//2] -= 1
        for f in [psi, by, bx, bz, jx, jy, jz]:
            f[:,:] = f[:, ind]

        y_coord = x.copy() # transform to spec coords
        x_coord = z.copy() # transform to spec coords

        X, Y = np.meshgrid(x_coord[::1], y_coord[::1])
        c_levels = np.linspace(np.min(psi), np.max(psi), num_conts, endpoint=True)[1:-1]
        c = plt.contour(Y, X, psi.T, levels=c_levels, colors='black') #psi

        plt.ylim([np.min(x_coord)-0.1, np.max(x_coord)+0.1])
        plt.xlim([np.min(y_coord)-0.1, np.max(y_coord)+0.1])
        if(not plot_flag):
            plt.close()

        num_conts = len(c.collections)

        width = np.max(y_coord)

        rbc = []
        pflux = []
        tflux = []
        areas = []

        # by_func = RectBivariateSpline(z, x/w*np.pi+np.pi, by, kx=3, ky=3)
        by_func = interp2d(x_coord, y_coord, by, 3)
        bx_func = interp2d(x_coord, y_coord, bx, 3)
        mu_func = interp2d(x_coord, y_coord, (jx*bx+jy*by+jz*bz)/(bx**2+by**2+bz**2), 3)
        jy_func = interp2d(x_coord, y_coord, jy, 3)

        if(plot_flag):
            plt.contourf(Y, X, psi.T, levels=100, cmap='viridis', alpha=0.9)
            # plt.colorbar()

        mean_bx_vals = []
        mean_by_vals = []
        mean_mu_vals = []
        mean_jy_vals = []

        xgrid = np.linspace(np.min(x_coord), np.max(x_coord), 1600, endpoint=True)
        ygrid = np.linspace(np.min(y_coord), np.max(y_coord), 400, endpoint=False)
        xm, ym = np.meshgrid(xgrid, ygrid)

        ynew = np.linspace(np.min(y_coord)+1e-10, np.max(y_coord), 40, endpoint=True)

        if(plot_flag):
            plt.figure()

        for cont in range(num_conts): # iterate over each level
            level = c.allsegs[cont]
            color = tuple(np.random.rand(3))

            num_paths = len(level)
            if(num_paths==0):
                continue

            for p in range(len(level)):
                # print(f"cont {cont} -- p {p} len {len(level)}")

                path_array = np.array(level[p])
                x = path_array[:,1]
                y = path_array[:,0]

                # if(len(level)==1):
                #     plt.plot(y,x,'r-')
                # else:
                #     plt.plot(y,x,'b--')
                if(plot_flag):
                    plt.ylim([-np.pi*1.05, np.pi*1.05])

                # check if the path is flat (or belongs to an island
                if(np.abs(path_array[-1,0]-path_array[0,0]) < 1e-4):
                    pass
                else:
                    if(y[-1] < y[0]):
                        y = y[::-1]
                    f = InterpolatedUnivariateSpline(y, x)
                    xnew = f(ynew)   # use interpolation function returned by `interp1d`

                    ### get rbc, interfaces
                    ft = fft(xnew[:-1])
                    rbc.append(np.array([ft.real[m] if m==0 else 2 * ft.real[m] for m in range(mpol+1)])/len(ft))
                    rbc[-1][0] += np.pi

                    # check reconstructed interfaces are correct (and that ft was done well)
                    if(plot_flag):
                        xrecons = np.zeros_like(ynew)
                        xrecons += ft.real[0] / (len(ft))
                        for m in range(1, mpol+1):
                            xrecons += 2 * ft.real[m]*np.cos(m*(np.pi + np.pi * ynew / width)) / (len(ft))
                        plt.plot(ynew, xrecons , '-.', color=color, zorder=5)

                    mask = f(ym) > xm
                    if(np.sum(mask) < 1):
                        mean_bx_vals.append(0)
                        mean_by_vals.append(0)
                        continue

                    xm_masked = xm[mask].flatten()
                    ym_masked = ym[mask].flatten()

                    bx_vals = bx_func(xm_masked, ym_masked) # slow
                    mean_bx_vals.append(np.sum(bx_vals))

                    by_vals = by_func(xm_masked, ym_masked) # slow
                    mean_by_vals.append(np.sum(by_vals))

                    mu_vals = mu_func(xm_masked, ym_masked)
                    mean_mu_vals.append(np.sum(mu_vals))

                    jy_vals = jy_func(xm_masked, ym_masked)
                    mean_jy_vals.append(np.sum(jy_vals))

                    area = f.integral(np.min(y_coord), np.max(y_coord))
                    areas.append(area)

        rbc = np.array(rbc)
        interface_order = np.argsort(rbc[:,0])

        total_area = 2*np.pi * 2*width

        adj_bx_flux = np.array(mean_bx_vals)[interface_order] * (total_area / len(xm.flatten()))
        adj_by_flux = np.array(mean_by_vals)[interface_order] * (total_area / len(xm.flatten()))

        areas_vol = np.diff(np.array(areas)[interface_order])
        vol_center_pos = 0.5 * (rbc[interface_order,0][1:] + rbc[interface_order,0][:-1])

        adj_mean_mu = np.diff(np.array(mean_mu_vals)[interface_order])
        adj_mean_mu *= - (total_area / len(xm.flatten()))
        adj_mean_mu /= areas_vol

        adj_Ivol = np.diff(np.array(mean_jy_vals)[interface_order])
        adj_Ivol *= - (total_area / len(xm.flatten()))

        phiedge = adj_by_flux[-1]

        rbc = rbc[interface_order]

        if(plot_flag):
            plt.figure()
            plt.plot(rbc[:,0], adj_by_flux/phiedge, 'd-')
            plt.axvline(np.pi, color='r')

            plt.figure()
            plt.plot(rbc[:,0], -adj_bx_flux/phiedge, 'd-')
            plt.axvline(np.pi, color='r')

            plt.figure()
            plt.plot(vol_center_pos, adj_mean_mu, 'd-')
            plt.axvline(np.pi, color='r')

            plt.figure()
            plt.plot(vol_center_pos, adj_Ivol, 'd-')
            plt.axvline(np.pi, color='r')

        return rbc, adj_bx_flux, adj_by_flux, adj_mean_mu, adj_Ivol, width


    def gen_profiles_from_psi(psi_string):
        x, psi0_sym, bz0_sym = sym.symbols('x, psi0, Bz0')

        psi_sym = sym.sympify(psi_string)
        by_sym = sym.diff(psi_sym, x)
        jz_sym = sym.diff(by_sym, x)
        bz_sym = sym.sqrt(bz0_sym**2 - by_sym**2)
        jy_sym = sym.diff(bz_sym, x)

        psi_sym = sym.Add(psi_sym, -psi_sym.evalf(6, subs={x: -np.pi}))

        output_dict = {}
        output_dict["psi"] = sym.lambdify([x,psi0_sym], psi_sym)
        output_dict["by"] = sym.lambdify([x,psi0_sym], by_sym)
        output_dict["jz"] = sym.lambdify([x,psi0_sym], jz_sym)
        output_dict["bz"] = sym.lambdify([x,psi0_sym, bz0_sym], bz_sym)
        output_dict["jy"] = sym.lambdify([x,psi0_sym, bz0_sym], -1 * jy_sym)

        output_dict["psi_sym"] = psi_sym
        output_dict["by_sym"] = by_sym
        output_dict["jz_sym"] = jz_sym
        output_dict["bz_sym"] = bz_sym
        output_dict["jy_sym"] = -1 * jy_sym

        return output_dict

    def gen_profiles_from_by(by_string):
        x, psi0_sym, bz0_sym = sym.symbols('x, psi0, Bz0')

        by_sym = sym.sympify(by_string)

        psi_sym = sym.integrate(by_sym, x)
        psi_sym = sym.Add(psi_sym, -psi_sym.evalf(6, subs={x: -np.pi}))

        jz_sym = sym.diff(by_sym, x)
        bz_sym = sym.sqrt(bz0_sym**2 - by_sym**2)
        jy_sym = sym.diff(bz_sym, x)

        output_dict = {}
        output_dict["psi"] = sym.lambdify([x,psi0_sym], psi_sym)
        output_dict["by"] = sym.lambdify([x,psi0_sym], by_sym)
        output_dict["jz"] = sym.lambdify([x,psi0_sym], jz_sym)
        output_dict["bz"] = sym.lambdify([x,psi0_sym, bz0_sym], bz_sym)
        output_dict["jy"] = sym.lambdify([x,psi0_sym, bz0_sym], -1 * jy_sym)

        output_dict["psi_sym"] = psi_sym
        output_dict["by_sym"] = by_sym
        output_dict["jz_sym"] = jz_sym
        output_dict["bz_sym"] = bz_sym
        output_dict["jy_sym"] = -1 * jy_sym

        return output_dict

    def compare_profile_to_louriero(x_array, config, initial_config, psi_mag, bz_mag, which=['psi','by','jz','bz','jy']):

        # plt.rc('axes', titlesize=17)
        # plt.rc('axes', labelsize=13)
        # plt.rc('legend',fontsize=13)

        for f in which:
            if(f in ['bz','jy']):
                args = [psi_mag, bz_mag]
            else:
                args = [psi_mag]

            plt.figure(f)
            plt.title(f)
            plt.plot(x_array, initial_config[f](x_array, *args))
            plt.plot(x_array, config[f](x_array, *args), 'r--')
            plt.vlines(0, plt.ylim()[0], plt.ylim()[1], alpha=0.5)
            plt.xlabel("x position")
            plt.xlim([-np.pi-0.0,np.pi+0.0])
            plt.axhline(0, alpha=0.7, color='k')

            if(f == 'psi'):
                plt.legend(['loureiro','new config'])

    def calc_delprime_loureiro(k):
        return 2*(5-k**2)*(3+k**2)/(k**2*np.sqrt(4+k**2))

    def plot_paper_poem(delprime):
        a = 0.35
        poem_results = 2.44 * delprime * a
        plt.plot(delprime*a, poem_results,'k-', label="POEM")

    def set_hmhd_profiles(symm_config, make=False):
        # change the profiles in prob.f90 of HMHD

        ## divison sign must have backslash before it (not '/', but '\/')
        psii_string = str((symm_config['psi_sym'])).replace("exp", "???").replace("psi0","a").replace("x","z(k)").replace("/","\/").replace("???", "exp")
        byi_string = str((symm_config['bz_sym'])).replace("exp", "???").replace("psi0","a").replace("Bz0","qpa").replace("x","z(k)").replace("/","\/").replace("???", "exp")
        byi_string = byi_string[:-1] + " - 4.0 * prei(i,k) )" # make sure initially there is force balance, even with finite, non-const pressure

        # print("psii_string", psii_string)
        # print("byi_string", byi_string)

        sed_string_psii = f"sed -i  '/.*\!/!s/.*psii(i,k)=1.00.*/\tpsii(i,k)=1.00*{psii_string}/' ~/HMHD2D/prob.f"
        sed_string_byi = f"sed -i  '/.*\!/!s/.*byi(i,k)=1.00.*/\tbyi(i,k)=1.00*{byi_string}/' ~/HMHD2D/prob.f"
        subprocess.run(sed_string_psii, shell=True)
        subprocess.run(sed_string_byi, shell=True)

        sed_string_psii = f"sed -i  '/.*\!/!s/.*psiii(i,k)=1.00.*/\tpsiii(i,k)=1.00*{psii_string}/' ~/HMHD2D/prob.f"
        sed_string_byi = f"sed -i  '/.*\!/!s/.*byii(i,k)=1.00.*/\tbyii(i,k)=1.00*{byi_string}/' ~/HMHD2D/prob.f"
        subprocess.run(sed_string_psii, shell=True)
        subprocess.run(sed_string_byi, shell=True)

        if(make):
            subprocess.run(f"cd ~/HMHD2D; make", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    def set_hmhd_dens(den_string):
        # change the density in prob.f90 of HMHD

        ## divison sign must have backslash before it (not '/', but '\/')

        sed_string_den = f"sed -i  '/.*\!/!s/.*deni(i,k)=1.00.*/\tdeni(i,k)=1.00*{den_string}/' ~/HMHD2D/prob.f"
        subprocess.run(sed_string_den, shell=True)

        # subprocess.run(f"cd ~/HMHD2D; make", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def set_hmhd_pres(pre_string):
        # change the pressure in prob.f90 of HMHD

        ## divison sign must have backslash before it (not '/', but '\/')

        sed_string_den = f"sed -i  '/.*\!/!s/.*prei(i,k)=1.00.*/\tprei(i,k)=1.00*{pre_string}/' ~/HMHD2D/prob.f"
        subprocess.run(sed_string_den, shell=True)

        # subprocess.run(f"cd ~/HMHD2D; make", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    def set_hmhd_heatcond(perp_cond, para_cond, flag_cond):
        # change the parallel and perpendicular heat conductivities in equation.f90

        if(flag_cond!=0 and flag_cond!=1):
            raise ValueError(f"flag_cond ({flag_cond}) in HMHDslab.set_hmhd_heatcond has to be 0 or 1 (numerical)")
        sed_string_den = f"sed -i  '/.*\!/!s/#define THERMAL_CONDUCT.*/#define THERMAL_CONDUCT {flag_cond}/' ~/HMHD2D/inc.h"
        subprocess.run(sed_string_den, shell=True)

        subprocess.run(f"cd ~/HMHD2D; make", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        sed_string_den = f"sed -i  '/.*\!/!s/.*k_para1=1.00.*/    k_para1=1.00* {para_cond}/' ~/HMHD2D/equation.f"
        subprocess.run(sed_string_den, shell=True)

        sed_string_den = f"sed -i  '/.*\!/!s/.*k_perp1=1.00.*/    k_perp1=1.00* {perp_cond}/' ~/HMHD2D/equation.f"
        subprocess.run(sed_string_den, shell=True)

        # subprocess.run(f"cd ~/HMHD2D; make", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    def run_hmhd(name='run_Tearing', num_cpus=9):
        """
        Run HMHD2D from input file "iTearing"
        params:
            name -- name of the folder where simulation will be stored
            num_cpus -- number of cpus for running HMHD (used through mpirun)
        """
        subprocess.run(f"rm -rf {name}; mkdir {name}; cp iTearing {name}", shell=True)
        with change_dir(name):
            subprocess.run(f"cp ~/HMHD2D/build/globals.f90 .; cp ~/HMHD2D/build/prob.f90 .; cp ~/HMHD2D/build/equation.f90 .", shell=True)
            subprocess.run(f"mpirun -n {num_cpus} ~/HMHD2D/build/hmhd2d Tearing", shell=True)

    def animate_imshow(i):
        data = get_array_from_hdf("run_Tearing/"+f"data{i+1:04}.hdf","psi")
        im.set_array(data)
        im.set_clim(vmin=np.min(data), vmax=np.max(data))
        cb.set_clim(vmin=np.min(data), vmax=np.max(data))
        return im

    def animate_cont(frame, root_fname):

        nx=250
        nz=250
        ncont=150
        plot=True
        plot_title=None
        xlims=[-2, 2]
        ylims=[-np.pi, np.pi]

        global cont, contf, pl1, pl2, pl3, pl4, pl5, pl6

        for item in [cont, contf]:
            # print('item', item)
            for c in item.collections:
                c.remove()

        with h5py.File(root_fname+f"/data{frame+1:04}.hdf",'r') as h5:
            psi = h5['psi'][2:-2, 2:-1]
            psi *= -1.0
            x = h5["x"][2:-1]
            z = h5["z"][2:-2]

        ind = np.arange(psi.shape[1])
        ind = np.roll(ind, len(ind)//2)
        ind[:len(ind)//2] -= 1
        psi[:,:] = psi[:, ind]

        x_interp = np.linspace(np.min(x), np.max(x), nx, endpoint=True)
        z_interp = np.linspace(np.min(z), np.max(z), nz, endpoint=True)
        psi_interp_spline = RectBivariateSpline(z, x, psi)
        psi_interp_val = psi_interp_spline(z_interp, x_interp, grid=True)

        psi_main = psi_interp_val
        xm_main, zm_main = np.meshgrid(x_interp, z_interp)

        o_point = np.unravel_index((psi_main[:,:]).argmin(), psi_main[:,:].shape)
        x_point = np.unravel_index(((psi_main[:,:])**2).argmin(), psi_main[:,:].shape)

        plt.ioff()
        levels = np.linspace(psi_main[o_point], psi_main[x_point], ncont)[1:-1]
        c2 = plt.contour(xm_main, zm_main, psi_main[:,:], levels=levels, colors='black', linestyles='solid', alpha=0)
        plt.close()
        plt.ion()

        max_closed_ind = -1
        for i in range(len(c2.collections)):
            verts = c2.collections[i].get_paths()[0].vertices
            if(len(c2.collections[i].get_paths()) == 1 and np.linalg.norm(verts[-1]-verts[0]) < 1e-5):
                max_closed_ind = i
        cont_pts = [c2.collections[max_closed_ind].get_paths()[0].vertices]

        max_val = np.max(psi_main[:])
        contf = ax.contourf(xm_main, zm_main, psi_main[:,:]/max_val, levels=20, alpha=0.8)
        cont = ax.contour(xm_main, zm_main, psi_main[:,:]/max_val, levels=20, alpha=0.8, colors='black', linestyles='solid')

        if(len(cont_pts) < 1):
            island_w = 0.0
        else:
            cont_pts = np.concatenate(cont_pts)
            island_w = np.max(cont_pts[:, 1]) - np.min(cont_pts[:, 1])

            r_up = np.max(cont_pts[:, 1])
            r_down = np.min(cont_pts[:, 1])

            r_x = cont_pts[np.argmin(cont_pts[:,0]), 1]
            Asym = (r_up-r_x)/(r_x-r_down) - 1

            pl1[0].set_data([[],[]])
            pl2.set_data([[],[]])
            pl3.set_data([[],[]])
            pl4.set_data([[],[]])
            pl5[0].set_data([[],[]])
            pl6[0].set_data([[],[]])

            pl1 = ax.plot(cont_pts[:,0], cont_pts[:,1], 'r-', lw=2)
            pl2 = ax.axhline(np.min(cont_pts[:, 1]), color='red', linestyle='dashed', lw=1)
            pl3 = ax.axhline(np.max(cont_pts[:, 1]), color='red', linestyle='dashed', lw=1)
            pl4 = ax.axhline(r_x, color='k', linestyle='dashed', lw=1.5)
            pl5 = ax.plot(cont_pts[np.argmin(cont_pts[:,0]), 0],cont_pts[np.argmin(cont_pts[:,0]), 1],'rX', ms=9)
            pl6 = ax.plot(xm_main[o_point], zm_main[o_point],'ro', ms=9)

        ax.set_title(f"HMHD A_z resonant volume (width {island_w:.4f} Asym {Asym:.4f})")

    ## OLD version
    # def animate_cont(i, root_fname):
    #
    #     global cont, contf
    #
    #     with h5py.File(root_fname+f"/data{i+1:04}.hdf",'r') as h5file:
    #         data = h5file['psi'][:]
    #         x = h5file["x"][:]
    #         z = h5file["z"][:]
    #
    #     for c in contf.collections:
    #         c.remove()
    #     for c in cont.collections:
    #         c.remove()
    #
    #     X, Z = np.meshgrid(x,z)
    #     data_max = np.max(np.abs(data))
    #     contf = ax.contourf(X, Z, data/data_max, levels=num_contours_global, alpha=0.4, cmap='plasma')
    #     cont = ax.contour(X, Z, data/data_max, levels=num_contours_global, colors='black', linestyles='solid')
    #     #fig.colorbar(contf)
    #     return contf, cont

    def animate_run(root_fname="run_Tearing/", num_contours=30):

        global fig, ax, cont, contf, pl1, pl2, pl3, pl4, pl5, pl6, num_contours_global

        fig, ax = plt.subplots()
        ax.set_title("Plot of $\psi(x,y)$")
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        contf = ax.contourf([[1,2],[2,4]], alpha=0)
        cont = ax.contour([[1,2],[5,2]], alpha=0)
        pl1 = ax.plot([1], [1], alpha=0)
        pl2 = ax.axhline(0, alpha=0)
        pl3 =  ax.axhline(0, alpha=0)
        pl4 =  ax.axhline(0, alpha=0)
        pl5 = ax.plot([1],[1], alpha=0)
        pl6 = ax.plot([1],[1], alpha=0)
        num_contours_global = num_contours

        fps = 8
        frames = len(glob.glob(root_fname+'/data*.hdf'))
        ani = anim.FuncAnimation(fig, lambda x: HMHDslab.animate_cont(x, root_fname), frames=frames, interval=1000/fps, blit=False, repeat=False);
        plt.close()

        return HTML(ani.to_jshtml())

    def roll_scalar(scalar):
        ind = np.arange(scalar.shape[1])
        ind = np.roll(ind, len(ind)//2)
        ind[:len(ind)//2] -= 1
        return scalar[:, ind]

    def plot_scalar(fname, field_name, nx=250, nz=250, ncont=70, xlims=None, ylims=[-np.pi, np.pi], fig=None):

        if(fig is None):
            fig, ax = plt.subplots()
        else:
            ax = fig.gca()

        with h5py.File(fname,'r') as h5:
            x = h5["x"][2:-1]
            z = h5["z"][2:-2]
            field = HMHDslab.roll_scalar(h5[field_name][2:-2, 2:-1])
            psi = HMHDslab.roll_scalar(h5['psi'][2:-2, 2:-1]) * -1.0

        if(xlims is None):
            xlims = [np.min(x), np.max(x)]

        x_interp = np.linspace(np.min(x), np.max(x), nx, endpoint=True)
        z_interp = np.linspace(np.min(z), np.max(z), nz, endpoint=True)
        xm_main, zm_main = np.meshgrid(x_interp, z_interp)

        field_interp_spline = RectBivariateSpline(z, x, field)
        field_interp_val = field_interp_spline(z_interp, x_interp, grid=True)

        contf = ax.contourf(xm_main, zm_main, field_interp_val[:,:], levels=50, alpha=1)
        cont = ax.contour(xm_main, zm_main, field_interp_val[:,:], levels=30, alpha=0, colors='black', linestyles='solid')
        cbar = fig.colorbar(contf, format='{x:.3f}')

        psi_interp_spline = RectBivariateSpline(z, x, psi)
        psi_interp_val = psi_interp_spline(z_interp, x_interp, grid=True)

        field_interp_spline = RectBivariateSpline(z, x, field)
        field_interp_val = field_interp_spline(z_interp, x_interp, grid=True)

        psi_main = psi_interp_val

        o_point = np.unravel_index((psi_main[:,:]).argmin(), psi_main[:,:].shape)
        x_point = np.unravel_index(((psi_main[:,:])**2).argmin(), psi_main[:,:].shape)

        fig2, ax2 = plt.subplots()
        levels = np.linspace(psi_main[o_point], psi_main[x_point], ncont)[1:-1]
        c2 = ax2.contour(xm_main, zm_main, psi_main[:,:], levels=levels, colors='black', linestyles='solid', alpha=0)
        plt.close(fig2)

        max_closed_ind = -1
        for i in range(len(c2.collections)):
            verts = c2.collections[i].get_paths()[0].vertices
            if(len(c2.collections[i].get_paths()) == 1 and np.linalg.norm(verts[-1]-verts[0]) < 1e-5):
                max_closed_ind = i
        cont_pts = [c2.collections[max_closed_ind].get_paths()[0].vertices]

        if(len(cont_pts) < 1):
            island_w = 0.0
        else:
            cont_pts = np.concatenate(cont_pts)
            island_w = np.max(cont_pts[:, 1]) - np.min(cont_pts[:, 1])
            r_up = np.max(cont_pts[:, 1])
            r_down = np.min(cont_pts[:, 1])

            r_x = cont_pts[np.argmin(cont_pts[:,0]), 1]
            Asym = (r_up-r_x)/(r_x-r_down+1e-12) - 1

            pl1 = ax.plot(cont_pts[:,0], cont_pts[:,1], 'r-', lw=1.2)
            pl2 = ax.axhline(np.min(cont_pts[:, 1]), color='red', linestyle='dashed', lw=0.8)
            pl3 = ax.axhline(np.max(cont_pts[:, 1]), color='red', linestyle='dashed', lw=0.8)
            pl4 = ax.axhline(r_x, color='k', linestyle='dashed', lw=0.8)
            pl5 = ax.plot(cont_pts[np.argmin(cont_pts[:,0]), 0],cont_pts[np.argmin(cont_pts[:,0]), 1],'rX', ms=9)
            pl6 = ax.plot(xm_main[o_point], zm_main[o_point],'ro', ms=9)

        ax.set_title(r"HMHD $\bf{ " + '{:}'.format(field_name) + "}$ "+ f"{fname[:-4]}")
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        # ax.text(0.24*np.max(xm_main), -3, fname[:-4], fontsize=11, fontweight='normal', bbox=dict(facecolor='white', alpha=0.8))

        return xm_main, zm_main, field_interp_val, cont_pts

    def plot_scalar_3d(args):

        from matplotlib.widgets import Button

        matplotlib.rcParams['keymap.back'].remove('left')
        matplotlib.rcParams['keymap.forward'].remove('right')

        print("Use arrow keys to navigate...\ntop/down - change field\nleft/right - change frame")

        if(len(args)<2):
            raise ValueError("Invalid input...")

        root_fname = args[1] + '/data'
        fields = ['bx', 'by', 'bz', 'den', 'ex', 'ey', 'ez', 'jx', 'jy', 'jz', 'pre', 'psi', 'vx', 'vy', 'vz']

        y_pos = 0.02
        y_height = 0.06

        fig = plt.figure(figsize=(9,8))

        callback = Index(root_fname, fields, fig)

        axprev = fig.add_axes([0.58, y_pos, 0.18, y_height])
        axnext = fig.add_axes([0.78, y_pos, 0.18, y_height])
        bnext = Button(axnext, 'Next frame')
        bnext.on_clicked(callback.next)
        bprev = Button(axprev, 'Prev frame')
        bprev.on_clicked(callback.prev)

        axprev2 = fig.add_axes([0.05, y_pos, 0.18, y_height])
        axnext2 = fig.add_axes([0.25, y_pos, 0.18, y_height])
        bnext2 = Button(axnext2, 'Next field')
        bnext2.on_clicked(callback.next_field)
        bprev2 = Button(axprev2, 'Prev field')
        bprev2.on_clicked(callback.prev_field)

        def on_press(event):
            sys.stdout.flush()
            if event.key == 'left':
                callback.prev(None)
                fig.canvas.draw()
            elif event.key == 'right':
                callback.next(None)
                fig.canvas.draw()
            elif event.key == 'up':
                callback.next_field(None)
                fig.canvas.draw()
            elif event.key == 'down':
                callback.prev_field(None)
                fig.canvas.draw()
        fig.canvas.mpl_connect('key_press_event', on_press)

        plt.tight_layout()
        plt.show()


    def animate_field_helper(frame, root_fname, field_name, nx=250, nz=250, ncont=70, xlims=[-2, 2], ylims=[-np.pi, np.pi]):

        if(frame==0):
            return

        global cont, contf, pl1, pl2, pl3, pl4, pl5, pl6, cbar

        for item in [cont, contf]:
            for c in item.collections:
                c.remove()

        fname = sorted(glob.glob(root_fname+"/data*.hdf"))[frame]
        with h5py.File(fname,'r') as h5:
            field = HMHDslab.roll_scalar(h5[field_name][2:-2, 2:-1])
            psi = HMHDslab.roll_scalar(h5['psi'][2:-2, 2:-1]) * -1.0
            x = h5["x"][2:-1]
            z = h5["z"][2:-2]

        x_interp = np.linspace(np.min(x), np.max(x), nx, endpoint=True)
        z_interp = np.linspace(np.min(z), np.max(z), nz, endpoint=True)
        xm_main, zm_main = np.meshgrid(x_interp, z_interp)

        psi_interp_spline = RectBivariateSpline(z, x, psi)
        psi_interp_val = psi_interp_spline(z_interp, x_interp, grid=True)

        field_interp_spline = RectBivariateSpline(z, x, field)
        field_interp_val = field_interp_spline(z_interp, x_interp, grid=True)

        psi_main = psi_interp_val

        o_point = np.unravel_index((psi_main[:,:]).argmin(), psi_main[:,:].shape)
        x_point = np.unravel_index(((psi_main[:,:])**2).argmin(), psi_main[:,:].shape)

        plt.ioff()
        levels = np.linspace(psi_main[o_point], psi_main[x_point], ncont)[1:-1]
        c2 = plt.contour(xm_main, zm_main, psi_main[:,:], levels=levels, colors='black', linestyles='solid', alpha=0)
        plt.close()
        plt.ion()

        max_closed_ind = -1
        for i in range(len(c2.collections)):
            verts = c2.collections[i].get_paths()[0].vertices
            if(len(c2.collections[i].get_paths()) == 1 and np.linalg.norm(verts[-1]-verts[0]) < 1e-5):
                max_closed_ind = i
        cont_pts = [c2.collections[max_closed_ind].get_paths()[0].vertices]

        contf = ax.contourf(xm_main, zm_main, field_interp_val[:,:], levels=50, alpha=1)
        cont = ax.contour(xm_main, zm_main, field_interp_val[:,:], levels=30, alpha=0, colors='black', linestyles='solid')

        # for c in contf.collections:
        #     c.set_edgecolor("face")

        fig.colorbar(contf, cbar.ax, format='{x:.3f}')

        if(len(cont_pts) < 1):
            island_w = 0.0
        else:
            cont_pts = np.concatenate(cont_pts)
            island_w = np.max(cont_pts[:, 1]) - np.min(cont_pts[:, 1])

            r_up = np.max(cont_pts[:, 1])
            r_down = np.min(cont_pts[:, 1])

            r_x = cont_pts[np.argmin(cont_pts[:,0]), 1]
            Asym = (r_up-r_x)/(r_x-r_down+1e-12) - 1

            pl1[0].set_data([[],[]])
            pl2.set_data([[],[]])
            pl3.set_data([[],[]])
            pl4.set_data([[],[]])
            pl5[0].set_data([[],[]])
            pl6[0].set_data([[],[]])

            pl1 = ax.plot(cont_pts[:,0], cont_pts[:,1], 'r-', lw=1.2)
            # pl2 = ax.axhline(np.min(cont_pts[:, 1]), color='red', linestyle='dashed', lw=0.8)
            # pl3 = ax.axhline(np.max(cont_pts[:, 1]), color='red', linestyle='dashed', lw=0.8)
            # pl4 = ax.axhline(r_x, color='k', linestyle='dashed', lw=0.8)
            pl5 = ax.plot(cont_pts[np.argmin(cont_pts[:,0]), 0],cont_pts[np.argmin(cont_pts[:,0]), 1],'rX', ms=9)
            pl6 = ax.plot(xm_main[o_point], zm_main[o_point],'ro', ms=9)

        ax.set_title(f"HMHD {field_name} frame {frame}")
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

    def animate_field(root_fname="run_Tearing/", field_name="psi", fps=8, nx=250, nz=250, ncont=70, xlims=[-2, 2], ylims=[-np.pi, np.pi]):
        # animates the evolution of a field, along with the island outline

        global fig, ax
        fig, ax = plt.subplots()

        def init():

            fname = sorted(glob.glob(root_fname+"/data*.hdf"))[-1]

            global cont, contf, pl1, pl2, pl3, pl4, pl5, pl6, cbar

            with h5py.File(fname,'r') as h5:
                x = h5["x"][2:-1]
                z = h5["z"][2:-2]
                field = HMHDslab.roll_scalar(h5[field_name][2:-2, 2:-1])
                psi = HMHDslab.roll_scalar(h5['psi'][2:-2, 2:-1]) * -1.0

            x_interp = np.linspace(np.min(x), np.max(x), nx, endpoint=True)
            z_interp = np.linspace(np.min(z), np.max(z), nz, endpoint=True)
            xm_main, zm_main = np.meshgrid(x_interp, z_interp)

            field_interp_spline = RectBivariateSpline(z, x, field)
            field_interp_val = field_interp_spline(z_interp, x_interp, grid=True)

            contf = ax.contourf(xm_main, zm_main, field_interp_val[:,:], levels=50, alpha=1)
            cont = ax.contour(xm_main, zm_main, field_interp_val[:,:], levels=30, alpha=0, colors='black', linestyles='solid')
            cbar = fig.colorbar(contf, format='{x:.3f}')

            psi_interp_spline = RectBivariateSpline(z, x, psi)
            psi_interp_val = psi_interp_spline(z_interp, x_interp, grid=True)

            field_interp_spline = RectBivariateSpline(z, x, field)
            field_interp_val = field_interp_spline(z_interp, x_interp, grid=True)

            psi_main = psi_interp_val

            o_point = np.unravel_index((psi_main[:,:]).argmin(), psi_main[:,:].shape)
            x_point = np.unravel_index(((psi_main[:,:])**2).argmin(), psi_main[:,:].shape)

            plt.ioff()
            levels = np.linspace(psi_main[o_point], psi_main[x_point], ncont)[1:-1]
            c2 = plt.contour(xm_main, zm_main, psi_main[:,:], levels=levels, colors='black', linestyles='solid', alpha=0)
            plt.close()
            plt.ion()

            max_closed_ind = -1
            for i in range(len(c2.collections)):
                verts = c2.collections[i].get_paths()[0].vertices
                if(len(c2.collections[i].get_paths()) == 1 and np.linalg.norm(verts[-1]-verts[0]) < 1e-5):
                    max_closed_ind = i
            cont_pts = [c2.collections[max_closed_ind].get_paths()[0].vertices]

            if(len(cont_pts) < 1):
                island_w = 0.0
            else:
                cont_pts = np.concatenate(cont_pts)
                island_w = np.max(cont_pts[:, 1]) - np.min(cont_pts[:, 1])
                r_up = np.max(cont_pts[:, 1])
                r_down = np.min(cont_pts[:, 1])

                r_x = cont_pts[np.argmin(cont_pts[:,0]), 1]
                Asym = (r_up-r_x)/(r_x-r_down+1e-12) - 1

                pl1 = ax.plot(cont_pts[:,0], cont_pts[:,1], 'r-', lw=1.2)
                pl2 = ax.axhline(np.min(cont_pts[:, 1]), color='red', linestyle='dashed', lw=0.8)
                pl3 = ax.axhline(np.max(cont_pts[:, 1]), color='red', linestyle='dashed', lw=0.8)
                pl4 = ax.axhline(r_x, color='k', linestyle='dashed', lw=0.8)
                pl5 = ax.plot(cont_pts[np.argmin(cont_pts[:,0]), 0],cont_pts[np.argmin(cont_pts[:,0]), 1],'rX', ms=9)
                pl6 = ax.plot(xm_main[o_point], zm_main[o_point],'ro', ms=9)

            ax.set_title(f"HMHD {field_name} (file: {fname})")
            ax.set_xlabel("y")
            ax.set_ylabel("x")
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)

        frames = len(glob.glob(root_fname+'/data*.hdf'))
        anim_func = lambda x: HMHDslab.animate_field_helper(x, root_fname, field_name, nx, nz, ncont, xlims, ylims)
        ani = anim.FuncAnimation(fig, anim_func, frames=frames, interval=1000/fps, blit=False, repeat=False, init_func=init);

        plt.tight_layout(pad=2)
        plt.close()

        # ani.save(f'hmhd_anim_{field_name}.mp4', writer='ffmpeg')
        return HTML(ani.to_jshtml())


    def get_width_As_HMHD_old(outfile, flag_plot=True):

        if(flag_plot):
            plt.figure()

        with h5py.File(outfile,'r') as h5:
            psi = h5["psi"][:]
            x = h5["x"][:]
            z = h5["z"][:]

        mask = np.ones_like(psi)
        mask[:,:psi.shape[1]//2] = 0
        maxindl = np.unravel_index((psi*mask).argmax(), psi.shape)
        maxindr = np.unravel_index((psi*(1-mask)).argmax(), psi.shape)

        ind_maxrow = np.argmax(np.sum(psi**1, axis=1))
        ind_xpt_col = np.argmin(psi[ind_maxrow])
        xm, zm = np.meshgrid(x, z)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = plt.contour(xm, zm, psi-psi[ind_maxrow,ind_xpt_col],levels=[0.0], colors='black')

        if(len(c.collections[0].get_paths()) > 2):
            max_len_ind = np.argmax(np.array([ len(c.collections[0].get_paths()[i]) for i in range(len(c.collections[0].get_paths())) ]))

            cont1 = c.collections[0].get_paths()[max_len_ind].vertices
            cont2 = c.collections[0].get_paths()[max_len_ind].vertices
        elif(len(c.collections[0].get_paths()) < 2):
            if(flag_plot):
                plt.contourf(xm, zm, psi, levels=30, cmap='viridis')
                plt.xlabel("z")
                plt.ylabel("x")
                plt.title(f"Contours of psi ({outfile})")
            return 0, 0
        else:
            cont1 = np.array(c.collections[0].get_paths()[0].vertices)
            cont2 = np.array(c.collections[0].get_paths()[1].vertices)

        r_top = max(np.max(cont1[:,1]), np.max(cont2[:,1]))
        r_bottom = min(np.min(cont1[:,1]), np.min(cont2[:,1]))

        if(np.abs(cont1[0,0]-cont1[-1,0]) < 1e-2):
            # r_x = (cont1[np.argmax(cont1[:,0]),1] + cont2[np.argmin(cont2[:,0]),1]) * 0.5
            r_x = (np.max(cont1[:,1]) + np.min(cont2[:,1])) * 0.5
        elif(np.abs(cont1[0,1]-cont1[-1,1]) < 1e-2):
            r_x = (np.max(cont1[:,1]) + np.min(cont2[:,1])) * 0.5
        else:
            # raise ValueError("Couldn't get contours of the resonant island!")
            r_x = (cont1[np.argmax(cont1[:,0]),1] + cont2[np.argmin(cont2[:,0]),1]) * 0.5

        island_w = r_top - r_bottom
        island_As = (r_top - r_x) / (r_x - r_bottom) - 1

        if(flag_plot):
            plt.contourf(xm, zm, psi, levels=30, cmap='viridis')
            plt.xlabel("z")
            plt.ylabel("x")
            plt.title(f"Contours of psi ({outfile})", fontsize=16)
            plt.axhline(r_x, color='black', ls='dashed')
            plt.axhline(r_top, color='red',ls='dashed')
            plt.axhline(r_bottom, color='red',ls='dashed')
            plt.text(1, 2.8, f"Resonant island\nw   {island_w:.4f}  \nA_s   {island_As:.4f}", bbox=dict(facecolor='white',edgecolor='black',alpha=0.6), fontsize=16, verticalalignment='top')

            # print(f"HMHD Island width {island_w:.4f}  As {island_As:.4f}")

        else:
            plt.close()

        return island_w, island_As


    def get_width_As_HMHD(outfile, nx=600, nz=600, ncont=600, plot=True, plot_title=None, xlims=None, ylims=[0, 2*np.pi]):

        with h5py.File(outfile, 'r') as h5:
            psi = h5['psi'][2:-2, 2:-1]
            psi *= -1.0
            x = h5["x"][2:-1]
            z = h5["z"][2:-2]

        if(xlims is None):
            xlims = [np.min(x), np.max(x)]
            xlims = [0, 2*np.pi]

        ind = np.arange(psi.shape[1])
        ind = np.roll(ind, len(ind)//2)
        ind[:len(ind)//2] -= 1
        psi[:,:] = psi[:, ind]

        x_interp = np.linspace(np.min(x), np.max(x), nx, endpoint=True)
        z_interp = np.linspace(np.min(z), np.max(z), nz, endpoint=True)
        psi_interp_spline = RectBivariateSpline(z, x, psi)
        psi_interp_val = psi_interp_spline(z_interp, x_interp, grid=True)

        psi_main = psi_interp_val
        xm_main, zm_main = np.meshgrid(x_interp, z_interp)
        xl = np.max(xm_main)

        o_point = np.unravel_index((psi_main[:,:]).argmin(), psi_main[:,:].shape)
        x_point = np.unravel_index(((psi_main[:,:])**2).argmin(), psi_main[:,:].shape)

        plt.ioff()
        plt.figure()
        levels = np.linspace(psi_main[o_point], psi_main[x_point], ncont)[1:-1]
        c2 = plt.contour(xm_main, zm_main, psi_main[:,:], levels=levels, colors='black', linestyles='solid', alpha=0)
        plt.close()
        plt.ion()

        max_closed_ind = -1
        for i in range(len(c2.collections)):
            verts = c2.collections[i].get_paths()[0].vertices
            if(len(c2.collections[i].get_paths()) == 1 and np.linalg.norm(verts[-1]-verts[0]) < 1e-5):
                max_closed_ind = i
        cont_pts = [c2.collections[max_closed_ind].get_paths()[0].vertices]

        if(plot):
            plt.figure()
            # plt.contourf(xm_main, zm_main, psi_main[:,:], levels=20, alpha=0.9)
            # plt.colorbar()
            xm_main *= np.pi / xl
            xm_main += np.pi
            plt.contour(xm_main, zm_main+np.pi, psi_main[:,:], levels=40, alpha=0.9, colors='black', linestyles='solid')

        if(len(cont_pts) < 1):
            island_w = 0.0
        else:
            cont_pts = np.concatenate(cont_pts)
            island_w = np.max(cont_pts[:, 1]) - np.min(cont_pts[:, 1])

            r_up = np.max(cont_pts[:, 1])
            r_down = np.min(cont_pts[:, 1])

            # r_x = zm[x_point]
            r_x = cont_pts[np.argmin(cont_pts[:,0]), 1]
            Asym = (r_up-r_x)/(r_x-r_down) - 1

            # print(f"r_up,down,x -- {r_up} {r_down} {r_x}")

            if(plot):
                plt.plot(cont_pts[:,0]*np.pi/xl + np.pi, cont_pts[:,1] + np.pi, 'r-', lw=2.5)
                plt.axhline(np.min(cont_pts[:, 1]) + np.pi, color='red', linestyle='dashed', lw=1.8)
                plt.axhline(np.max(cont_pts[:, 1]) + np.pi, color='red', linestyle='dashed', lw=1.8)
                plt.axhline(r_x + np.pi, color='red', linestyle='dashed', lw=2.3)
                plt.plot(cont_pts[np.argmin(cont_pts[:,0]), 0]*np.pi/xl + np.pi, cont_pts[np.argmin(cont_pts[:,0]), 1] + np.pi,'rX', ms=11)
                plt.plot(xm_main[o_point],zm_main[o_point] + np.pi,'ro', ms=10)
                plt.plot(xm_main[0], zm_main[0] + np.pi, 'k-', lw=1)
                plt.plot(xm_main[-1], zm_main[-1] + np.pi, 'k-', lw=1)

        if(plot):
            if(plot_title is None):
                plot_title = f"HMHD island (width {island_w:.3f} Asym {Asym:.3f})"
                # plot_title = f"HMHD island (width {island_w:.4f})"
            plt.title(plot_title, fontsize=22)
            plt.gcf().canvas.manager.set_window_title(plot_title + f" w={island_w:.3f}")
            plt.ylim(ylims)
            plt.xlim(xlims)
            plt.xlabel("$\\theta$", fontsize=18)
            plt.ylabel("R", fontsize=18)
            plt.tight_layout()

            plt.text(4.32, 0.18, outfile[:-4], fontsize=11, fontweight='normal', bbox=dict(facecolor='white', alpha=0.8))

        # print(f"HMHD Island width {island_w:.8f}")

        return island_w, Asym

    def get_width_As_rpmx_HMHD(outfile, nx=600, nz=600, ncont=600, plot=True, plot_title=None, xlims=[-2, 2], ylims=[-np.pi, np.pi]):

        with h5py.File(outfile, 'r') as h5:
            psi = h5['psi'][2:-2, 2:-1]
            psi *= -1.0
            x = h5["x"][2:-1]
            z = h5["z"][2:-2]

        ind = np.arange(psi.shape[1])
        ind = np.roll(ind, len(ind)//2)
        ind[:len(ind)//2] -= 1
        psi[:,:] = psi[:, ind]

        x_interp = np.linspace(np.min(x), np.max(x), nx, endpoint=True)
        z_interp = np.linspace(np.min(z), np.max(z), nz, endpoint=True)
        psi_interp_spline = RectBivariateSpline(z, x, psi)
        psi_interp_val = psi_interp_spline(z_interp, x_interp, grid=True)

        psi_main = psi_interp_val
        xm_main, zm_main = np.meshgrid(x_interp, z_interp)

        o_point = np.unravel_index((psi_main[:,:]).argmin(), psi_main[:,:].shape)
        x_point = np.unravel_index(((psi_main[:,:])**2).argmin(), psi_main[:,:].shape)

        plt.ioff()
        plt.figure()
        levels = np.linspace(psi_main[o_point], psi_main[x_point], ncont)[1:-1]
        c2 = plt.contour(xm_main, zm_main, psi_main[:,:], levels=levels, colors='black', linestyles='solid', alpha=0)
        plt.close()
        plt.ion()

        max_closed_ind = -1
        for i in range(len(c2.collections)):
            verts = c2.collections[i].get_paths()[0].vertices
            if(len(c2.collections[i].get_paths()) == 1 and np.linalg.norm(verts[-1]-verts[0]) < 1e-5):
                max_closed_ind = i
        cont_pts = [c2.collections[max_closed_ind].get_paths()[0].vertices]

        if(plot):
            plt.figure()
            plt.contourf(xm_main, zm_main, psi_main[:,:], levels=20, alpha=0.9)
            # plt.colorbar()
            plt.contour(xm_main, zm_main, psi_main[:,:], levels=20, alpha=0.5, colors='black', linestyles='solid')

        if(len(cont_pts) < 1):
            island_w = 0.0
        else:
            cont_pts = np.concatenate(cont_pts)
            island_w = np.max(cont_pts[:, 1]) - np.min(cont_pts[:, 1])

            r_up = np.max(cont_pts[:, 1])
            r_down = np.min(cont_pts[:, 1])

            # r_x = zm[x_point]
            r_x = cont_pts[np.argmin(cont_pts[:,0]), 1]
            Asym = (r_up-r_x)/(r_x-r_down) - 1

            # print(f"r_up,down,x -- {r_up} {r_down} {r_x}")

            if(plot):
                plt.plot(cont_pts[:,0], cont_pts[:,1], 'r-', lw=2)
                plt.axhline(np.min(cont_pts[:, 1]), color='red', linestyle='dashed', lw=1)
                plt.axhline(np.max(cont_pts[:, 1]), color='red', linestyle='dashed', lw=1)
                plt.axhline(r_x, color='k', linestyle='dashed', lw=1.5)
                plt.plot(cont_pts[np.argmin(cont_pts[:,0]), 0],cont_pts[np.argmin(cont_pts[:,0]), 1],'rX', ms=9)
                plt.plot(xm_main[o_point],zm_main[o_point],'ro', ms=9)
                plt.plot(xm_main[0], zm_main[0], 'k-', lw=1)
                plt.plot(xm_main[-1], zm_main[-1], 'k-', lw=1)

        if(plot):
            if(plot_title is None):
                plot_title = f"HMHD A_z resonant volume (width {island_w:.4f} Asym {Asym:.4f})"
            plt.title(plot_title)
            # plt.gcf().canvas.manager.set_window_title(plot_title + f" w={island_w:.3f}")
            plt.ylim(ylims)
            plt.xlim(xlims)
            plt.tight_layout()

        # print(f"HMHD Island width {island_w:.8f}")

        return island_w, Asym, r_up, r_down, r_x


    def get_width_run(root_fname="run_Tearing"):
        fnames_h5files = sorted(glob.glob(root_fname+"/data*.hdf"))
        w_sat_vals = []
        As_sat_vals = []

        for f in tqdm(range(len(fnames_h5files))):
            w_curr, As_curr = HMHDslab.get_width_As_HMHD(fnames_h5files[f], nx=260, nz=260, ncont=150, plot=False)
            w_sat_vals.append(w_curr)
            As_sat_vals.append(As_curr)
        w_sat_vals[0] = 0 # island at time 0 is 0
        return w_sat_vals, As_sat_vals

    def plot_w_vs_time(root_fname):
        widths, A_s = HMHDslab.get_width_run(root_fname)
        plt.rcParams['figure.figsize'] = [13, 4]
        plt.figure()
        plt.plot(widths, 'd--')
        plt.xlabel("iteration")
        plt.ylabel("Island width")
        return np.array(widths), np.array(A_s)

    def plot_w_As_vs_time(root_fname):
        widths, A_s = HMHDslab.get_width_run(root_fname)
        A_s = [i if abs(i)<20 else 0 for i in A_s]
        plt.rcParams['figure.figsize'] = [13, 4]
        plt.figure()
        plt.plot(widths, 'd--', label='width')
        plt.plot(A_s, 'gx--', label='A_s')
        plt.axhline(0, color='black', ls='--')
        plt.xlabel("iteration")
        plt.ylabel("Island width / assymmetry")
        plt.legend()
        return np.array(widths), np.array(A_s)

    def find_psi_perturbed(k, psi0_fun, by_fun, d2by_fun, numpts_xmesh=32, flag_BCs=0, domain_wall_halfpos=np.pi):
        """Calculates the perturbed flux function for a classical tearing mode in slab geometry

        Args:
            k (double): wavenumber (2*np.pi/L)
            psi0_fun (func): initial flux function
            by_fun (func): initial magnetic field in theta
            d2by_fun (func): deriviative of initial theta magnetic field
            numpts_xmesh (int, optional): number of x mesh points. Defaults to 32.
            flag_BCs (int, optional): what boundary condition to use for the perturbed flux function. Defaults to 0.

        Returns:
            sol_left(dict): solution in the left side of domain
            sol_right(dict): solution in the right side of domain 
        """

        # have to solve left and right regions
        # for loureiro the two are symmetric
        # but not in general

        if(domain_wall_halfpos is None):
            # need to compute the domain wall, which is where by=0
            # assume symmetry in x (only look at the right side x>0)
            x_temp = np.linspace(1e-1, np.pi, 400)
            y_temp = by_fun(x_temp)
            domain_wall_halfpos = x_temp[np.argmin(y_temp**2)-1]
        
        def f(x):
            return k**2 + d2by_fun(x) / by_fun(x)

        def linsys_fun(x, y):
            return np.vstack((y[1], f(x) * y[0]))

        if(flag_BCs == 0):
            # psi is 1 at x=0, and 0 at x=pi (like wall BC??)
            def bc_right(ya, yb):
                return np.array([ya[0] - 1, yb[0]])
            def bc_left(ya, yb):
                return np.array([ya[0], yb[0]-1])
        elif(flag_BCs == 1):
            # psi is 1 at x=0, and 0 at x=pi (like wall BC??)
            def bc_right(ya, yb):
                return np.array([yb[1], ya[0]-1])
            def bc_left(ya, yb):
                return np.array([ya[1], yb[0]-1])
        else:
            return None



        x_init = np.linspace(1e-10, domain_wall_halfpos, numpts_xmesh)
        y_init = np.zeros((2, x_init.shape[0]))
        y_init[0] = psi0_fun(x_init)
        y_init[1] = by_fun(x_init)

        sol_right = integrate.solve_bvp(linsys_fun, bc_right, x_init, y_init,
                  p=None, S=None, fun_jac=None, bc_jac=None, tol=0.001, max_nodes=1000, verbose=0, bc_tol=None)

        x_init = x_init[::-1] * -1
        y_init[0] = psi0_fun(x_init)
        y_init[1] = by_fun(x_init)

        sol_left = integrate.solve_bvp(linsys_fun, bc_left, x_init, y_init,
                  p=None, S=None, fun_jac=None, bc_jac=None, tol=0.001, max_nodes=1000, verbose=0, bc_tol=None)

        if(sol_left.status!=0 or sol_right.status!=0):
            print("Error in find_psi_perturbed():")
            print(f"\tsol_left.message: {sol_left.message} (status {sol_left.status})")
            print(f"\tsol_right.message: {sol_right.message} (status {sol_right.status})")

        return sol_left, sol_right

    def gen_profiles_for_delprime(psi_string):
        x, psi0_sym = sym.symbols('x, psi0')

        psi_sym = sym.sympify(psi_string)
        by_sym = sym.diff(psi_sym, x)
        d2by_sym = sym.diff(sym.diff(by_sym, x), x)

        output_dict = {}
        output_dict["psi"] = sym.lambdify([x, psi0_sym], psi_sym)
        output_dict["by"] = sym.lambdify([x, psi0_sym], by_sym)
        output_dict["d2by"] = sym.lambdify([x, psi0_sym], d2by_sym)

        output_dict["psi_sym"] = psi_sym
        output_dict["by_sym"] = by_sym
        output_dict["d2by_sym"] = d2by_sym

        return output_dict

    def eval_delprime_sym(psi0_string, psi0_mag, x=None, k=None, plot=False, figsize=(10, 5), numpts_xmesh=128, domain_wall_halfpos=np.pi):

        a = 0.35

        if(x is not None):
            k = 2 * np.pi / x
        else:
            if(k is None):
                raise ValueError("Have to set k or x in eval_delprime")

        sym_config = HMHDslab.gen_profiles_for_delprime(psi0_string)
        psi0_fun = lambda x: sym_config['psi'](x, psi0_mag)
        by_fun = lambda x: sym_config['by'](x, psi0_mag)
        d2by_fun = lambda x: sym_config['d2by'](x, psi0_mag)
    
        sol_left, sol_right = HMHDslab.find_psi_perturbed(k, psi0_fun, by_fun, d2by_fun, numpts_xmesh, 0, domain_wall_halfpos)

        delprime = (sol_right.y[1,0] - sol_left.y[1,-1]) / sol_right.y[0,0]
        sigprime = (sol_right.y[1,0] + sol_left.y[1,-1]) / sol_right.y[0,0]

        if(plot):
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            fig.set_size_inches(figsize)

            # x_fact = 1/k
            x_fact = 1

            ax1.plot(sol_left.x * x_fact, sol_left.y[0])
            ax1.plot(sol_right.x * x_fact, sol_right.y[0])
            ax1.set_title("Perturbed flux function $\Psi_{p}$" + f"  ($\Delta' a$={delprime*a:.3f})")
            ax1.axhline(0, color='k', alpha=0.5)
            ax1.axvline(0, color='k', alpha=0.5)
            ax1.set_xlim(-np.pi, np.pi)
                
            ax2.plot(sol_left.x * x_fact, sol_left.y[1])
            ax2.plot(sol_right.x * x_fact, sol_right.y[1])
            ax2.set_title("Derivative of pert flux function $d\Psi_{p}/dr$")
            ax2.axhline(0, color='k', alpha=0.5)
            ax2.axvline(0, color='k', alpha=0.5)
            ax2.set_xlabel("Radial coordinate r")
            ax2.set_xlim(-np.pi, np.pi)

        return delprime, sigprime

    def gen_init_config_loureiro():

        # initial symmetric equilibrium (from Loureiro)
        def psi0(x, psi0_mag):
            # initial flux func
            return psi0_mag / np.cosh(x)**2 - psi0_mag / np.cosh(-np.pi)**2
        def by0(x, psi0_mag):
            # poloidal field
            return -2 * psi0_mag * np.sinh(x) / np.cosh(x)**3
        def jz0(x, psi0_mag):
            # toroidal current
            return 4 * psi0_mag * np.sinh(x)**2 / np.cosh(x)**4 - 2 * psi0_mag / np.cosh(x)**4
        def bz0(x, psi0_mag, bz0_mag):
            # toroidal field
            return np.sqrt(bz0_mag**2 - by0(x, psi0_mag)**2)
        def jy0(x, psi0_mag, bz0_mag):
            # poloidal current
            return -(4 * by0(x, psi0_mag)**2 * np.tanh(x) - 4 * by0(x,psi0_mag)**2 / np.sinh(2*x)) / (2*bz0(x, psi0_mag, bz0_mag))

        initial_config = {'psi':psi0, 'by':by0, 'jz':jz0, 'bz':bz0, 'jy':jy0}
        return initial_config

    def run_hmhd_case_pres_cond(inputs, run=True):

        inputs.config = HMHDslab.gen_profiles_from_psi(inputs.psi_profile)

        HMHDslab.set_hmhd_dens(inputs.dens_profile)
        HMHDslab.set_hmhd_pres(inputs.pres_profile)
        HMHDslab.set_hmhd_profiles(inputs.config)
        HMHDslab.set_hmhd_heatcond(inputs.heatcond_perp, inputs.heatcond_para, inputs.heatcond_flag)
        subprocess.run(f"cd ~/HMHD2D/build; make", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        HMHDslab.set_inputfile_var('Tearing', "tmax", inputs.tmax)
        HMHDslab.set_inputfile_var('Tearing', "tpltxint", inputs.tpltxint)
        HMHDslab.set_inputfile_var('Tearing', "xl", inputs.xl)
        HMHDslab.set_inputfile_var('Tearing', "dt", inputs.dt) # d is 2e-3
        HMHDslab.set_inputfile_var('Tearing', "eta", inputs.eta)
        HMHDslab.set_inputfile_var('Tearing', "bs_curr_const", inputs.bs_curr_const)

        if(run):
            HMHDslab.run_hmhd(inputs.root_fname)

            print("".join(['-']*50))
            print("--->> HMHD DONE <<---")

            inputs.files = sorted(glob.glob(inputs.root_fname + '/data*.hdf'))
            inputs.num_files = len(inputs.files)


    def whats_the_convention_again():
        print("""
            Goldston(SPEC) |  HMHD
                  x        |   z
                  z        |   y
                  y        |   x
        """)


class interp2d(object):
    def __init__(self, xv, yv, z, k=3):
        """
        xv are the x-data nodes, in strictly increasing order
        yv are the y-data nodes, in strictly increasing order
            both of these must be equispaced!
        (though x, y spacing need not be the same)
        z is the data
        k is the order of the splines (int)
            order k splines give interp accuracy of order k+1
            only 1, 3, 5, supported
        """
        if k not in [1, 3, 5]:
            raise Exception('k must be 1, 3, or 5')

        self.xv_offset = -xv.min()
        self.xv_scale = 1.0 / (xv.max()-xv.min())

        self.yv_offset = -yv.min()
        self.yv_scale = 1.0 / (yv.max()-yv.min())

        self.xv = (xv+self.xv_offset)*self.xv_scale
        self.yv = (yv+self.yv_offset)*self.yv_scale


        self.z = z
        self.k = k
        self._dtype = yv.dtype
        terp = RectBivariateSpline(self.xv, self.yv, self.z, kx=self.k, ky=self.k)
        self._tx, self._ty, self._c = terp.tck
        self._nx = self._tx.shape[0]
        self._ny = self._ty.shape[0]
        self._hx = self.xv[1] - self.xv[0]
        self._hy = self.yv[1] - self.yv[0]
        self._nnx = self.xv.shape[0]-1
        self._nny = self.yv.shape[0]-1
        self._cr = self._c.reshape(self._nnx+1, self._nny+1)

    def __call__(self, op_x, op_y, out=None):
        """
        out_points are the 1d array of x values to interp to
        out is a place to store the result
        """
        op_x = (op_x+self.xv_offset)*self.xv_scale
        op_y = (op_y+self.yv_offset)*self.yv_scale

        m = int(np.prod(op_x.shape))
        copy_made = False
        if out is None:
            _out = np.empty(m, dtype=self._dtype)
        else:
            # hopefully this doesn't make a copy
            _out = out.ravel()
            if _out.base is None:
                copy_made = True
        _op_x = op_x.ravel()
        _op_y = op_y.ravel()
        splev2(self._tx, self._nx, self._ty, self._ny, self._cr, self.k, \
            _op_x, _op_y, m, _out, self._hx, self._hy, self._nnx, self._nny)
        _out = _out.reshape(op_x.shape)
        if copy_made:
            # if we had to make a copy, update the provided output array
            out[:] = _out
        return _out

class Index:
    def __init__(self, root_fname, fields, fig):
        self.ind = 0
        self.field_ind = 11
        self.num_frames = len(glob.glob(root_fname+"*.hdf"))
        self.num_fields = len(fields)
        self.root_fname = root_fname
        self.fields = fields

        x, y, f, cont_pts = HMHDslab.plot_scalar(self.root_fname+'0001.hdf', self.fields[self.field_ind], nx=200, nz=100, ncont=70, xlims=None, ylims=None)
        plt.close()

        ax =fig.add_subplot(projection='3d')
        self.ax = ax

        surf = ax.plot_surface(x, y, f, edgecolor='navy', lw=0.2, rstride=3, cstride=3, alpha=0.9, cmap='plasma')

        plane_offset = np.min(f)-(np.max(f)-np.min(f))*0.15
        ax.contourf(x, y, f, zdir='z', offset=plane_offset, levels=50, cmap='plasma', alpha=0.8)
        ax.plot(cont_pts[:,0], cont_pts[:,1], plane_offset, 'r-', lw=2.5, alpha=1)

        ax.set_xlabel("Theta")
        ax.set_ylabel("Radial")
        ax.set_zlabel(self.fields[self.field_ind])
        ax.set_title(f"Plotting {self.fields[self.field_ind]} for {self.root_fname+f'{self.ind:04}.hdf'}")

        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))
        ax.set_zlim(plane_offset, np.max(f))

        labels = ax.zaxis.get_ticklabels()
        labelstr = "".join([str(l.get_position()[0]) for l in labels])
        if("e-" in labelstr):
            ax.zaxis.get_offset_text().set_visible(False)
            exponent = int('{:.2e}'.format(np.min(f)).split('e')[1])
            ax.set_zlabel(ax.get_zlabel() + ' ($\\times\\mathdefault{10^{%d}}\\mathdefault{}$)' % exponent, fontsize=14, labelpad=8.0)


    def plot_frame(self):
        ax = self.ax
        curr_lims = ax.get_xlim(), ax.get_ylim()
        fname = self.root_fname+f'{self.ind+1:04}.hdf'
        x, y, f, cont_pts = HMHDslab.plot_scalar(fname, self.fields[self.field_ind], nx=200, nz=100, ncont=70, xlims=None, ylims=None)
        plt.close()
        ax.cla()
        surf = ax.plot_surface(x, y, f, edgecolor='navy', lw=0.2, rstride=3, cstride=3, alpha=0.9, cmap='plasma')
        plane_offset = np.min(f)-(np.max(f)-np.min(f))*0.15
        ax.contourf(x, y, f, zdir='z', offset=plane_offset, levels=50, cmap='plasma', alpha=0.8)
        ax.plot(cont_pts[:,0], cont_pts[:,1], plane_offset, 'r-', lw=2.5, alpha=1.0)
        ax.set_xlim(curr_lims[0])
        ax.set_ylim(curr_lims[1])
        ax.set_zlim(plane_offset, np.max(f))
        ax.set_title(f"Plotting {self.fields[self.field_ind]} for {fname}")
        ax.set_xlabel("Theta")
        ax.set_ylabel("Radial")
        ax.set_zlabel(self.fields[self.field_ind])

        labels = ax.zaxis.get_ticklabels()
        labelstr = "".join([str(l.get_position()[0]) for l in labels])
        if("e-" in labelstr):
            ax.zaxis.get_offset_text().set_visible(False)
            exponent = int('{:.2e}'.format(np.min(f)).split('e')[1])
            ax.set_zlabel(ax.get_zlabel() + ' ($\\times\\mathdefault{10^{%d}}\\mathdefault{}$)' % exponent, fontsize=14, labelpad=8.0)

        plt.draw()

    def next(self, event):
        self.ind += 1
        self.ind %= self.num_frames
        self.plot_frame()

    def prev(self, event):
        self.ind -= 1
        self.ind %= self.num_frames
        self.plot_frame()

    def next_field(self, event):
        self.field_ind += 1
        self.field_ind %= self.num_fields
        self.plot_frame()

    def prev_field(self, event):
        self.field_ind -= 1
        self.field_ind %= self.num_fields
        self.plot_frame()

@numba.njit(parallel=True, fastmath=True, cache=True, error_model='numpy')
def splev2(tx, nx, ty, ny, c, k, x, y, m, z, dx, dy, nnx, nny):
    # fill in the h values for x
    k1 = k+1
    hbx = np.empty((m, 6))
    hhbx = np.empty((m, 5))
    lxs = np.empty(m, dtype=np.int64)
    splev_short(tx, nx, k, x, m, dx, nnx, hbx, hhbx, lxs)
    hby = np.empty((m, 6))
    hhby = np.empty((m, 5))
    lys = np.empty(m, dtype=np.int64)
    splev_short(ty, ny, k, y, m, dy, nny, hby, hhby, lys)
    for i in numba.prange(m):
        sp = 0.0
        llx = lxs[i] - k1
        for j in range(k1):
            llx += 1
            lly = lys[i] - k1
            for k in range(k1):
                lly += 1
                sp += c[llx,lly] * hbx[i,j] * hby[i,k]
        z[i] = sp

@numba.njit(parallel=True, fastmath=True, cache=True, error_model='numpy')
def splev_short(t, n, k, x, m, dx, nn, hb, hhb, lxs):
    # fetch tb and te, the boundaries of the approximation interval
    k1 = k+1
    nk1 = n-k1
    tb = t[k1-1]
    te = t[nk1+1-1]
    l = k1
    l1 = l+1
    adj = int(k/2) + 1
    # main loop for the different points
    for i in numba.prange(m):
        h = hb[i]
        hh = hhb[i]
        # fetch a new x-value arg
        arg = x[i]
        arg = max(tb, arg)
        arg = min(te, arg)
        # search for knot interval t[l] <= arg <= t[l+1]
        l = int(arg/dx) + adj
        l = max(l, k)
        l = min(l, nn)
        lxs[i] = l
        # evaluate the non-zero b-splines at arg.
        h[0] = 1.0
        for j in range(k):
            for ll in range(j+1):
                hh[ll] = h[ll]
            h[0] = 0.0
            for ll in range(j+1):
                li = l + ll + 1
                lj = li - j - 1
                f = hh[ll]/(t[li]-t[lj])
                h[ll] += f*(t[li]-arg)
                h[ll+1] = f*(arg-t[lj])

def find_instr(func, keyword, sig=0, limit=5):
    import re
    count = 0
    for l in func.inspect_asm(func.signatures[sig]).split('\n'):
        if re.match(keyword, l):
            count += 1
            print(l)
            if(count >= limit and limit > 0):
                break
    if count == 0:
        print('No instructions found')

    # a=splev2.inspect_llvm()
    # s = list(a.values())[0]
    # s[s.find('mmX'):]
    # find_instr(splev2, keyword=r'\s*v.*d.*', sig=0, limit=20)
    # numba.threading_layer()

@contextlib.contextmanager
def change_dir(path):
   old_path = os.getcwd()
   os.chdir(path)
   try:
       yield
   finally:
       os.chdir(old_path)

if __name__ == "__main__":

    print("hmhd_slab.py contains tools for running and postprocessing HMHD2D")
