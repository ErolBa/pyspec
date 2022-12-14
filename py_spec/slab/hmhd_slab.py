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
from scipy.fftpack import fft
import matplotlib.animation as anim
from IPython.display import HTML

class HMHDslab():

    def set_inputfile_var(file, varname, varvalue):
        subprocess.run(f"sed -i s/^{varname}=.*/{varname}="+f"{varvalue:.2e}".replace('e','D')+f"\ \ \ \ !auto\ insert\ with\ py/ i{file}", shell=True)

    def get_hmhd_profiles(hmhd_fname, num_conts, mpol, plot_flag):

        plt.rcParams['figure.figsize'] = [13, 8]

        numba.set_num_threads(4)

        if(plot_flag):
            fig, ax = plt.subplots()
            ax.set_title("Plot of $\psi(x,y)$")
            ax.set_xlabel("y")
            ax.set_ylabel("x")

        h5file = h5py.File(hmhd_fname)
        psi = h5file['psi'][2:-2, 2:-1]
        bz = h5file['bz'][2:-2, 2:-1]
        by = h5file['by'][2:-2, 2:-1]
        bx = h5file['bx'][2:-2, 2:-1]
        jz = h5file['jz'][2:-2, 2:-1]
        jy = h5file['jy'][2:-2, 2:-1]
        jx = h5file['jx'][2:-2, 2:-1]
        x = h5file["x"][2:-1]
        z = h5file["z"][2:-2]

        y_coord = x.copy() # transform to spec coords
        x_coord = z.copy() # transform to spec coords

        # c_levels = np.linspace(np.min(psi), np.max(psi), num_conts, endpoint=True)

        X, Y = np.meshgrid(x_coord[::1], y_coord[::1])
        c = plt.contour(Y, X, psi.T, levels=num_conts, colors='black')
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

        plt.figure()

        for cont in range(num_conts): # iterate over each level
        # for cont in [2,7,12]: # iterate over each level
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

                if(len(level)==1):
                    plt.plot(y,x,'r-')
                else:
                    plt.plot(y,x,'b--')

                # check if the path is flat (or belongs to an island
                if(np.abs(path_array[-1,0]-path_array[0,0]) < 1e-4):
                    # if(plot_flag):
                    #     ax1.plot(y, x, '-', color=color)
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

                    # # check reconstructed interfaces are correct (and that ft was done well)
                    # if(plot_flag):
                    #     xrecons = np.zeros_like(ynew)
                    #     xrecons += ft.real[0] / (len(ft))
                    #     for m in range(1, mpol+1):
                    #         xrecons += 2 * ft.real[m]*np.cos(m*(np.pi + np.pi * ynew / width)) / (len(ft))
                    #     ax1.plot(ynew, xrecons , '-.', color=color, zorder=5)

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
            plt.rcParams['figure.figsize'] = [12, 4]

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

        plt.rcParams['figure.figsize'] = [10.5, 3.5]
        plt.rc('axes', titlesize=17)
        plt.rc('axes', labelsize=13)
        plt.rc('legend',fontsize=13)

        for f in which:
            if(f in ['bz','jy']):
                args = [psi_mag, bz_mag]
            else:
                args = [psi_mag]

            plt.figure()
            plt.title(f)
            plt.plot(x_array, initial_config[f](x_array, *args))
            plt.plot(x_array, config[f](x_array, *args), 'r--')
            plt.vlines(0, plt.ylim()[0], plt.ylim()[1], alpha=0.5)
            plt.xlabel("x position")
            plt.xlim([-np.pi-0.0,np.pi+0.0])
            plt.axhline(0, alpha=0.7, color='k')

            if(f is 'psi'):
                plt.legend(['loureiro','new config'])

    def calc_delprime_loureiro(k):
        return 2*(5-k**2)*(3+k**2)/(k**2*np.sqrt(4+k**2))

    def calc_xl_for_delprimea(delprimea):


        return xl

    def plot_paper_poem(delprime):
        a = 0.35
        poem_results = 2.44 * delprime * a
        plt.plot(delprime*a, poem_results,'k-', label="POEM")

    def set_hmhd_profiles(symm_config):
        # change the profiles in prob.f90 of HMHD

        psii_string = str((symm_config['psi_sym'])).replace("exp", "???").replace("psi0","a").replace("x","z(k)").replace("/","\/").replace("???", "exp")
        byi_string = str((symm_config['bz_sym'])).replace("exp", "???").replace("psi0","a").replace("Bz0","qpa").replace("x","z(k)").replace("/","\/").replace("???", "exp")

        sed_string_psii = f"sed -i  '/.*\!/!s/.*psii(i,k)=1.00.*/\t\tpsii(i,k)=1.00*{psii_string}/' ~/HMHD2D/prob.f90"
        sed_string_byi = f"sed -i  '/.*\!/!s/.*byi(i,k)=1.00.*/\t\tbyi(i,k)=1.00*{byi_string}/' ~/HMHD2D/prob.f90"
        subprocess.run(sed_string_psii, shell=True)
        subprocess.run(sed_string_byi, shell=True)

        sed_string_psii = f"sed -i  '/.*\!/!s/.*psiii(i,k)=1.00.*/\t\tpsiii(i,k)=1.00*{psii_string}/' ~/HMHD2D/prob.f90"
        sed_string_byi = f"sed -i  '/.*\!/!s/.*byii(i,k)=1.00.*/\t\tbyii(i,k)=1.00*{byi_string}/' ~/HMHD2D/prob.f90"
        subprocess.run(sed_string_psii, shell=True)
        subprocess.run(sed_string_byi, shell=True)

        subprocess.run(f"cd ~/HMHD2D; make", shell=True)

    def run_hmhd():
        subprocess.run(f"sh runhmhd.sh Tearing", shell =True)

    def animate_imshow(i):
        data = get_array_from_hdf("run_Tearing/"+f"data{i+1:04}.hdf","psi")
        im.set_array(data)
        im.set_clim(vmin=np.min(data), vmax=np.max(data))
        cb.set_clim(vmin=np.min(data), vmax=np.max(data))
        return im

    def animate_cont(i):

        global cont, contf

        h5file = h5py.File("run_Tearing/"+f"data{i+1:04}.hdf",'r')
        data = h5file['psi'][:]
        x = h5file["x"][:]
        z = h5file["z"][:]

        for c in contf.collections:
            c.remove()
        for c in cont.collections:
            c.remove()

        X, Z = np.meshgrid(x,z)
        data_max = np.max(np.abs(data))
        contf = ax.contourf(X, Z, data/data_max, levels=num_contours_global, alpha=0.4, cmap='plasma')
        cont = ax.contour(X, Z, data/data_max, levels=num_contours_global, colors='black', linestyles='solid')
        #fig.colorbar(contf)
        return contf, cont

    def animate_run(root_fname="run_Tearing/", num_contours=30):

        global fig, ax, cont, contf, num_contours_global

        fig, ax = plt.subplots()
        ax.set_title("Plot of $\psi(x,y)$")
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        contf = ax.contourf([[1,2],[2,4]], alpha=0)
        cont = ax.contour([[1,2],[5,2]], alpha=0)
        num_contours_global = num_contours

        fps = 8
        ani = anim.FuncAnimation(fig, HMHDslab.animate_cont, frames=len(glob.glob('run_Tearing/data*.hdf')), interval=1000/fps, blit=False, repeat=False);
        plt.close()

        return HTML(ani.to_jshtml())

    def get_width_As_HMHD(outfile, flag_plot=True):

        if(flag_plot):
            # plt.rcParams['figure.figsize'] = [8, 9]
            plt.figure()

        h5 = h5py.File(outfile,'r')
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

        # print((c.collections[0].get_paths())[2])
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
            r_x = (cont1[np.argmax(cont1[:,0]),1] + cont2[np.argmin(cont2[:,0]),1]) * 0.5
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

            print(f"HMHD Island width {island_w:.4f}  As {island_As:.4f}")

        else:
            plt.close()

        return island_w, island_As

    def get_width_run(root_fname="run_Tearing"):
        fnames_h5files = sorted(glob.glob(root_fname+"/data*.hdf"))
        w_sat_vals = []
        As_sat_vals = []
        for f in range(len(fnames_h5files)):
            w_curr, As_curr = HMHDslab.get_width_As_HMHD(fnames_h5files[f], False)
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

    def find_psi_perturbed(k, psi0_fun, by_fun, d2by_fun, numpts_xmesh=32, flag_BCs=0):

        # have to solve left and right regions
        # for loureiro the two are symmetric
        # but not in general

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

        x_init = np.linspace(1e-10, np.pi, numpts_xmesh)
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

    def eval_delprime_sym(psi0_string, psi0_mag, x=None, k=None):

        if(x is not None):
            k = 2 * np.pi / x
        else:
            if(k is None):
                raise ValueError("Have to set k or x in eval_delprime")

        sym_config = HMHDslab.gen_profiles_for_delprime(psi0_string)
        psi0_fun = lambda x: sym_config['psi'](x, psi0_mag)
        by_fun = lambda x: sym_config['by'](x, psi0_mag)
        d2by_fun = lambda x: sym_config['d2by'](x, psi0_mag)

        sol_left, sol_right = HMHDslab.find_psi_perturbed(k, psi0_fun, by_fun, d2by_fun, 54, 0)

        delprime = (sol_right.y[1,0] - sol_left.y[1,-1]) / sol_right.y[0,0]
        sigprime = (sol_right.y[1,0] + sol_left.y[1,-1]) / sol_right.y[0,0]

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


if __name__ == "__main__":

    print("HMHDslab contains tools for running and analyzing HMHD")
