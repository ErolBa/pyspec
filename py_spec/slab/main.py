## Written by Erol Balkovic

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
from py_spec import SPECNamelist, SPECout
from scipy import integrate, interpolate, optimize
import subprocess
import sys
import numpy.linalg as linalg
import numba
import os
import contextlib
import sympy as sym
from scipy.optimize import minimize_scalar

from .input_dict import input_dict

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

		Linfaces_val = np.linspace(values[0], values[ind_Lresinface], (Nvol+1)//2, endpoint=True)
		Rinfaces_val = np.linspace(values[-1], values[ind_Rresinface], (Nvol+1)//2, endpoint=True)[::-1]
		Linfaces_ind = np.argmin(np.abs(values[:len(x)//2,None]-Linfaces_val[None,:]), axis=0)
		Rinfaces_ind = np.argmin(np.abs(values[len(x)//2:,None]-Rinfaces_val[None,:]), axis=0) + (len(x)//2)
		infaces_ind = SPECslab.join_arrays(Linfaces_ind, Rinfaces_ind)
		infaces_val = SPECslab.join_arrays(Linfaces_val, Rinfaces_val)
		infaces_x = x[infaces_ind]
		return infaces_ind, infaces_x, infaces_val

	def gen_ifaces_funcspaced_new(x, ind_Lresinface, ind_Rresinface, func_deriv, Nvol):
		deriv = func_deriv(x)
		a = 1.0
		dx = a / deriv

		integ = [integrate.quad(func_deriv, -np.pi, a)[0] for a in x]

		plt.figure()
		plt.plot(x, deriv)
		plt.show()


		values = func_deriv(x)
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
		if(psi(x[0]) > 1e-8):
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
			#mode_number = int((nvol-1) * (mpol + 1))
			H = np.fromfile(fid, 'float64').reshape((int(ngdof), int(ngdof))).T

		return H

	def get_hessian_new(fname):

		# reads the hessian from .hessian file, apparently made prior to fourier transforms
		if(fname[-3:] == '.sp'):
			fname = fname + ".hessian"
		elif(fname[-3:] == '.h5'):
			fname = fname[:-3]+".hessian"
		elif(fname[-4:] == '.end'):
			fname = fname[:-4]+".hessian"
		elif(fname[-8:] == ".hessian"):
			fname = fname
		else:
			raise ValueError("Invalid file given to get_hessian()")

		def read_int(fid):
			return np.fromfile(fid, 'int32',1)

		with open(fname, 'rb') as fid:
			read_int(fid)
			NGdof = int(read_int(fid))
			read_int(fid)
			read_int(fid)

			data = np.fromfile(fid, 'float64').reshape((NGdof, NGdof)).T

		return data


	def get_eigenstuff(fname, which='new'):
		"""
		ARGS:
			fname:
				name of SPEC file
			which:
				"old" gets hessian from .###.sp.DF (not that good, has asymmetry)
				"new" gets hessian from .###.hessian (default)
		"""

		if(which=="old"):
			h = SPECslab.get_hessian(fname)
		elif(which=="new"):
			h = SPECslab.get_hessian_new(fname)
		else:
			raise ValueError("Invalid args to get_eigenstuff()")

		w, v = linalg.eig(h)

		## there are two ways of determining the smallest eigenvalue (choose one)
		# first, looking at the smallest real part eigenvalue (numpy defualt btw)
		indmin = np.argmin(w)

		## second, looking at the largest magnitude of the complex number (default in MATLAB)
		# indmin2 = np.argmin(np.abs(w))

		minvec = v[:, indmin]

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

		# geometry =  data['input']['physics']['Igeometry'][0]
		# rpol =  data['input']['physics']['rpol'][0]
		# rtor =  data['input']['physics']['rtor'][0]

		geometry =  data.input.physics.Igeometry
		rpol =  data.input.physics.rpol
		rtor =  data.input.physics.rtor

		Rarr, Zarr = SPECslab.get_spec_R_derivatives(data, lvol, sarr, theta, zeta, 'R')

		if(geometry == 1):
			return Rarr[1] #* rtor * rpol # fixed here
		elif(geometry == 2):
			return Rarr[0] * Rarr[1]
		elif(geometry == 3):
			return Rarr[0] * (Rarr[2]*Zarr[1] - Rarr[1]*Zarr[2])
		else:
			raise ValueError("Error: unsupported dimension")

	def get_spec_R_derivatives(data, lvol, sarr, tarr, zarr, RorZ):
		# the vol index is -1 compared to the matlab one

		geometry = data.input.physics.Igeometry
		im = data.output.im
		_in = data.output.in_
		mn = data.output.mn
		mregular = data.input.numerics.Mregular
		Rbc = data.output.Rbc
		Rmn_p = Rbc[lvol+1,:]
		Rmn = Rbc[lvol,:]
		Zbs = data.output.Zbs
		Zmn = Zbs[lvol,:]
		Zmn_p = Zbs[lvol+1,:]

		ns = len(sarr)
		nt = len(tarr)
		nz = len(zarr)

		cosa = np.cos(im[:,None,None] * tarr[None,:,None] - _in[:,None,None] * zarr[None,None,:]) # [j, it, iz]
		sina = np.sin(im[:,None,None] * tarr[None,:,None] - _in[:,None,None] * zarr[None,None,:]) # [j, it, iz]
		factor = SPECslab.get_spec_regularization_factor(geometry, mn, im, lvol, sarr, mregular, 'G')

		R = np.zeros((4, ns, nt, nz))
		sum_string = 'mtz,ms->stz'
		R[0] = np.einsum(sum_string, cosa * (Rmn + (Rmn_p - Rmn))[:,None,None], factor[:,0])
		R[1] = np.einsum(sum_string, cosa * (Rmn_p - Rmn)[:,None,None], factor[:,1])
		R[2] = np.einsum(sum_string,-sina * ((Rmn + (Rmn_p - Rmn)) * im)[:,None,None], factor[:,0])
		R[3] = np.einsum(sum_string, sina * ((Rmn_p - Rmn) * _in)[:,None,None], factor[:,0])

		Z = np.zeros((4, ns, nt, nz))
		Z[0] = np.einsum(sum_string, sina * (Zmn + (Zmn_p - Zmn))[:,None,None], factor[:,0])
		Z[1] = np.einsum(sum_string, sina * (Zmn_p - Zmn)[:,None,None], factor[:,1])
		Z[2] = np.einsum(sum_string, cosa * ((Zmn + (Zmn_p - Zmn)) * im)[:,None,None], factor[:,0])
		Z[3] = np.einsum(sum_string,-cosa * ((Zmn_p - Zmn) * _in)[:,None,None], factor[:,0])

		return R, Z

	def bla():
		o = SPECout('test.sp')

		o.get_B()


	def get_spec_poly_basis(Lsingularity, mpol, lrad, sarr):

		ns = len(sarr)

		T = np.ones((lrad+1, 2, ns))
		T[0,1] = 0
		T[1,0] = sarr

		if(Lsingularity):
			#NASTY STUFF
			zernike = np.zeros((lrad+1, mpol+1, 2, ns))
			rm = np.ones(ns)
			rm1 = np.zeros(ns)

			sbar = (sarr+1)/2

			for m in range(-1, mpol):
				if(lrad >= m):
					zernike[m+1,m+1,0,:] = rm
					zernike[m+1,m+1,1,:] = (m+1) * rm1

				if(lrad >= m+3):
					zernike[m+3,m+1,0,:] = (m+3) * rm * sbar**2 - (m+2) * rm
					zernike[m+3,m+1,1,:] = (m+3)**2 * rm * sbar - (m+2)*(m+1)*rm1

				for n in range(m+4-1,lrad,2):
					factor1 = (n+1) / ((n+1)**2 - (m+1)**2)
					factor2 = 4 * (n)
					factor3 = (n-1+m+1)**2/(n-1) + (n-m)**2/(n+1)
					factor4 = ((n-1)**2-(m+1)**2)/(n-1)

					zernike[n+1,m+1,0,:] = factor1 * ((factor2*sbar**2-factor3)*zernike[n-1,m+1,0,:]) - factor4 * zernike[n-3,m+1,1,:]
					zernike[n+1,m+1,1,:] = factor1 * (2*factor2*sbar * zernike[n-1,m+1,1,:] + (factor2*sbar**2 - factor3)*zernike[n-1,m+1,1,:] - factor4 * zernike[n-3,m+1,1,:])

				rm1 = rm
				rm *= sbar

			for n in range(1, lrad, 2):
				zernike[n+1,0,0,:] -= (-1)**((n+1)/2)

			if(mpol >= 1):
				for n in range(3, lrad, 2):
					zernike[n+1,1,0,:] -= (-1)**((n-1)/2) * (n+1)/2 * sbar
					zernike[n+1,1,1,:] -= (-1)**((n-1)/2) * (n+1)/2

			for m in range(-1,mpol):
				for n in range(m, lrad, 2):
					zernike[n+1,m+1,:,:] /= (n)

			T[:,0,ns] = np.reshape(zernike[:,:,0,:],(mpol+1,ns))
			T[:,1,ns] = np.reshape(zernike[:,:,1,:],(mpol+1,ns))

		else:
			for l in range(2, lrad+1):
				T[l,0] = 2*sarr*T[l-1,0] - T[l-2,0]
				T[l,1] = 2*T[l-1,0] + 2*sarr*T[l-1,1] - T[l-2,1]

			for l in range(lrad):
				T[l+1,0] -= (-1)**(l+1)

			for l in range(-1, lrad):
				T[l+1,0] /= (l+2)
				T[l+1,1] /= (l+2)

		return T

	def get_contra2cov(data, lvol, vec_contrav, sarr, theta, phi, norm):

		g = SPECslab.get_spec_metric(data, lvol, sarr, theta, phi)

		# print(g.shape, vec_contrav.shape)
		vec_cov = np.einsum('xystz,ystz->xstz', g, vec_contrav)
		return vec_cov

	def get_spec_regularization_factor(geometry, mn, im, lvol, sarr, mregular, ForG):

		ns = len(sarr)

		sbar = np.array((1+sarr)/2)
		reumn = im/2
		fac = np.zeros((mn, 2, ns))

		if(mregular > 1):
			ind = np.argmax(regumn > mregular)
			regumn[ind] = mregular / 2

		if(ForG == 'G'):
			if(geometry == 1):
				for j in range(mn):
					fac[j,0] = sbar
					fac[j,1] = 0.5*np.ones(ns)
			elif(geometry == 2):
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
			elif(geometry == 3):
				for j in range(mn):
					if(lvol == 0):
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
		nt = len(theta)
		nz = len(zeta)
		
		G = data.input.physics.Igeometry
		rtor = data.input.physics.rtor
		rpol = data.input.physics.rpol

		gmat = np.zeros((3, 3, ns, nt, nz))
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

	def plot_kamsurf(fname, title=None, ax=None):

		if(fname[-3:] == '.sp'):
			fname = fname+".h5"
		elif(fname[-4:] == '.end'):
			fname = fname[:-4]+".h5"

		if(not os.path.isfile(fname)):
			print(f"File '{fname}' does not exist (plot_kamsurf())")
			return

		if(title is None):
			# title = f"KAM surfaces ({fname})"
			title = ""

		if(ax is None):
			fig, ax = plt.subplots()
		
		myspec = SPECout(fname)
		surfs = myspec.plot_kam_surface(marker='', ls='-', c='b', lw=1.8, ax=ax)

		ax.set_ylim([0-0.2,2*np.pi+0.2])
		ax.set_xlim([0-0.1,2*np.pi+0.1])

		ax.set_xlabel("y position")
		ax.set_ylabel("x position")

		plt.gcf().canvas.manager.set_window_title(title)
		plt.title(title)

		# thetas = np.linspace(0, 2*np.pi, 200)
		# x_four = myspec.output.Rbc[:,:]
		# x = np.zeros((x_four.shape[0], len(thetas)))
		# for i in range(x_four.shape[1]):
		#     x += np.cos(i*thetas)[None,:] * (x_four[:,i])[:,None]
		#
		# plt.figure()
		# for vol in range(x_four.shape[0]):
		#     plt.plot(thetas, x[vol])

	def old_get_spec_vecpot(data, lvol, sarr, tarr, zarr):

		# get vector potential from .h5 spec outpout that uses old/regular chebyshev basis (not recombined)

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

		At_dAt = np.einsum('sa,lia,ljtz->istz', fac[:,0], T, term_t, optimize=True)
		Az_dAz = np.einsum('sa,lia,ljtz->istz', fac[:,0], T, term_z, optimize=True)

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

	def get_islandwidth(fname, ns=500, nt=500, plot=False, plot_title=None, ax=None, xlims=[0, 2*np.pi], ylims=None):

		if(fname[-3:] == '.sp'):
			fname = fname+".h5"
		elif(fname[-4:] == '.end'):
			fname = fname[:-4]+".h5"

		if(not os.path.isfile(fname)):
			print(f"File '{fname}' does not exist (get_islandwidth())")
			return

		data = SPECout(fname)

		lvol =  data.input.physics.Nvol // 2 #-11
		sarr = np.linspace(-1, 1, ns)
		tarr = np.linspace(0, 2*np.pi, nt)

		if(plot):
			if(ax is None):
				fig, ax = plt.subplots()
				fig.set_size_inches(5, 5)

		# plot vecpot in other volumes as well
		if(plot):
			col = 'blue'
			for v in np.arange(1, lvol, 1):
				At, Az = SPECslab.get_spec_vecpot(fname, v, sarr, tarr, np.array([0]))
				Rarr, Tarr, dRarr = SPECslab.get_rtarr(data, v, sarr, tarr, np.array([0]))
				ax.contour(Tarr, Rarr, Az[:,:,0], levels=1, alpha=0.9, colors=col, linestyles='solid')

			for v in np.arange(lvol+1, data.input.physics.Nvol-1, 1):
				At, Az = SPECslab.get_spec_vecpot(fname, v, sarr, tarr, np.array([0]))
				Rarr, Tarr, dRarr = SPECslab.get_rtarr(data, v, sarr, tarr, np.array([0]))
				ax.contour(Tarr, Rarr, Az[:,:,0], levels=[np.mean(Az)], alpha=0.9, colors=col, linestyles='solid')
	#
		At, Az = SPECslab.get_spec_vecpot(fname, lvol, sarr, tarr, np.array([0]))
		Rarr, Tarr, dRarr = SPECslab.get_rtarr(data, lvol, sarr, tarr, np.array([0]))

		o_point = np.unravel_index((Az[:,:,0]).argmin(), Az[:,:,0].shape)
		x_point = np.unravel_index(((Az[:,:,0])**2).argmin(), Az[:,:,0].shape)

		# plt.ioff()
		plt.figure()
		levels = np.linspace(Az[o_point][0], Az[x_point][0], 700)[1:-1]
		c2 = plt.contour(Tarr, Rarr, Az[:,:,0], levels=levels, colors='black',linestyles='solid', alpha=0)
		plt.close()
		# plt.ion()

		max_closed_ind = -1
		for i in range(len(c2.collections)):
			verts = c2.collections[i].get_paths()[0].vertices
			if(len(c2.collections[i].get_paths()) == 1 and np.linalg.norm(verts[-1]-verts[0]) < 1e-5):
				max_closed_ind = i
		cont_pts = [c2.collections[max_closed_ind].get_paths()[0].vertices]

		if(plot):
			# plt.contourf(Tarr, Rarr, Az[:,:,0], levels=20, alpha=0.9)
			# plt.colorbar()
			ax.contour(Tarr, Rarr, Az[:,:,0], levels=15, alpha=1., colors='black', linestyles='solid')

			# plot the res. volume bondary
			theta_bnd = np.linspace(0, 2*np.pi, 200)
			rarr_bnd, tarr_bnd, _ = SPECslab.get_rtarr(data, lvol, [-1,1], theta_bnd, np.array([0]))
			for i in [0, 1]:
				ax.plot(theta_bnd, rarr_bnd[i], color='royalblue', lw=3, zorder=20)

		if(len(cont_pts) < 1):
			island_w = 0.0
		else:
			cont_pts = np.concatenate(cont_pts)
			island_w = np.max(cont_pts[:, 1]) - np.min(cont_pts[:, 1])

			r_up = np.max(cont_pts[:, 1])
			r_down = np.min(cont_pts[:, 1])

			# r_x = Rarr[x_point]
			r_x = cont_pts[np.argmin(cont_pts[:,0]), 1]
			
			if(abs(r_x-r_down) < 1e-10):
				Asym = 0
			else:
				Asym = (r_up-r_x)/(r_x-r_down) - 1

			# print(f"r_up,down,x -- {r_up-np.pi} {r_down-np.pi} {r_x-np.pi}")

			if(plot):
				ax.plot(cont_pts[:,0], cont_pts[:,1], 'r-', lw=3.5)

				# ax.axhline(np.min(cont_pts[:, 1]), color='red', linestyle='dashed', lw=2.3)
				# ax.axhline(np.max(cont_pts[:, 1]), color='red', linestyle='dashed', lw=2.3)
				ax.axhline(r_x, color='r', linestyle='dashed', lw=2.3)
				ax.plot(cont_pts[np.argmin(cont_pts[:,0]), 0], cont_pts[np.argmin(cont_pts[:,0]), 1],'rX', ms=11)
				ax.plot(Tarr[o_point],Rarr[o_point],'ro', ms=10)
				ax.plot(Tarr[0], Rarr[0], 'k-', lw=1)
				ax.plot(Tarr[-1], Rarr[-1], 'k-', lw=1)

		if(plot):
			if(plot_title is None):
				plot_title = f"SPEC island (width {island_w:.3f} Asym {Asym:.3f})"
				# plot_title = f"SPEC island (width {island_w:.4f})"
			ax.set_title(plot_title, fontsize=20)
			plt.gcf().canvas.manager.set_window_title(plot_title + f" w={island_w:.3f}")

			if(ylims is None):
				if(island_w > 1e-3):
					ylims = [r_x-island_w, r_x+island_w]
				else:
					ylims = [0, 2*np.pi]
			ax.set_ylim(ylims)
			ax.set_xlim(xlims)
			ax.set_xlabel("y", fontsize=16)
			ax.set_ylabel("x", fontsize=16)
			plt.tight_layout()

		return island_w, Asym


	def get_islandwidth_returnrpmx(fname, ns=500, nt=500, plot=False, plot_title=None, xlims=[0, 2*np.pi], ylims=[0, 2*np.pi]):

		if(fname[-3:] == '.sp'):
			fname = fname+".h5"
		elif(fname[-4:] == '.end'):
			fname = fname[:-4]+".h5"

		if(not os.path.isfile(fname)):
			print(f"File '{fname}' does not exist (get_islandwidth())")
			return

		data = SPECout(fname)

		fdata = data.vector_potential
		lvol =  data.input.physics.Nvol // 2 #-11
		sarr = np.linspace(-1, 1, ns)
		tarr = np.linspace(0, 2*np.pi, nt)

		At, Az = SPECslab.get_spec_vecpot(fname, lvol, sarr, tarr, np.array([0]))
		Rarr, Tarr, dRarr = SPECslab.get_rtarr(data, lvol, sarr, tarr, np.array([0]))

		o_point = np.unravel_index((Az[:,:,0]).argmin(), Az[:,:,0].shape)
		x_point = np.unravel_index(((Az[:,:,0])**2).argmin(), Az[:,:,0].shape)

		plt.ioff()
		plt.figure()
		levels = np.linspace(Az[o_point][0], Az[x_point][0], 700)[1:-1]
		c2 = plt.contour(Tarr, Rarr, Az[:,:,0], levels=levels, colors='black',linestyles='solid', alpha=0)
		plt.close()
		# plt.ion()

		max_closed_ind = -1
		for i in range(len(c2.collections)):
			verts = c2.collections[i].get_paths()[0].vertices
			if(len(c2.collections[i].get_paths()) == 1 and np.linalg.norm(verts[-1]-verts[0]) < 1e-5):
				max_closed_ind = i
		cont_pts = [c2.collections[max_closed_ind].get_paths()[0].vertices]

		if(plot):
			plt.figure()
			plt.contourf(Tarr, Rarr, Az[:,:,0], levels=20, alpha=0.9)
			# plt.colorbar()
			plt.contour(Tarr, Rarr, Az[:,:,0], levels=20, alpha=0.5, colors='black', linestyles='solid')

		if(len(cont_pts) < 1):
			island_w = 0.0
		else:
			cont_pts = np.concatenate(cont_pts)
			island_w = np.max(cont_pts[:, 1]) - np.min(cont_pts[:, 1])

			r_up = np.max(cont_pts[:, 1])
			r_down = np.min(cont_pts[:, 1])

			# r_x = Rarr[x_point]
			r_x = cont_pts[np.argmin(cont_pts[:,0]), 1]
			Asym = (r_up-r_x)/(r_x-r_down) - 1

			# print(f"r_up,down,x -- {r_up-np.pi} {r_down-np.pi} {r_x-np.pi}")

			if(plot):
				plt.plot(cont_pts[:,0], cont_pts[:,1], 'r-', lw=2)

				plt.axhline(np.min(cont_pts[:, 1]), color='red', linestyle='dashed', lw=1)
				plt.axhline(np.max(cont_pts[:, 1]), color='red', linestyle='dashed', lw=1)
				plt.axhline(r_x, color='k', linestyle='dashed', lw=1.5)
				plt.plot(cont_pts[np.argmin(cont_pts[:,0]), 0], cont_pts[np.argmin(cont_pts[:,0]), 1],'rX', ms=9)
				plt.plot(Tarr[o_point],Rarr[o_point],'ro', ms=9)
				plt.plot(Tarr[0], Rarr[0], 'k-', lw=1)
				plt.plot(Tarr[-1], Rarr[-1], 'k-', lw=1)

		if(plot):
			if(plot_title is None):
				plot_title = f"SPEC A_z resonant volume (width {island_w:.4f} Asym {Asym:.4f})"
			plt.title(plot_title)
			plt.gcf().canvas.manager.set_window_title(plot_title + f" w={island_w:.3f}")
			plt.ylim(ylims)
			plt.xlim(xlims)
			plt.tight_layout()

		print(f"SPEC Island width {island_w:.8f}")

		return island_w, Asym, r_up, r_down, r_x


	def old_get_islandwidth(fname, ns=200, nt=200, plot=False, plot_title=None):

		if(fname[-3:] == '.sp'):
			fname = fname+".h5"
		elif(fname[-4:] == '.end'):
			fname = fname[:-4]+".h5"

		if(not os.path.isfile(fname)):
			print(f"File '{fname}' does not exist (get_islandwidth())")
			return

		data = SPECout(fname)

		fdata = data.vector_potential
		lvol =  data.input.physics.Nvol // 2
		sarr = np.linspace(-1, 1, ns)
		tarr = np.linspace(0, 2*np.pi, nt)

		# std approach
		At, Az, dAt, dAz = SPECslab.old_get_spec_vecpot(data, lvol, sarr, tarr, np.array([0]))
		Rarr, Tarr, dRarr = SPECslab.get_rtarr(data, lvol, sarr, tarr, np.array([0]))

		o_point = np.unravel_index((Az[:,:,0]).argmin(), Az[:,:,0].shape)
		x_point = np.unravel_index(((Az[:,:,0])**2).argmin(), Az[:,:,0].shape)

		plt.figure()
		# c = plt.contour(Tarr, Rarr, Az[:,:,0], levels=[Az[x_point][0]],colors='red',linestyles='solid')
		levels = np.linspace(Az[o_point][0], Az[x_point][0], 41)[1:-1]
		c2 = plt.contour(Tarr, Rarr, Az[:,:,0], levels=levels, colors='black',linestyles='solid', alpha=0)
		plt.close()

		max_closed_ind = -1
		for i in range(len(c2.collections)):
			if(len(c2.collections[i].get_paths()) == 1):
				max_closed_ind = i

		cont_pts = [c2.collections[max_closed_ind].get_paths()[0].vertices]

		if(plot):
			plt.figure()
			plt.contourf(Tarr, Rarr, Az[:,:,0], levels=20, alpha=0.9)
			plt.colorbar()
			plt.contour(Tarr, Rarr, Az[:,:,0], levels=20, alpha=0.5, colors='black', linestyles='solid')

		if(len(cont_pts) < 1):
			island_w = 0.0
		else:
			cont_pts = np.concatenate(cont_pts)
			island_w = np.max(cont_pts[:, 1]) - np.min(cont_pts[:, 1])

			if(plot):
				plt.plot(cont_pts[:,0], cont_pts[:,1], 'r-', lw=2)

				plt.axhline(np.min(cont_pts[:, 1]), color='red', linestyle='dashed', lw=1)
				plt.axhline(np.max(cont_pts[:, 1]), color='red', linestyle='dashed', lw=1)
				# plt.plot(Tarr[x_point],Rarr[x_point],'rX', ms=9)
				plt.plot(Tarr[o_point],Rarr[o_point],'ro', ms=9)

				plt.plot(Tarr[0], Rarr[0], 'k-', lw=1)
				plt.plot(Tarr[-1], Rarr[-1], 'k-', lw=1)


		if(plot):
			if(plot_title is None):
				plot_title = f"A_z resonant volume (island width {island_w:.4f})"
			plt.title(plot_title)
			plt.gcf().canvas.manager.set_window_title(plot_title + f" w={island_w:.3f}")

			# plt.ylim([np.pi-1,np.pi+1])
			plt.ylim([0, 2*np.pi])
			# plt.ylim([np.min(Rarr)-0.1, np.max(Rarr)+0.1])

			plt.xlim([0, 2*np.pi])
			# plt.xlim([0, np.pi+0.1])

			plt.tight_layout()

		print(f"SPEC Island width {island_w:.8f}")

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
		plt.gcf().canvas.manager.set_window_title('A_z')

		for lvol in lvol_list:
			Rarr, Tarr, dRarr = SPECslab.get_rtarr(data, lvol, sarr, tarr, z0)
			At, Az = SPECslab.get_spec_vecpot(fname, lvol, sarr, tarr, z0)
			ax.contourf(Tarr,Rarr,Az[:,:,0], levels=40)

		return ax

	def get_full_field(data, r, theta, zeta, nr, const_factor=None):

		nvol = data.output.Mvol
		G = data.input.physics.Igeometry
		mpol = data.input.physics.Mpol 
		im = data.output.im
		_in = data.output.in_

		iimin = 0
		iimax = 1

		r0, z0 = SPECslab.get_spec_radius(data, theta, zeta, -1)
		B = np.zeros((3, len(r)))
		for i in range(nvol):

			lrad = data.input.physics.Lrad[i]

			ri, zi = SPECslab.get_spec_radius(data, theta, zeta, i-1)
			rmin = np.sqrt((ri-r0)**2 + zi**2)

			ri, zi = SPECslab.get_spec_radius(data, theta, zeta, i)
			rmax = np.sqrt((ri-r0)**2 + zi**2)
			r_vol = np.linspace(rmin, rmax, nr+1)#[1:]

			sarr = 2 * (r_vol - rmin) / (rmax - rmin) - 1
			if(i == 0 and G != 1):
				sarr = 2 * ((r_vol - rmin) / (rmax - rmin))**2 - 1

			jac = SPECslab.get_spec_jacobian(data, i, sarr, theta, zeta)
			g = SPECslab.get_spec_metric(data, i, sarr, theta, zeta)

			Lsingularity = (i == 0) and (G != 1)

			T = SPECslab.get_spec_poly_basis(Lsingularity, mpol, lrad, sarr)

			Ate = np.array(data.vector_potential.Ate)[i]
			Aze = np.array(data.vector_potential.Aze)[i]
			Ato = np.array(data.vector_potential.Ato)[i]
			Azo = np.array(data.vector_potential.Azo)[i]
			
			B_contrav = vecpot_to_contrav(T, Ate, Aze, Ato, Azo, jac, im, _in, theta, zeta)

			iimax = iimax + len(r_vol)

			# B_cov = SPECslab.get_contra2cov(data, i, B_contrav, sarr, theta, zeta, 1)

			B_cart = np.zeros_like(B_contrav)
			B_cart[0] = B_contrav[0] / g[0,0,0,0]**0.5
			B_cart[1] = B_contrav[1] / g[1,1,0,0]**0.5#/ (np.sqrt(g[1,1]) )#/ jac[:])
			B_cart[2] = B_contrav[2] / g[2,2,0,0]**0.5 #/ (np.sqrt(g[2,2]) )#/ jac[:])

			if(const_factor is not None):
				B_cart *= const_factor[i] # multiply by mu to get current, or else

			iimin = iimin + len(r_vol)
			ind = np.logical_and(r <= rmax, r >= rmin)
			r_int = r[ind]

			for comp in range(3):
				f = interpolate.interp1d(r_vol, B_cart[comp,:,0,0], kind='cubic')
				B[comp, ind] = f(r_int)
		
		return B


	def get_spec_vecpot(file, lvol, sarr=None, tarr=None, zarr=None):

		# get vector potential from .h5 spec outpout that uses recombined chebyshev basis (not regular)

		if(isinstance(file, str)):
			data = h5py.File(file, 'r')
		elif(isinstance(file, h5py.File)):
			data = file
		else:
			raise ValueError("Invalid input 'file' to get_spec_vecpot", file)

		lrad = data['input']['physics']['Lrad'][lvol]
		nvol = data['input']['physics']['Nvol'][0]
		geometry =  data['input']['physics']['Igeometry'][0]
		mpol =  data['input']['physics']['Mpol'][0]

		im = np.array(data['output']['im'][:])
		_in = np.array(data['output']['in'][:])
		mn = data['output']['mn'][0]
		mregular = data['input']['numerics']['Mregular'][0]

		lim1 = lvol * (lrad+1)
		lim2 = (lvol+1) * (lrad+1)
		Ate = data['vector_potential']['Ate'][:, lim1:lim2]
		Aze = data['vector_potential']['Aze'][:, lim1:lim2]
		Ato = data['vector_potential']['Ato'][:, lim1:lim2]
		Azo = data['vector_potential']['Azo'][:, lim1:lim2]

		if(isinstance(file, str)):
			data.close()

		sarr = np.linspace(-1, 1, 10) if sarr is None else sarr
		zarr = np.array([0]) if zarr is None else zarr
		tarr = np.linspace(0, 2*np.pi, 100) if tarr is None else tarr

		ns = len(sarr)
		nz = len(zarr)
		nt = len(tarr)

		Lsingularity = (lvol == 0) and (geometry != 1)

		T = SPECslab.get_spec_poly_basis(Lsingularity, mpol, lrad, sarr)

		factors = SPECslab.get_spec_regularization_factor(geometry, mn, im, lvol, sarr, mregular, 'F')

		cosa = np.cos(im[:,None,None] * tarr[None,:,None] - _in[:,None,None] * zarr[None,None,:]) # [j, it, iz]
		sina = np.sin(im[:,None,None] * tarr[None,:,None] - _in[:,None,None] * zarr[None,None,:]) # [j, it, iz]

		if(Lsingularity):
			At = np.zeros((ns, nt, nz))
			Az = np.zeros((ns, nt, nz))

			for j in range(mn):
				basis = T[:,0,im[j]+1]

				At += np.einsum('js,s,jltz->stz', factors[:,0,:], basis, Ate[:,:,None,None]*cosa[:,None,:,:]+ Ato[:,:,None,None]*sina[:,None,:,:], optimize=True)
				Az += np.einsum('js,s,jltz->stz', factors[:,0,:], basis, Aze[:,:,None,None]*cosa[:,None,:,:] + Azo[:,:,None,None]*sina[:,None,:,:], optimize=True)
		else:
			basis = T[:,0,:]

			At = np.einsum('js,ls,jltz->stz', factors[:,0,:], basis, Ate[:,:,None,None]*cosa[:,None,:,:] + Ato[:,:,None,None]*sina[:,None,:,:], optimize=True)
			Az = np.einsum('js,ls,jltz->stz', factors[:,0,:], basis, Aze[:,:,None,None]*cosa[:,None,:,:] + Azo[:,:,None,None]*sina[:,None,:,:], optimize=True)

		return At, Az

	def run_spec_master(fname, num_cpus=8, show_output=False, print_force=True, log_file=None):

		if(print_force):
			print(f"Running SPEC newton with {fname}")

		try:
			if(show_output):
				if(log_file is None):
					subprocess.run(f'mpirun -n {num_cpus} ~/SPEC/xspec {fname}', shell=True)
				else:
					subprocess.run(f'mpirun -n {num_cpus} ~/SPEC/xspec {fname}', shell=True, stdout=log_file, stderr=log_file)
			else:
				subprocess.run(f'mpirun -n {num_cpus} ~/SPEC/xspec {fname}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

			data = SPECout(fname+".h5")

			if(print_force):
				print(f"SPEC completed  ---  |f| = {data.output.ForceErr:.5e}")

			return True

		except Exception as e:
			print(f"Running SPEC failed!!! ({e})")
			return False


	def run_spec_descent(fname, num_cpus=8, show_output=False, log_file=None, print_force=True):

		if(print_force):
			print(f"Running SPEC descent with {fname}",)

		try:
			if(show_output):
				if(log_file is None):
					subprocess.run(f'mpirun -n {num_cpus} ~/spec_descent/SPEC/xspec {fname}', shell=True)
				else:
					subprocess.run(f'mpirun -n {num_cpus} ~/spec_descent/SPEC/xspec {fname}', shell=True, stdout=log_file, stderr=log_file)
			else:
				subprocess.run(f'mpirun -n {num_cpus} ~/spec_descent/SPEC/xspec {fname}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


			data = SPECout(fname+".h5")
			if(print_force):
				print(f"SPEC completed  ---  |f| = {data.output.ForceErr:.5e}")

			return True

		except Exception as e:
			print(f"Running SPEC failed!!! ({e})")
			return False

	def run_spec(f, **kwargs):
		nml=SPECNamelist(f)
		if(nml['globallist']['Lfindzero']==3):
			return SPECslab.run_spec_descent(f, **kwargs)
		else:
			return SPECslab.run_spec_master(f, **kwargs)


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
	
	def get_Bfield(file, theta=None, phi=None, r=None, num_radpts_pervol=10):

		if(isinstance(file, str)):
			data = SPECout(file)
		elif(isinstance(file, SPECout)):
			data = file

		if(r is None):
			np.linspace(0, 2*np.pi, 100, endpoint=True)
		elif(isinstance(r, int)):
			r = np.linspace(0, 2*np.pi, r, endpoint=True)

		if(isinstance(theta, float)):
			theta = np.array([theta])
		elif(theta is None):
			theta = np.array([np.pi])

		if(isinstance(phi, float)):
			phi = np.array([phi])
		elif(phi is None):
			phi = np.array([0])

		B = SPECslab.get_full_field(data, r, theta, phi, num_radpts_pervol)

		return B


	def check_profile_Bfield(fname, func_by0=None, func_bz0=None):
		
		fig, ax = plt.subplots(1, 1)
		plt.title('Plot of magnetic field B')
		plt.gcf().canvas.manager.set_window_title('Field B')

		data = SPECout(fname)

		theta = np.array([np.pi])
		phi = np.array([0])
		num_radpts = 300
		num_radpts_pervol = 10
		r = np.linspace(0, 2*np.pi, num_radpts, endpoint=True)

		B = SPECslab.get_full_field(data, r, theta, phi, num_radpts_pervol)

		ax.plot(r, B[0,:], '.-', picker=SPECslab.line_picker, label='spec B_psi')
		ax.plot(r, B[1,:], '.-', picker=SPECslab.line_picker, label='spec B_theta')
		ax.plot(r, B[2,:], '.-', picker=SPECslab.line_picker, label='spec B_ phi')

		if(func_by0 is not None):
			ax.plot(r, func_by0(r - np.pi), '--', picker=SPECslab.line_picker, label='B_y0')

		if(func_bz0 is not None):
			ax.plot(r, func_bz0(r-np.pi), '--', picker=SPECslab.line_picker, label='B_z0')

		ax.legend()
		ax.set_xlabel("Radial coordinate r / x [m]")
		ax.set_ylabel("Field strength [T]")

		fig.canvas.mpl_connect('pick_event', SPECslab.onpick2)
		fig.tight_layout()
	

	def check_profile_curr(fname, func_jy0, func_jz0):

		fig, ax = plt.subplots(1, 1)
		plt.title('Plot of current J')
		plt.gcf().canvas.manager.set_window_title('Current J')

		data = SPECout(fname)

		theta = [np.pi]
		phi = [0]
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
		plt.gcf().canvas.manager.set_window_title('Mu')

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
		fig.canvas.manager.set_window_title('Flux')

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
		plt.gcf().canvas.manager.set_window_title('Iota')

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
		inpdict.set_if_none('isurf', np.zeros(inpdict.Nvol))

		x_resolution = 2**14
		x_array = np.linspace(-np.pi, np.pi, x_resolution)

		ind_Lresinface, ind_Rresinface = SPECslab.infacesres_ind(x_array, inpdict.psi_w, inpdict.bz0)

		# fluxt_norm = SPECslab.norm_torflux(x_array, inpdict.bz0)
		# ind_Lresinface = np.argmin(np.abs(fluxt_norm - 0.5 * (1 - inpdict.psi_w)))
		# ind_Rresinface = np.argmin(np.abs(fluxt_norm - 0.5 * (1 + inpdict.psi_w)))
		#
		# infaces_ind, infaces_x, infaces_psi = SPECslab.gen_ifaces_funcspaced_new(x_array, ind_Lresinface, ind_Rresinface, inpdict.by0, inpdict.Nvol)

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

		# plt.figure()
		# plt.plot(x_array, mu_func(x_array),'.-')
		# print("mu_vol\n",mu_vol)

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
			inputnml['physicslist']['isurf'] = inpdict.isurf
			inputnml['physicslist']['ivolume'] = curr_vol[1:]#*1.03
			inputnml['physicslist']['tflux'] = tflux[1:]
			inputnml['physicslist']['pflux'] = pflux[1:] / phi_edge
		else:
			raise ValueError("Lconstraint has to be 0,1,2, or 3")

		# Make sure that the simple, direct solver is used
		# Bcs gmres will fail for mpol>3
		inputnml['locallist']['Lmatsolver'] = 1

		inputnml.write_simple(inpdict.fname_input)

		if(inpdict.lfindzero == 3):
			SPECslab.run_spec_descent(inpdict.fname_input, show_output=show_output, num_cpus=10, print_force=False)
		else:
			SPECslab.run_spec_master(inpdict.fname_input, show_output=show_output, num_cpus=10, print_force=False)


	def run_spec_asymm_slab(inpdict, show_output=False):

		if(not inpdict.has_key('r_up') or not inpdict.has_key('r_down')):
			raise ValueError("For creating SPEC run with asymmetric profile need to provide r_up and r_down")

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

		# make sure psi0(-pi) = 0
		psi0_leftbc = inpdict.psi0(-np.pi)
		if(np.abs(psi0_leftbc) > 1e-8):
			psi0_func = lambda x: inpdict.psi0(x) - psi0_leftbc
		else:
			psi0_func = inpdict.psi0

		ind_Lresinface = np.argmin(np.abs(x_array - inpdict.r_down))
		ind_Rresinface = np.argmin(np.abs(x_array - inpdict.r_up))

		if(inpdict.infaces_spacing == 'uniform'):
			infaces_ind, infaces_x, infaces_psi = SPECslab.gen_ifaces_funcspaced(x_array, ind_Lresinface, ind_Rresinface, lambda x: x, inpdict.Nvol) #uniform spacing
		elif(inpdict.infaces_spacing == 'psi'):
			infaces_ind, infaces_x, infaces_psi = SPECslab.gen_ifaces_funcspaced(x_array, ind_Lresinface, ind_Rresinface, psi0_func, inpdict.Nvol)
		else:
			raise ValueError("infaces_spacing has to be either uniform or psi")

		# # debugging stuff
		# plt.figure()
		# plt.plot(x_array, psi0_func(x_array),'d-')
		# plt.title("Psi_func")
		# for i in infaces_x:
		#     plt.axvline(i, color='red')
		# plt.show()

		# phi_edge = 2 * np.pi * inpdict.rslab * integrate.trapz(inpdict.bz0(x_array), x_array)
		# tflux = SPECslab.norm_torflux(x_array, inpdict.bz0)[infaces_ind]

		tflux = np.zeros(len(infaces_x))
		for t in range(len(infaces_x)):
			tflux[t] = integrate.quad(inpdict.bz0, -np.pi, infaces_x[t])[0]
		phi_edge = tflux[-1] * 2 * np.pi * inpdict.rslab
		tflux /= tflux[-1]

		pflux = psi0_func(infaces_x) * (2 * np.pi * inpdict.rslab)

		mu_vol = np.zeros(inpdict.Nvol)
		mu_func = lambda x: SPECslab.mu(x, inpdict.by0, inpdict.bz0, inpdict.jy0, inpdict.jz0)
		for t in range(0, inpdict.Nvol):
			mu_vol[t] = integrate.quad(mu_func, infaces_x[t], infaces_x[t+1])[0] / (infaces_x[t+1] - infaces_x[t])

		# plt.figure()
		# plt.plot(x_array, mu_func(x_array),'.-')
		# print("mu_vol\n",mu_vol)

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
				(psi0_func(x) - psi0_func(infaces_x[i])) * inpdict.bz0(x),
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
			SPECslab.check_psi(x_array, psi0_func)
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
			# SPECslab.check_psi(x_array, psi0_func)
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
		data = SPECout(fname)
		x = SPECslab.get_infaces_pts(data)
		insect_flag = check_intersect_helper(x, x.shape[0])
		return insect_flag

	def check_intersect_initial(inputnml, plot_flag=False, ax=None):
		infaces_fourier = inputnml.interface_guess
		thetas = np.linspace(0, 2*np.pi, 200)
		x = np.zeros((inputnml['physicslist']['Nvol'], len(thetas)))

		for i in range(inputnml['physicslist']['Mpol']+1):
			# print(np.array(infaces_fourier[(i,0)]['Rbc']))
			x += np.cos(i*thetas)[None,:] * (np.array(infaces_fourier[(i,0)]['Rbc']))[:,None]

		x = np.vstack([np.zeros_like(thetas), x])

		if(plot_flag):
			if(ax is None):
				fig, ax = plt.subplots()
			
			for vol in range(inputnml['physicslist']['Nvol']+1):
				ax.plot(thetas, x[vol])

		insect_flag = check_intersect_helper(x, x.shape[0])
		return insect_flag
	
	def get_infaces_initial(inputnml):
		infaces_fourier = inputnml.interface_guess
		thetas = np.linspace(0, 2*np.pi, 200)
		x = np.zeros((inputnml['physicslist']['Nvol'], len(thetas)))

		for i in range(inputnml['physicslist']['Mpol']+1):
			# print(np.array(infaces_fourier[(i,0)]['Rbc']))
			x += np.cos(i*thetas)[None,:] * (np.array(infaces_fourier[(i,0)]['Rbc']))[:,None]

		x = np.vstack([np.zeros_like(thetas), x])

		return x

	def get_infaces_pts(data, num_pts=200):
		thetas = np.linspace(0, 2*np.pi, num_pts)
		x_four = data.output.Rbc[:,:]
		x = np.zeros((x_four.shape[0], num_pts))
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

		# print(f"Del\' * a {SPECslab.calc_delprime(1/inpdict.rslab)*a:.3f} rslab {inpdict.rslab:.3f} psi_w {inpdict.psi_w:.3f}\n")

		try:
			SPECslab.get_spec_energy(inpdict.fname_outh5)
		except AttributeError:
			pass
		# SPECslab.plot_kamsurf(inpdict.fname_outh5, f"init kam surfaces psi_w={inpdict.psi_w:.3f}")

	def add_perturbation(input_fname, ouput_fname, kick_amplitude=0.8, max_num_iters=200, perturbation=None):

		# subprocess.run(f"cp {input_fname} {ouput_fname}", shell=True)
		inputnml = SPECNamelist(ouput_fname)        

		if(perturbation is None):
			eigval, eigvec, min_eigval_ind, min_eigvec = SPECslab.get_eigenstuff(input_fname)
			if(eigval[min_eigval_ind] > 0.0):
				print("Smallest eigenvalue of the force-gradient matrix is positive!")
				# return False

			perturbation = np.real(min_eigvec[1::inputnml._Mpol+1])
			perturbation /= np.max(np.abs(perturbation))
			perturbation *= -1*np.sign(perturbation[len(perturbation)//2])
			perturbation *= kick_amplitude

		# print("Finding good pertubation amplitude... ",end='')
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
		print("Eigenmode found ", n)

		inputnml.write_simple(ouput_fname)

		return perturbation

	def print_tree_h5(fname):
		with h5py.File(fname,'r') as hf:
			hf.visititems(print_attrs)

	def comapre_hdf5_files(fname1, fname2):

		with h5py.File(fname1) as f:
			f1_flat = {}
			f.visititems(lambda a, b: flatten_hdf5_file(a, b, f1_flat))

		with h5py.File(fname2) as f:
			f2_flat = {}
			f.visititems(lambda a, b: flatten_hdf5_file(a, b, f2_flat))

		for key in f1_flat.keys():

			if('iterations' in key):
				continue

			if(key not in f2_flat.keys()):
				raise KeyError("Two hdf5 files have different keys")

			min_len = min(len(f1_flat[key]), len(f2_flat[key]))
			same = np.allclose(f1_flat[key][:min_len], f2_flat[key][:min_len])

			if(not same):
				print(f"\n --> Value of {key} is different")
				range = max(len(f1_flat[key]), 15)
				with np.printoptions(threshold=200, edgeitems=10):
					print(f1_flat[key][:range])
					print(f2_flat[key][:range])

	def calc_asym_inface_pos(psi_w, r_s, delp, sigp, psi_string, psi_func, tflux_func):

		r_s_norm = (r_s + np.pi) / (2*np.pi)

		boldA = SPECslab.calc_boldA(psi_string, r_s)
		plusA = 0.5 * (sigp + delp)
		minusA = 0.5 * (sigp - delp)

		# def gr(r, constA):
		#     return (psi_func(r_s) - psi_func(r)) / (
		#         2 + boldA*(r - r_s)*np.log(np.abs(r-r_s)) + constA * (r - r_s) / np.pi )

		def gr(x, constA):
			r = (x + np.pi) / (2*np.pi)
			# print('r ',r)

			return (psi_func(r_s) - psi_func(x)) / (
				2 + boldA*(r - r_s_norm)*np.log(np.abs(r-r_s_norm)) + constA * (r - r_s_norm))

		def equation(x):
			# x[0] is r-, x[1] is r+
			eq1_psiw = (tflux_func(x[1]) - tflux_func(x[0])) / tflux_func(np.pi) - psi_w
			eq2_gr = gr(x[0], minusA) - gr(x[1], plusA)
			return [eq1_psiw, eq2_gr]

		## approach based on root-finding
		# root = optimize.fsolve(equation, [-0.1, 0.1], full_output=True)
		# print(root)
		# return root[0][0], root[0][1]

		## approach based on minimization (MORE ROBUST THAN ROOT-FINDING)
		minim = optimize.minimize(lambda x: np.sum(np.array(equation(x))**2), x0=(-0.1,0.1))
		# print(minim)

		return minim.x[0], minim.x[1]

	def calc_boldA(psi_string, r_s):

		x, psi0_sym = sym.symbols('x, psi0')

		psi_sym = sym.sympify(psi_string)
		by_sym = sym.diff(psi_sym, x)
		dby_sym = sym.diff(by_sym, x) # jz
		d2by_sym = sym.diff(dby_sym, x) # jz'

		boldA = d2by_sym.evalf(6, subs={x: r_s}) / dby_sym.evalf(6, subs={x: r_s})
		return boldA

	def run_slab_profile(inpdict, config, show_output=False):

		inpdict.fname_outh5 = inpdict.fname_input+'.h5'

		inpdict.psi0 = lambda x: config["psi"](x, inpdict.psi0_mag)
		inpdict.by0 = lambda x: config["by"](x, inpdict.psi0_mag)
		inpdict.bz0 = lambda x: config["bz"](x, inpdict.psi0_mag, inpdict.bz0_mag)
		inpdict.jy0 = lambda x: config["jy"](x, inpdict.psi0_mag, inpdict.bz0_mag)
		inpdict.jz0 = lambda x: config["jz"](x, inpdict.psi0_mag)

		SPECslab.run_spec_slab(inpdict, show_output)

	def find_max_psiw_symm(delprimeas, psiw_lims, constraint=2, print_debug=False):
		# finds max psiw for each delprimea

		if(isinstance(delprimeas, float)):
			delprimeas = [delprimeas]
			psiw_lims = [psiw_lims]

		inpdict = input_dict()
		inpdict.psi0_mag = 3 * np.sqrt(3) / 4
		inpdict.bz0_mag = 10.0
		inpdict.linitialize = 0
		inpdict.lbeltrami = 4
		inpdict.lfindzero = 2
		# inpdict.infaces_spacing = 'psi'
		inpdict.infaces_spacing = 'uniform'
		inpdict.pre = np.zeros(inpdict.Nvol)
		inpdict.lconstraint = constraint
		inpdict.fname_template = 'template.sp'
		inpdict.fname_input = 'stability_analysis.sp'
		inpdict.Nvol = 21
		inpdict.Mpol = 1 # min eigval independent of mpol (so use smallest mpol=1)

		profile = f'psi0 * (1 / cosh(x)**2 - 1/cosh(-pi)**2)' ## PROFILES
		inpdict.config = SPECslab.gen_profiles_from_psi(profile)
		inpdict.lamba_val = 1e10

		def psiw_func(psiw, delprimea, inpdict):

			inpdict.psi_w = psiw
			inpdict.rslab = SPECslab.calc_rslab_for_delprime(delprimea/0.35)

			with open(os.devnull, 'w') as devnull:
				with contextlib.redirect_stdout(devnull):
					SPECslab.run_slab_profile(inpdict, inpdict.config, False)

			eigval, eigvec, min_eigval_ind, min_eigvec = SPECslab.get_eigenstuff(inpdict.fname_input)
			min_eigval = eigval[min_eigval_ind]
			inpdict.lamba_val = min_eigval

			if(print_debug):
				print(f"psiw {psiw:.4f} min_eigval {min_eigval:.5e}")
			# return -psiw if min_eigval < 0 else 1.0
			return -psiw if min_eigval < 0 else psiw

		best_psiws = []
		for d in range(len(delprimeas)):

			if(psiw_func(psiw_lims[d][0], delprimeas[d], inpdict) > 0):
				raise ValueError("Error in psiw search. Need left bound for psiw to have lambda<0!")

			res = minimize_scalar(lambda x: psiw_func(x, delprimeas[d], inpdict),
								  bounds=(psiw_lims[d][0], psiw_lims[d][1]), method='bounded',
								  options={'xatol': 1e-5, 'maxiter': 30, 'disp': 0}) ## dafulat xatol: 1e-6
			best_psiws.append(res.x)

			print(f"Final lambda {inpdict.lamba_val} psiw {best_psiws[-1]} delprimea {delprimeas[d]}")

		return np.array(best_psiws) if len(best_psiws) > 1 else best_psiws[0]

	@contextlib.contextmanager
	def change_dir(path):
		old_path = os.getcwd()
		
		if(not os.path.exists(path)):
			os.mkdir(path)

		os.chdir(path)

		try:
			yield
		finally:
			os.chdir(old_path)

@numba.njit(cache=True)
def check_intersect_helper(x, nvol):
	for v in range(nvol):
		for sv in range(v-1, v+1):
			diff = x[v] - x[sv]
			for i in range(len(diff)-1):
				if((diff[i]*diff[i+1]) < 0):
					return True
	return False


def print_attrs(name, obj):
	shift = (name.count('/')) * '\t'
	item_name = name.split("/")[-1]

	if(isinstance(obj, h5py.Group)):
		print(shift, '--'*10)
		print(shift, 'GROUP ', item_name)
	try:
		for key, val in obj.items():
			if(val.shape != (1,)):
				print(shift + '\t' + f"{key}: {val.shape} \t\n{val[:]}")
			else:
				print(shift + '\t' + f"{key}: {val[0]}")
	except:
		pass

def flatten_hdf5_file(name, node, file_dict):
	fullname = node.name

	if isinstance(node, h5py.Dataset):
		file_dict[fullname] = node[:]
	else:
		pass


def vecpot_to_contrav(T, Ate, Aze, Ato, Azo, jac, im, _in, tarr, zarr):

		basis = T[:,0,:]
		dbasis = T[:,1,:]

		mn = Azo.shape[0]
		lrad = Azo.shape[1] # actually is lrad+1
		nt = tarr.shape[0]
		nz = zarr.shape[0]
		nr = basis.shape[1]

		# cosa = np.cos(im[:,None,None] * tarr[None,:,None] - _in[:,None,None] * zarr[None,None,:])
		# sina = np.sin(im[:,None,None] * tarr[None,:,None] - _in[:,None,None] * zarr[None,None,:]) # [j, it, iz]

		# if(Lsingularity):
		#     Bs = np.zeros((ns, nt, nz))
		#     Bt = np.zeros((ns, nt, nz))
		#     Bz = np.zeros((ns, nt, nz))

		#     for j in range(mn):
		#         basis = T[:,0,im[j]+1]
		#         dbasis = T[:,1,im[j]+1]

		#         Bs_term = (im[:,None]*Azo[:,:] + _in[:,None]*Ato[:,:])[:,:,None,None] * cosa[:,None,:,:] - (im[:,None]*Aze[:,:] + _in[:,None]*Ate[:,:])[:,:,None,None] * sina[:,None,:,:]
		#         Bs += np.einsum('s,jltz->stz', basis, Bs_term, optimize=True)
		#         Bt += np.einsum('s,jltz->stz', -dbasis, Aze[:,:,None,None]*cosa[:,None,:,:] + Azo[:,:,None,None]*sina[:,None,:,:], optimize=True)
		#         Bz += np.einsum('s,jltz->stz', dbasis, Ate[:,:,None,None]*cosa[:,None,:,:] + Ato[:,:,None,None]*sina[:,None,:,:], optimize=True)
		# else:
			# basis = T[:,0,:]
			# dbasis = T[:,1,:]
			# Bs_term = (im[:,None]*Azo[:,:] + _in[:,None]*Ato[:,:])[:,:,None,None] * cosa[:,None,:,:] - (im[:,None]*Aze[:,:] + _in[:,None]*Ate[:,:])[:,:,None,None] * sina[:,None,:,:]
			
			# print("shapes", basis.shape, Bs_term.shape)
			# print(Azo.shape, Ate.shape)

			# Bs = np.einsum('ls,jtzl->stz', basis, Bs_term, optimize=True)
			# Bt = np.einsum('ls,jltz->stz', -dbasis, Aze[:,:,None,None]*cosa[:,None,:,:] + Azo[:,:,None,None]*sina[:,None,:,:], optimize=True)
			# Bz = np.einsum('ls,jltz->stz', dbasis, Ate[:,:,None,None]*cosa[:,None,:,:] + Ato[:,:,None,None]*sina[:,None,:,:], optimize=True)
		

		# should use np.zeros, not np.empty
		Bs = np.zeros((nr, nt, nz))
		Bt = np.zeros_like(Bs)
		Bz = np.zeros_like(Bs)

		# print(Aze.shape, dbasis.shape)

		for j in range(mn):
			for t in range(nt):
				for z in range(nz):
					alpha = im[j]*tarr[t] - _in[j]*zarr[z]
					cosa = np.cos(alpha)
					sina = np.sin(alpha)
						
					for l in range(lrad):
						Bs_term = (im[j]*Azo[j,l] + _in[j]*Ato[j,l]) * cosa - (im[j]*Aze[j,l] + _in[j]*Ate[j,l]) * sina
						
						for s in range(nr):
							Bs[s, t, z] += (basis[l,s] * Bs_term) / jac[s,t,z]
							Bt[s, t, z] += (-dbasis[l,s] * (Aze[j,l]*cosa + Azo[j,l]*sina)) / jac[s,t,z]
							Bz[s, t, z] += (dbasis[l,s] * (Ate[j,l]*cosa + Ato[j,l]*sina)) / jac[s,t,z]

		# Bs_term = (im[:,None]*Azo[:,:] + _in[:,None]*Ato[:,:])[:,:,None,None] * cosa[:,None,:,:] - (im[:,None]*Aze[:,:] + _in[:,None]*Ate[:,:])[:,:,None,None] * sina[:,None,:,:]
		
		# print("shapes", basis.shape, Bs_term.shape)
		# print(Azo.shape, Ate.shape)

		# Bs = np.einsum('ls,jtzl->stz', basis, Bs_term, optimize=True)
		# Bt = np.einsum('ls,jltz->stz', -dbasis, Aze[:,:,None,None]*cosa[:,None,:,:] + Azo[:,:,None,None]*sina[:,None,:,:], optimize=True)
		# Bz = np.einsum('ls,jltz->stz', dbasis, Ate[:,:,None,None]*cosa[:,None,:,:] + Ato[:,:,None,None]*sina[:,None,:,:], optimize=True)
		
		# print('bt', Bz[:,0,0])

		return np.array([Bs, Bt, Bz])