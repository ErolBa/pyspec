#!/usr/bin/env python

## Test custom implementation of the metric and g, against existing implementation
from py_spec import SPECout
import numpy as np

if __name__ == '__main__':

	atol = 1e-4
	rtol = 1e-9

	print("Testing implementation of metric and jacobian...")

	print("Slab geometry 1")
	o1 = SPECout('test_g1.sp.h5')

	sarr = np.linspace(-0.99, 1, 24)
	tarr = np.array([0., 1.2, 5.0])
	zarr = np.array([0., 0.3, 4.0])
	for lvol in [0, 17, 40]:
		print(f'lvol {lvol}')
		_, _, jac_pyo, g_pyo = o1.get_grid_and_jacobian_and_metric(lvol, sarr, tarr, zarr, False, False)
		g_pyo = np.moveaxis(g_pyo, -1, 0)
		g_pyo = np.moveaxis(g_pyo, -1, 0)

		g_cust = o1.get_metric(sarr, tarr, zarr, lvol)
		jac_cust = o1.get_jacobian(sarr, tarr, zarr, lvol)

		same = np.allclose(g_pyo, g_cust, atol=atol, rtol=rtol)
		if(not same):
			raise ValueError(f"Slab geometry 1: metric not the same for lvol={lvol}")

		same = np.allclose(jac_pyo, jac_cust, atol=atol, rtol=rtol)
		if(not same):
			raise ValueError(f"Slab geometry 1: jacobian not the same for lvol={lvol}")  
	
	
	print("Cylindrical geometry 2")
	o2 = SPECout('test_g2.sp.h5')

	sarr = np.linspace(-0.99, 1, 24)
	tarr = np.array([0., 1.2, 5.0])
	zarr = np.array([0., 0.3, 4.0])
	for lvol in [0, 7]:
		print(f'lvol {lvol}')
		_, _, jac_pyo, g_pyo = o2.get_grid_and_jacobian_and_metric(lvol, sarr, tarr, zarr, False, False)
		g_pyo = np.moveaxis(g_pyo, -1, 0)
		g_pyo = np.moveaxis(g_pyo, -1, 0)
		
		g_cust = o2.get_metric(sarr, tarr, zarr, lvol)
		jac_cust = o2.get_jacobian(sarr, tarr, zarr, lvol)

		same = np.allclose(g_pyo, g_cust, atol=atol, rtol=rtol)
		if(not same):
			raise ValueError(f"Cylindrical geometry 1: metric not the same for lvol={lvol}")

		same = np.allclose(jac_pyo, jac_cust, atol=atol, rtol=rtol)
		if(not same):
			raise ValueError(f"Cylindrical geometry 2: jacobian not the same for lvol={lvol}")  


	print("Toroidal geomoetry 3")
	o3 = SPECout('test_g3.sp.h5')

	sarr = np.linspace(-0.99, 1, 24)
	tarr = np.array([0., 1.2, 5.0])
	zarr = np.array([0., 0.3, 4.0])
	for lvol in [0, 4]:
		print(f'lvol {lvol}')
		_, _, jac_pyo, g_pyo = o3.get_grid_and_jacobian_and_metric(lvol, sarr, tarr, zarr, False, False)
		g_pyo = np.moveaxis(g_pyo, -1, 0)
		g_pyo = np.moveaxis(g_pyo, -1, 0)
		
		g_cust = o3.get_metric(sarr, tarr, zarr, lvol)
		jac_cust = o3.get_jacobian(sarr, tarr, zarr, lvol)

		same = np.allclose(g_pyo, g_cust, atol=atol, rtol=rtol)
		if(not same):
			raise ValueError(f"Toroidal geometry 3: metric not the same for lvol={lvol}")

		same = np.allclose(jac_pyo, jac_cust, atol=atol, rtol=rtol)
		if(not same):
			raise ValueError(f"Toroidal geometry 3: jacobian not the same for lvol={lvol}")   
		
	print("All tests passed!")