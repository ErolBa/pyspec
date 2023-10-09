import numpy as np
from py_spec import SPECout

atol = 1e-4
rtol = 1e-9

def test_metric_slab_g1():
    o1 = SPECout('testing/test_g1.sp.h5')
    sarr = np.linspace(-0.99, 1, 24)
    tarr = np.array([0., 1.2, 5.0])
    zarr = np.array([0., 0.3, 4.0])
    for lvol in [0, 17, 40]:
        g_pyo = o1.get_grid_and_jacobian_and_metric(lvol, sarr, tarr, zarr, False, False)[3]
        g_pyo = np.moveaxis(g_pyo, -1, 0)
        g_pyo = np.moveaxis(g_pyo, -1, 0)
        g_cust = o1.get_metric(sarr, tarr, zarr, lvol)
        assert np.allclose(g_pyo, g_cust, atol=atol, rtol=rtol), f"Slab geometry 1: metric not the same for lvol={lvol}"

def test_jacobian_slab_g1():
    o1 = SPECout('testing/test_g1.sp.h5')
    sarr = np.linspace(-0.99, 1, 24)
    tarr = np.array([0., 1.2, 5.0])
    zarr = np.array([0., 0.3, 4.0])
    for lvol in [0, 17, 40]:
        jac_pyo = o1.get_grid_and_jacobian_and_metric(lvol, sarr, tarr, zarr, False, False)[2]
        jac_cust = o1.get_jacobian(sarr, tarr, zarr, lvol)
        assert np.allclose(jac_pyo, jac_cust, atol=atol, rtol=rtol), f"Slab geometry 1: jacobian not the same for lvol={lvol}"

def test_metric_cylindrical_g2():
    o2 = SPECout('testing/test_g2.sp.h5')
    sarr = np.linspace(-0.99, 1, 24)
    tarr = np.array([0., 1.2, 5.0])
    zarr = np.array([0., 0.3, 4.0])
    for lvol in [0, 7]:
        g_pyo = o2.get_grid_and_jacobian_and_metric(lvol, sarr, tarr, zarr, False, False)[3]
        g_pyo = np.moveaxis(g_pyo, -1, 0)
        g_pyo = np.moveaxis(g_pyo, -1, 0)
        g_cust = o2.get_metric(sarr, tarr, zarr, lvol)
        assert np.allclose(g_pyo, g_cust, atol=atol, rtol=rtol), f"Cylindrical geometry 2: metric not the same for lvol={lvol}"

def test_jacobian_cylindrical_g2():
    o2 = SPECout('testing/test_g2.sp.h5')
    sarr = np.linspace(-0.99, 1, 24)
    tarr = np.array([0., 1.2, 5.0])
    zarr = np.array([0., 0.3, 4.0])
    for lvol in [0, 7]:
        jac_pyo = o2.get_grid_and_jacobian_and_metric(lvol, sarr, tarr, zarr, False, False)[2]
        jac_cust = o2.get_jacobian(sarr, tarr, zarr, lvol)
        assert np.allclose(jac_pyo, jac_cust, atol=atol, rtol=rtol), f"Cylindrical geometry 2: jacobian not the same for lvol={lvol}"

def test_metric_toroidal_g3():
    o3 = SPECout('testing/test_g3.sp.h5')
    sarr = np.linspace(-0.99, 1, 24)
    tarr = np.array([0., 1.2, 5.0])
    zarr = np.array([0., 0.3, 4.0])
    for lvol in [0, 4]:
        g_pyo = o3.get_grid_and_jacobian_and_metric(lvol, sarr, tarr, zarr, False, False)[3]
        g_pyo = np.moveaxis(g_pyo, -1, 0)
        g_pyo = np.moveaxis(g_pyo, -1, 0)
        g_cust = o3.get_metric(sarr, tarr, zarr, lvol)
        assert np.allclose(g_pyo, g_cust, atol=atol, rtol=rtol), f"Toroidal geometry 3: metric not the same for lvol={lvol}"

def test_jacobian_toroidal_g3():
    o3 = SPECout('testing/test_g3.sp.h5')
    sarr = np.linspace(-0.99, 1, 24)
    tarr = np.array([0., 1.2, 5.0])
    zarr = np.array([0., 0.3, 4.0])
    for lvol in [0, 4]:
        jac_pyo = o3.get_grid_and_jacobian_and_metric(lvol, sarr, tarr, zarr, False, False)[2]
        jac_cust = o3.get_jacobian(sarr, tarr, zarr, lvol)
        assert np.allclose(jac_pyo, jac_cust, atol=atol, rtol=rtol), f"Toroidal geometry 3: jacobian not the same for lvol={lvol}"