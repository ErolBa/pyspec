import numpy as np
from py_spec import SPECout

atol = 1e-4
rtol = 1e-9

## Contravariant field tests

def test_field_contrav_slab_g1():
    o1 = SPECout('testing/test_g1.sp.h5')
    sarr = np.linspace(-0.99, 1, 24)
    tarr = np.array([0., 1.2, 5.0])
    zarr = np.array([0., 0.3, 4.0])
    for lvol in [0, 17, 40]:
        b_pyo = o1.get_B(lvol, None, sarr, tarr, zarr, False, False)
        b_pyo = b_pyo.swapaxes(0,3).swapaxes(1,3).swapaxes(2,3)
        b_cust = o1.get_field_contrav(lvol, sarr, tarr, zarr)
        assert np.allclose(b_cust, b_pyo, atol=atol, rtol=rtol)

def test_field_contrav_cylindrical_g2():
    o2 = SPECout('testing/test_g2.sp.h5')
    sarr = np.linspace(-0.99, 1, 24)
    tarr = np.array([0., 1.2, 5.0])
    zarr = np.array([0., 0.3, 4.0])
    for lvol in [0, 7]:
        b_pyo = o2.get_B(lvol, None, sarr, tarr, zarr, False, False)
        b_pyo = b_pyo.swapaxes(0,3).swapaxes(1,3).swapaxes(2,3)
        b_cust = o2.get_field_contrav(lvol, sarr, tarr, zarr)
        assert np.allclose(b_cust, b_pyo, atol=atol, rtol=rtol)

def test_field_contrav_toroidal_g3():
    o3 = SPECout('testing/test_g3.sp.h5')
    sarr = np.linspace(-0.99, 1, 24)
    tarr = np.array([0., 1.2, 5.0])
    zarr = np.array([0., 0.3, 4.0])
    for lvol in [0, 1, 2]:
        b_pyo = o3.get_B(lvol, None, sarr, tarr, zarr, False, False)
        b_pyo = b_pyo.swapaxes(0,3).swapaxes(1,3).swapaxes(2,3)
        b_cust = o3.get_field_contrav(lvol, sarr, tarr, zarr)
        assert np.allclose(b_cust, b_pyo, atol=atol, rtol=rtol)

## modB tests

def test_modB_slab_g1():
    o = SPECout('testing/test_g1.sp.h5')
    sarr = np.linspace(-0.99, 1, 24)
    tarr = np.array([0., 1.2, 5.0])
    zarr = np.array([0., 0.3, 4.0])
    for lvol in [0, 17, 40]:
        Bcontrav = o.get_B(lvol, None, sarr, tarr, zarr, False, False)
        _,_,_, g = o.get_grid_and_jacobian_and_metric(lvol, sarr, tarr, zarr)
        Bmod = o.get_modB(Bcontrav, g)
        Fmod = o.get_field_mod(lvol, sarr, tarr, zarr)
        assert np.allclose(Bmod, Fmod, atol=atol, rtol=rtol), f"Field mod not the same as B mod for lvol={lvol}"

def test_modB_cylindrical_g2():
    o = SPECout('testing/test_g2.sp.h5')
    sarr = np.linspace(-0.99, 1, 24)
    tarr = np.array([0., 1.2, 5.0])
    zarr = np.array([0., 0.3, 4.0])
    for lvol in [0, 7]:
        Bcontrav = o.get_B(lvol, None, sarr, tarr, zarr, False, False)
        _,_,_, g = o.get_grid_and_jacobian_and_metric(lvol, sarr, tarr, zarr)
        Bmod = o.get_modB(Bcontrav, g)
        Fmod = o.get_field_mod(lvol, sarr, tarr, zarr)
        assert np.allclose(Bmod, Fmod, atol=atol, rtol=rtol), f"Field mod not the same as B mod for lvol={lvol}"

def test_modB_toroidal_g3():
    o = SPECout('testing/test_g3.sp.h5')
    sarr = np.linspace(-0.99, 1, 24)
    tarr = np.array([0., 1.2, 5.0])
    zarr = np.array([0., 0.3, 4.0])
    for lvol in [0, 1, 2]:
        Bcontrav = o.get_B(lvol, None, sarr, tarr, zarr, False, False)
        _,_,_, g = o.get_grid_and_jacobian_and_metric(lvol, sarr, tarr, zarr)
        Bmod = o.get_modB(Bcontrav, g)
        Fmod = o.get_field_mod(lvol, sarr, tarr, zarr)
        assert np.allclose(Bmod, Fmod, atol=atol, rtol=rtol), f"Field mod not the same as B mod for lvol={lvol}"
