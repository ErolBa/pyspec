#!/usr/bin/env python

## Test custom implementation of B field calculation, against pyoculus implementation
from py_spec import SPECout
import numpy as np

if __name__ == '__main__':

    atol = 1e-4
    rtol = 1e-9

    print("Testing cov. field implementation...")

    print("Slab geometry 1")
    o1 = SPECout('test_g1.sp.h5')

    sarr = np.linspace(-0.99, 1, 24)
    tarr = np.array([0., 1.2, 5.0])
    zarr = np.array([0., 0.3, 4.0])
    for lvol in [0, 17, 40]:
        print(f'lvol {lvol}')
        b_pyo = o1.get_B(lvol, None, sarr, tarr, zarr, False, False)
        b_pyo = b_pyo.swapaxes(0,3).swapaxes(1,3).swapaxes(2,3)
        b_cust = o1.get_field_contrav(lvol, sarr, tarr, zarr)

        same = np.allclose(b_cust, b_pyo, atol=atol, rtol=rtol)

        if(not same):
            raise ValueError(f"Slab geometry 1: B field not the same for lvol={lvol}")  
    
    print("Cylindrical geometry 2")
    o2 = SPECout('test_g2.sp.h5')

    sarr = np.linspace(-0.99, 1, 24)
    tarr = np.array([0., 1.2, 5.0])
    zarr = np.array([0., 0.3, 4.0])
    for lvol in [0, 7]:
        print(f'lvol {lvol}')
        b_pyo = o2.get_B(lvol, None, sarr, tarr, zarr, False, False)
        b_pyo = b_pyo.swapaxes(0,3).swapaxes(1,3).swapaxes(2,3)
        b_cust = o2.get_field_contrav(lvol, sarr, tarr, zarr)

        same = np.allclose(b_cust, b_pyo, atol=atol, rtol=rtol)

        if(not same):
            raise ValueError(f"Cylindrical geometry 2: B field not the same for lvol={lvol}") 


    print("Toroidal geomoetry 3")
    o3 = SPECout('test_g3.sp.h5')

    sarr = np.linspace(-0.99, 1, 24)
    tarr = np.array([0., 1.2, 5.0])
    zarr = np.array([0., 0.3, 4.0])
    for lvol in [0, 4]:
        print(f'lvol {lvol}')
        b_pyo = o3.get_B(lvol, None, sarr, tarr, zarr, False, False)
        b_pyo = b_pyo.swapaxes(0,3).swapaxes(1,3).swapaxes(2,3)
        b_cust = o3.get_field_contrav(lvol, sarr, tarr, zarr)

        same = np.allclose(b_cust, b_pyo, atol=atol, rtol=rtol)

        if(not same):
            raise ValueError(f"Toroidal geometry 3: B field not the same for lvol={lvol}") 
        
    print("All tests passed!")