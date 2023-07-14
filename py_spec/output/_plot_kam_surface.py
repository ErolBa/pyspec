import numpy as np
import matplotlib.pyplot as plt

## Class copied from coilpy
class FourSurf(object):
    """
    toroidal surface in Fourier representation
    R = \sum RBC cos(mu-nv) + RBS sin(mu-nv)
    Z = \sum ZBC cos(mu-nv) + ZBS sin(mu-nv)
    """

    def __init__(self, xm=[], xn=[], rbc=[], zbs=[], rbs=[], zbc=[]):
        """Initialization with Fourier harmonics.
        Parameters:
          xm -- list or numpy array, array of m index (default: [])
          xn -- list or numpy array, array of n index (default: [])
          rbc -- list or numpy array, array of radial cosine harmonics (default: [])
          zbs -- list or numpy array, array of z sine harmonics (default: [])
          rbs -- list or numpy array, array of radial sine harmonics (default: [])
          zbc -- list or numpy array, array of z cosine harmonics (default: [])
        """
        self.xm = np.atleast_1d(xm)
        self.xn = np.atleast_1d(xn)
        self.rbc = np.atleast_1d(rbc)
        self.rbs = np.atleast_1d(rbs)
        self.zbc = np.atleast_1d(zbc)
        self.zbs = np.atleast_1d(zbs)
        self.mn = len(self.xn)
        return

    @classmethod
    def read_spec_output(cls, spec_out, ns=-1):
            """initialize surface from the ns-th interface SPEC output
            Parameters:
              spec_out -- SPEC class, SPEC hdf5 results
              ns -- integer, the index of SPEC interface (default: -1)
            Returns:
              fourier_surface class
            """
            # check if spec_out is in correct format
            # if not isinstance(spec_out, SPEC):
            #    raise TypeError("Invalid type of input data, should be SPEC type.")
            # get required data
            xm = spec_out.output.im
            xn = spec_out.output.in_
            rbc = spec_out.output.Rbc[ns, :]
            zbs = spec_out.output.Zbs[ns, :]
            if spec_out.input.physics.Istellsym:
                # stellarator symmetry enforced
                rbs = np.zeros_like(rbc)
                zbc = np.zeros_like(rbc)
            else:
                rbs = spec_out.output.Rbs[ns, :]
                zbc = spec_out.output.Zbc[ns, :]
            return cls(xm=xm, xn=xn, rbc=rbc, rbs=rbs, zbc=zbc, zbs=zbs)

    def rz(self, theta, zeta, normal=False):
        """get r,z position of list of (theta, zeta)
        Parameters:
          theta -- float array_like, poloidal angle
          zeta -- float array_like, toroidal angle value
          normal -- logical, calculate the normal vector or not (default: False)
        Returns:
           r, z -- float array_like
           r, z, [rt, zt], [rz, zz] -- if normal
        """
        assert len(np.atleast_1d(theta)) == len(
            np.atleast_1d(zeta)
        ), "theta, zeta should be equal size"
        # mt - nz (in matrix)
        _mtnz = np.matmul(
            np.reshape(self.xm, (-1, 1)), np.reshape(theta, (1, -1))
        ) - np.matmul(np.reshape(self.xn, (-1, 1)), np.reshape(zeta, (1, -1)))
        _cos = np.cos(_mtnz)
        _sin = np.sin(_mtnz)
        r = np.matmul(np.reshape(self.rbc, (1, -1)), _cos) + np.matmul(
            np.reshape(self.rbs, (1, -1)), _sin
        )
        z = np.matmul(np.reshape(self.zbc, (1, -1)), _cos) + np.matmul(
            np.reshape(self.zbs, (1, -1)), _sin
        )
        #if(isinstance(zeta, float)):
        #    print('zbs', self.zbs)
        #    print('sin', _sin)
        if not normal:
            return (r.ravel(), z.ravel())
        else:
            rt = np.matmul(np.reshape(self.xm * self.rbc, (1, -1)), -_sin) + np.matmul(
                np.reshape(self.xm * self.rbs, (1, -1)), _cos
            )
            zt = np.matmul(np.reshape(self.xm * self.zbc, (1, -1)), -_sin) + np.matmul(
                np.reshape(self.xm * self.zbs, (1, -1)), _cos
            )

            rz = np.matmul(np.reshape(-self.xn * self.rbc, (1, -1)), -_sin) + np.matmul(
                np.reshape(-self.xn * self.rbs, (1, -1)), _cos
            )
            zz = np.matmul(np.reshape(-self.xn * self.zbc, (1, -1)), -_sin) + np.matmul(
                np.reshape(-self.xn * self.zbs, (1, -1)), _cos
            )
            return (
                r.ravel(),
                z.ravel(),
                [rt.ravel(), zt.ravel()],
                [rz.ravel(), zz.ravel()],
            )

    def plot(self, zeta=0.0, npoints=700, **kwargs):
        """plot the cross-section at zeta using matplotlib.pyplot
        Parameters:
          zeta -- float, toroidal angle value
          npoints -- integer, number of discretization points (default: 360)
          kwargs -- optional keyword arguments for pyplot
        Returns:
           line class in matplotlib.pyplot
        """

        # get figure and ax data
        if plt.get_fignums():
            fig = plt.gcf()
            ax = plt.gca()
        else:
            fig, ax = plt.subplots()
        # set default plotting parameters
        if kwargs.get("linewidth") == None:
            kwargs.update({"linewidth": 2.0})  # prefer thicker lines
        if kwargs.get("label") == None:
            kwargs.update({"label": "toroidal surface"})  # default label
        # get (r,z) data
        _r, _z = self.rz(np.linspace(0, 2 * np.pi, npoints), zeta * np.ones(npoints))
        line = ax.plot(_r, _z, **kwargs)
        #plt.axis("equal")
        plt.xlabel("R [m]", fontsize=20)
        plt.ylabel("Z [m]", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        return line


def plot_kam_surface(self, ns=[], ntheta=1000, zeta=0.0, ax=None, **kwargs):
    """Plot SPEC KAM surfaces

    Args:
        ns (list, optional): List of surface index to be plotted (0 for axis, -1 for the computational boundary if applied).
                             Defaults to [] (plot all).
        zeta (float, optional): The toroidal angle where the cross-sections are plotted. Defaults to 0.0.
        ax (Matplotlib axis, optional): Matplotlib axis to be plotted on. Defaults to None.
        kwargs (dict, optional): Keyword arguments. Matplotlib.pyplot.plot keyword arguments
    Returns:
        list : list of FourSurf classes
    """
    import numpy as np
    import matplotlib.pyplot as plt

    Igeometry = self.input.physics.Igeometry
    # from coilpy import FourSurf

    surfs = []
    # check if plot all
    if len(ns) == 0:
        # 0 for the axis
        ns = np.arange(self.input.physics.Nvol + self.input.physics.Lfreebound + 1)
    else:
        ns = np.atleast_1d(ns)
    # get axix data
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    # set default plotting parameters
    if kwargs.get("label") == None:
        kwargs.update({"label": "SPEC_KAM"})  # default label
    if kwargs.get("c") == None:
        kwargs.update({"c": "red"})
    # plot all the surfaces
    if Igeometry == 3:

        for i in ns:
            _surf = FourSurf.read_spec_output(self, i)
            if i == 0:
                # plot axis as a curve
                _r, _z = _surf.rz(0.0, zeta)
                plt.scatter(_r, _z, **kwargs)
            else:
                _surf.plot(zeta=zeta, **kwargs)
            # surfs.append(_surf)
        #plt.axis("equal")
        return surfs
    elif Igeometry == 2:
        for i in ns:
            _surf = FourSurf.read_spec_output(self, i)
            if i == 0:
                pass  # don't do anything for the axis
            else:
                _theta = np.arange(
                    0, 2 * np.pi + 2 * np.pi / ntheta, 2 * np.pi / ntheta
                )
                _r, _z = _surf.rz(_theta, np.ones_like(_theta) * zeta)
                plt.scatter(_r * np.cos(_theta), _r * np.sin(_theta), **kwargs)
            # surfs.append(_surf)
        #plt.axis("equal")
        return surfs
    elif Igeometry == 1:
        for i in ns:
            _surf = FourSurf.read_spec_output(self, i)
            # plot axis as a curve
            _theta = np.arange(0, 2 * np.pi + 2 * np.pi / ntheta, 2 * np.pi / ntheta)
            _r, _z = _surf.rz(_theta, np.ones_like(_theta) * zeta)
            plt.plot(_theta, _r, **kwargs)
        #     surfs.append(_surf)
        # return surfs
