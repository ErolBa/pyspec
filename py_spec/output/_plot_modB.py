import numpy as np


def plot_modB(
    self,
    sarr=np.linspace(-0.999, 1, 4),
    tarr=np.linspace(0, 2 * np.pi, 5),
    zarr=np.linspace(0, 0, 2),
    lvol=None,
    ax=None,
    colorbar=True,
    **kwargs,
):
    """[summary]

    Args:
        sarr ([type], optional): [description]. Defaults to np.linspace(-0.999, 1, 4).
        tarr ([type], optional): [description]. Defaults to np.linspace(0, 2 * np.pi, 5).
        zarr ([type], optional): [description]. Defaults to np.linspace(0, 0, 2).
        lvol (int, optional): [description]. Defaults to 0.
        ax (Matplotlib axis, optional): Matplotlib axis to be plotted on. Defaults to None.
    """
    import matplotlib.pyplot as plt

    Nvol = self.input.physics.Nvol
    if lvol == None:
        lvollist = np.arange(0, Nvol, dtype=int).tolist()
    elif np.isscalar(lvol):
        lvollist = [lvol]
    else:
        lvollist = lvol
    # get axis data
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)

    #nr = sarr.size
    nr = len(sarr)
    #nt = tarr.size
    nt = len(tarr)

    plotR = np.zeros([len(lvollist) * nr, nt], dtype=np.float64)
    plotZ = np.zeros([len(lvollist) * nr, nt], dtype=np.float64)
    plotB = np.zeros([len(lvollist) * nr, nt], dtype=np.float64)

    for i, ivol in enumerate(lvollist):
        # parse data
        R, Z, jacobian, g = self.get_grid_and_jacobian_and_metric(
            lvol=ivol, sarr=sarr, tarr=tarr, zarr=zarr
        )
        Bcontrav = self.get_B(
            lvol=ivol, jacobian=jacobian, sarr=sarr, tarr=tarr, zarr=zarr
        )
        modB = self.get_modB(Bcontrav, g)

        Igeometry = self.input.physics.Igeometry

        if Igeometry == 1:
            plotR[i * nr : (i + 1) * nr, :] = tarr[None, :]
            plotZ[i * nr : (i + 1) * nr, :] = R[:, :, 0]
        if Igeometry == 2:
            plotR[i * nr : (i + 1) * nr, :] = R[:, :, 0] * np.cos(tarr[None, :])
            plotZ[i * nr : (i + 1) * nr, :] = R[:, :, 0] * np.sin(tarr[None, :])
        if Igeometry == 3:
            plotR[i * nr : (i + 1) * nr, :] = R[:, :, 0]
            plotZ[i * nr : (i + 1) * nr, :] = Z[:, :, 0]

        plotB[i * nr : (i + 1) * nr, :] = modB[:, :, 0]
    plot = ax.pcolormesh(plotR[:, :], plotZ[:, :], plotB[:, :], **kwargs)
    if Igeometry == 1:
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$R$")
    if Igeometry == 2:
        ax.set_aspect("equal")
        ax.set_xlabel(r"$X$")
        ax.set_ylabel(r"$Y$")
    if Igeometry == 3:
        #ax.set_aspect("equal")
        ax.set_xlabel(r"$R$")
        ax.set_ylabel(r"$Z$")

    if colorbar:
        cbar = plt.colorbar(plot, ax=ax)
        cbar.set_label(r"$|\vec B|$", rotation=0)
    return


def plot_fmn(self, intf, ntheta, **kwargs):  

    tarr=np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    zarr=np.linspace(0, 0, 1)

    sarr = np.array([1.0])
    R, Z, jacobian, g = self.get_grid_and_jacobian_and_metric(
            lvol=intf, sarr=sarr, tarr=tarr, zarr=zarr)
    # Bcontrav = self.get_B(
    #     lvol=intf, jacobian=jacobian, sarr=sarr, tarr=tarr, zarr=zarr)
    
    Bcontrav = self.get_field_contrav(intf, sarr, tarr, zarr)
    
    f_th_zeta_1 = np.sqrt(np.einsum("...i,...ji,...j->...", Bcontrav, g, Bcontrav))[0,:,0]

    sarr = np.array([-1.0])
    R, Z, jacobian, g = self.get_grid_and_jacobian_and_metric(
            lvol=intf+1, sarr=sarr, tarr=tarr, zarr=zarr)
    # Bcontrav = self.get_B(
    #     lvol=intf+1, jacobian=jacobian, sarr=sarr, tarr=tarr, zarr=zarr)
    
    Bcontrav = self.get_field_contrav(intf+1, sarr, tarr, zarr)
    
    f_th_zeta_2 = np.sqrt(np.einsum("...i,...ji,...j->...", Bcontrav, g, Bcontrav))[0,:,0]

    f_jump = f_th_zeta_2**2 - f_th_zeta_1**2

    f_mn = np.real(np.fft.rfft(f_jump))

    return f_jump, f_mn


def plot_cartB(
    self,
    sarr=np.linspace(-0.999, 1, 4),
    tarr=np.linspace(0, 2 * np.pi, 5),
    zarr=np.linspace(0, 0, 2),
    lvol=None,
    ax=None,
    colorbar=True,
    comp=0,
    **kwargs,
):
    """[summary]

    Args:
        sarr ([type], optional): [description]. Defaults to np.linspace(-0.999, 1, 4).
        tarr ([type], optional): [description]. Defaults to np.linspace(0, 2 * np.pi, 5).
        zarr ([type], optional): [description]. Defaults to np.linspace(0, 0, 2).
        lvol (int, optional): [description]. Defaults to 0.
        ax (Matplotlib axis, optional): Matplotlib axis to be plotted on. Defaults to None.
    """
    import matplotlib.pyplot as plt

    Nvol = self.input.physics.Nvol
    if lvol == None:
        lvollist = np.arange(0, Nvol, dtype=int).tolist()
    elif np.isscalar(lvol):
        lvollist = [lvol]
    else:
        lvollist = lvol
    # get axis data
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)

    #nr = sarr.size
    nr = len(sarr)
    #nt = tarr.size
    nt = len(tarr)

    plotR = np.zeros([len(lvollist) * nr, nt], dtype=np.float64)
    plotZ = np.zeros([len(lvollist) * nr, nt], dtype=np.float64)
    plotB = np.zeros([len(lvollist) * nr, nt], dtype=np.float64)

    for i, ivol in enumerate(lvollist):
        # parse data
        R, Z, jacobian, g = self.get_grid_and_jacobian_and_metric(
            lvol=ivol, sarr=sarr, tarr=tarr, zarr=zarr
        )
        Bcontrav = self.get_B(
            lvol=ivol, jacobian=jacobian, sarr=sarr, tarr=tarr, zarr=zarr
        )

        Igeometry = self.input.physics.Igeometry

        if Igeometry == 1:
            plotR[i * nr : (i + 1) * nr, :] = tarr[None, :]
            plotZ[i * nr : (i + 1) * nr, :] = R[:, :, 0]
        if Igeometry == 2:
            plotR[i * nr : (i + 1) * nr, :] = R[:, :, 0] * np.cos(tarr[None, :])
            plotZ[i * nr : (i + 1) * nr, :] = R[:, :, 0] * np.sin(tarr[None, :])
        if Igeometry == 3:
            plotR[i * nr : (i + 1) * nr, :] = R[:, :, 0]
            plotZ[i * nr : (i + 1) * nr, :] = Z[:, :, 0]

        # print(Bcontrav.shape)
        plotB[i * nr : (i + 1) * nr, :] = Bcontrav[:, :, 0, comp]
    plot = ax.pcolormesh(plotR[:, :], plotZ[:, :], plotB[:, :], **kwargs)
    if Igeometry == 1:
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$R$")
    if Igeometry == 2:
        ax.set_aspect("equal")
        ax.set_xlabel(r"$X$")
        ax.set_ylabel(r"$Y$")
    if Igeometry == 3:
        #ax.set_aspect("equal")
        ax.set_xlabel(r"$R$")
        ax.set_ylabel(r"$Z$")

    if colorbar:
        cbar = plt.colorbar(plot, ax=ax)
        cbar.set_label(r"$|\vec B|$", rotation=0)
    return