def plot_kam_surface(self, surfaces=None, num_theta=400, zeta=0.0, ax=None, color='r', **kwargs):
    
    import matplotlib.pyplot as plt
    from numpy import pi, linspace, arange
    
    if(ax is None):
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    
    thetas = linspace(0, 2*pi, num_theta)
    figsize = 8.0
    
    geometry = self.input.physics.Igeometry
    nvol = self.input.physics.Nvol
    
    if(surfaces is None):
        surfaces = arange(nvol)
    
    if(geometry == 1):
        for s in surfaces:
            rarr, _ = self.get_coord_transform(s, 1.0, thetas, zeta)
            r = rarr[0,0,:,0]
            
            plt.plot(thetas, r, color=color, **kwargs)
            
        ax.set_xlabel('R [m]', fontsize=14)
        ax.set_ylabel('Z [m]', fontsize=14)
        ax.set_xlim(0, 2*pi)
        ax.set_ylim(0, 2*pi)
        
        fig.set_size_inches(figsize, figsize)
    
    elif(geometry == 3):
        for s in surfaces:
            rarr, zarr = self.get_coord_transform(s, 1.0, thetas, zeta)
            r = rarr[0,0,:,0]
            z = zarr[0,0,:,0]
            
            if(s == (nvol-1)):
                plt.plot(r, z, color='k', **kwargs)
                continue
            
            plt.plot(r, z, color=color, **kwargs)
            
        ax.set_xlabel('R [m]', fontsize=14)
        ax.set_ylabel('Z [m]', fontsize=14)
        
        ax.axis('equal')
        # ax.set_adjustable('datalim')
        aspect_ratio = (z.max() - z.min()) / (r.max() - r.min())
        fig.set_size_inches(figsize, figsize * aspect_ratio)
        
    else:
        raise ValueError(f"Invalid geometry {geometry}")