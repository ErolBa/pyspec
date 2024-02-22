def get_hessian(self):
    
    # gets the hessian from the .sp.hessian file
    
    from numpy import fromfile

    # get .hessian filename
    hess_fname = self.filename[:-3] + ".hessian"
    
    def read_int(fid):
        return fromfile(fid, 'int32',1)

    with open(hess_fname, 'rb') as fid:
        read_int(fid)
        NGdof = int(read_int(fid))
        read_int(fid)
        read_int(fid)

        data = fromfile(fid, 'float64').reshape((NGdof, NGdof)).T

    return data


def get_eigenstuff(self):
    """Generates 'eigenstuff of the hessian'

    Returns:
        w, v, indmin, minvec
    """
    
    from numpy import linalg, argmin

    h = self.get_hessian()

    w, v = linalg.eig(h)

    ## there are two ways of determining the smallest eigenvalue (choose one)
    # first, looking at the smallest real part eigenvalue (numpy defualt btw)
    indmin = argmin(w)

    ## second, looking at the largest magnitude of the complex number (default in MATLAB)
    # indmin2 = np.argmin(np.abs(w))

    minvec = v[:, indmin]

    return w, v, indmin, minvec


def calc_simple_jbs(self):
    
    # calculate bootstrap current using simple formula
    # similar to approach from Antoine's thesis
    
    print(self.input.physics.Rbc)
    # const = sqrt(a * R0) / B0
    
    pass

# def plot_q_surface(self):
    
#     iotas = self.output.

def run_spec_master(fname, num_cpus=8, show_output=False, print_force=True, log_file=None):

    from subprocess import run, DEVNULL
    
    if(print_force):
        print(f"Running SPEC newton with {fname}")

    try:
        if(show_output):
            if(log_file is None):
                run(f'mpirun -n {num_cpus} ~/codes/SPEC/xspec {fname}', shell=True)
            else:
                run(f'mpirun -n {num_cpus} ~/codes/SPEC/xspec {fname}', shell=True, stdout=log_file, stderr=log_file)
        else:
            run(f'mpirun -n {num_cpus} ~/codes/SPEC/xspec {fname}', shell=True, stdout=DEVNULL, stderr=DEVNULL)

        # data = SPECout(fname+".h5")

        # if(print_force):
        #     print(f"SPEC completed  ---  |f| = {data.output.ForceErr:.5e}")

        return True

    except Exception as e:
        print(f"Running SPEC failed!!! ({e})")
        return False

def gen_poincare(fname, nppts=None, theta0=None, nptrj=None, ax=None):
    """
    Gen. new spec run for poincare, settings:
    
    if(nptrj is None):
        nptrj = 5*ones(nml['physicslist']['Nvol'])
    elif(isinstance(nptrj, int)):
        nptrj = nptrj*ones(nml['physicslist']['Nvol'])
    elif(isinstance(nptrj, list)):
        arr = zeros(nml['physicslist']['Nvol'])
        for i in range(len(nptrj)):
            arr[nptrj[i][0]] = nptrj[i][1]
        nptrj = arr
    elif(isinstance(nptrj, tuple)):
        arr = zeros(nml['physicslist']['Nvol'])
        for i in range(nptrj[0], nptrj[1]):
            arr[i] = nptrj[2]
        nptrj = arr
    """
    from subprocess import run
    from py_spec import SPECNamelist
    from numpy import ones, zeros

    if(fname[-3:] != ".sp"):
        raise ValueError("fname should be *.sp")

    run(f"cp {fname}.end temp_pncare.sp", shell=True)
    nml = SPECNamelist(f"temp_pncare.sp") 

    if(nptrj is None):
        nptrj = 5*ones(nml['physicslist']['Nvol'])
    elif(isinstance(nptrj, int)):
        nptrj = nptrj*ones(nml['physicslist']['Nvol'])
    elif(isinstance(nptrj, list)):
        arr = zeros(nml['physicslist']['Nvol'])
        for i in range(len(nptrj)):
            arr[nptrj[i][0]] = nptrj[i][1]
        nptrj = arr
    elif(isinstance(nptrj, tuple)):
        arr = zeros(nml['physicslist']['Nvol'])
        for i in range(nptrj[0], nptrj[1]):
            arr[i] = nptrj[2]
        nptrj = arr

    nml['diagnosticslist']['nPpts'] = nppts if nppts is not None else 500
    nml['diagnosticslist']['Ppts'] = theta0 if theta0 is not None else 1.0
    nml['diagnosticslist']['nPtrj'] = nptrj
    
    nml['globallist']['Lfindzero'] = 0
    
    nml.write_simple("temp_pncare.sp")
    run_spec_master("temp_pncare.sp")
    fname = "temp_pncare.sp"

    # if(ax is None):
    #     fig, ax = plt.subplots()
    # with SPECout(fname+".h5") as out:
    #     out.plot_poincare(prange='full', ax=ax);
            
    return ax        