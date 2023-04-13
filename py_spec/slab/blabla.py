def find_max_psiw(rslab, profile, r_s, constraint=2):
    # finds max psiw for each delprimea
    psiw_lims = [0.004, 0.35]

    def get_eigval_for_psiw(psiw, rslab, profile, r_s):
        inpdict = input_dict()
        inpdict.psi0_mag = 3 * np.sqrt(3) / 4
        inpdict.bz0_mag = 10.0
        inpdict.linitialize = 0
        inpdict.lbeltrami = 4
        inpdict.lfindzero = 2
        inpdict.infaces_spacing = 'psi'
        inpdict.pre = np.zeros(inpdict.Nvol)
        inpdict.lconstraint = constraint
        inpdict.fname_template = 'template.sp'
        inpdict.fname_input = 'stability_analysis.sp'
        inpdict.psi_w = psiw
        inpdict.Nvol = 41
        inpdict.Mpol = 2 # min eigval independent of mpol (so use smallest mpol=1)
        inpdict.rslab = rslab

        config = HMHDslab.gen_profiles_from_psi(profile)

        x, psi0_sym, bz0_sym = sym.symbols('x, psi0, Bz0')
        psi_sym = sym.sympify(profile)
        by_sym = sym.diff(psi_sym, x)
        bz_sym = sym.sqrt(bz0_sym**2 - by_sym**2)
        jy_sym = sym.diff(bz_sym, x)
        jz_sym = sym.diff(by_sym, x)
        psi_func = sym.lambdify(x, psi_sym.subs(psi0_sym, psi0_mag))
        by_func = sym.lambdify(x, by_sym.subs(psi0_sym, psi0_mag))
        bz_func = sym.lambdify(x, bz_sym.subs(psi0_sym, psi0_mag).subs(bz0_sym, bz0_mag))
        jz_func = sym.lambdify(x, jz_sym.subs(psi0_sym, psi0_mag))
        jy_func = sym.lambdify(x, jy_sym.subs(psi0_sym, psi0_mag).subs(bz0_sym, bz0_mag))
        tflux_func = lambda t: quad(bz_func, -np.pi, t)[0] * size

        delp, sigp = HMHDslab.eval_delprime_sym(profile, psi0_mag, k=1/rslab)
        r_down, r_up = SPECslab.calc_asym_inface_pos(r_s, psiw, delp, sigp, profile, psi_func, tflux_func)
        inpdict.r_up = r_up
        inpdict.r_down = r_down

        run_slab_profile(inpdict, config, False)

        eigval, eigvec, min_eigval_ind, min_eigvec = SPECslab.get_eigenstuff(inpdict.fname_input)
        min_eigval = eigval[min_eigval_ind]

        # print(delprimea, psiw, min_eigval)
        # plt.plot(psiw, min_eigval if min_eigval<0 else 0.1,'d')

        if(min_eigval < 0):
            return -psiw
        else:
            return 0.1

    res = minimize_scalar(lambda x: get_eigval_for_psiw(x, rslab, profile, r_s),
                          bounds=(psiw_lims[0], psiw_lims[1]), method='bounded',
                          options={'xatol': 1e-03, 'maxiter': 14, 'disp': 0})
    best_psiw = res.x
    get_eigval_for_psiw(res.x, rslab, profile)

    return best_psiw
