import matplotlib.pyplot as plt
import numpy as np
import cosmopower as cp
import camb
from cosmology import Cosmology
import glass
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline


import pylevin as levin


bias_emu = cp.cosmopower_NN(restore=True, restore_filename="./../cosmopower/bias_sq_model")
power_emu = cp.cosmopower_NN(restore=True, restore_filename="./../cosmopower/pkmm_nonlin_model")



h = 0.7
Oc = 0.25
Ob = 0.05
ns = 0.96

# basic parameters of the simulation
nside = 256
lmax = 1000

# set up CAMB parameters for matter angular power spectrum
pars = camb.set_params(
    H0=100 * h,
    omch2=Oc * h**2,
    ombh2=Ob * h**2,
    NonLinear=camb.model.NonLinear_both,
)

# get the cosmology from CAMB
cosmo_camb = Cosmology.from_camb(pars)

zet_max = 1.5
delta_chi = 400

zb = np.linspace(0,zet_max,5)
shells = glass.tophat_windows(zb)


zet_inter = np.linspace(shells[0].za[0], shells[0].za[-1], len(shells)*len(shells[0].za))
z_of_chi = UnivariateSpline(cosmo_camb.dc(zet_inter), zet_inter, k=1, s=0)
dzdchi = z_of_chi.derivative()(cosmo_camb.dc(zet_inter))
dzdchi = UnivariateSpline(cosmo_camb.dc(zet_inter), dzdchi, k=1, s=0)

chi_of_z = UnivariateSpline(zet_inter,cosmo_camb.dc(zet_inter), k=1, s=0)


chi_b = np.linspace(0,chi_of_z(zet_max), int(chi_of_z(zet_max)/delta_chi))

zb = z_of_chi(chi_b)
shells = glass.tophat_windows(zb)



logTAGN = 7.3
m_nu = 0.06

params = {'Omega_b': [Ob]*np.ones_like(zet_inter),
          'Omega_cdm' : [Oc]*np.ones_like(zet_inter),
          'h' : [h]*np.ones_like(zet_inter),
          'n_s' : [ns]*np.ones_like(zet_inter),
          'log10_T_heat' : [logTAGN]*np.ones_like(zet_inter),
          'm_nu' : [m_nu]*np.ones_like(zet_inter),
          'sigma8' : [0.8]*np.ones_like(zet_inter),
          'alpha_B' : [0.05]*np.ones_like(zet_inter),
          'alpha_M' : [0.05]*np.ones_like(zet_inter),
          'log10_k_screen' : [0.1]*np.ones_like(zet_inter),
          'z_val' : zet_inter,}
power = RegularGridInterpolator((power_emu.modes, 1/(zet_inter+1)), 10**((bias_emu.predictions_np(params) + power_emu.predictions_np(params)).T) ,bounds_error= False, fill_value = None)
power_mm = RegularGridInterpolator((power_emu.modes, 1/(zet_inter+1)), 10**((power_emu.predictions_np(params)).T) ,bounds_error= False, fill_value = None)
power = RectBivariateSpline(power_emu.modes, zet_inter, 10**((bias_emu.predictions_np(params) + power_emu.predictions_np(params)).T) ,kx=1, ky=1)




ell = np.unique(np.geomspace(1,lmax,100).astype(int))

integral_type = 0
N_thread = 4 # Number of threads used for hyperthreading
logx = False # Tells the code to create a logarithmic spline in x for f(x)
logy = False # Tells the code to create a logarithmic spline in y for y = f(x)
n_sub = 6 #number of collocation points in each bisection
n_bisec_max = 32 #maximum number of bisections used
rel_acc = 5e-3 #relative accuracy target
boost_bessel = True #should the bessel functions be calculated with boost instead of GSL, higher accuracy at high Bessel orders
verbose = False #should the code talk to you?

kmin, kmax, N_int = 1e-4, 1e1, int(1e3)

k_int = np.geomspace(kmin, kmax, N_int)



inner_int = np.zeros((len(ell), len(k_int)))


inner_int_shells = []
for i_shell in range(len(shells)):
    chi = cosmo_camb.dc(shells[i_shell].za)
    norm = np.trapezoid(shells[i_shell].wa,shells[i_shell].za)
    integrand = np.sqrt(power(k_int, shells[i_shell].za))*(shells[i_shell].wa*dzdchi(chi))[:,None]/norm
    idx = np.where(integrand[:,-1] > 0)[0]
    chi = chi[idx]
    integrand = integrand[idx,:]
    for i_ell, val_ell in enumerate(ell):
        ell_values = (val_ell*np.ones_like(k_int)).astype(int)
        lp = levin.pylevin(integral_type, chi, integrand, logx, logy, N_thread, True)
        lp.set_levin(n_sub, n_bisec_max, rel_acc, boost_bessel, verbose)
        lp.levin_integrate_bessel_single(chi[0]*np.ones_like(k_int), chi[-1]*np.ones_like(k_int), k_int, ell_values, inner_int[i_ell,:])
    inner_int_shells.append(inner_int*k_int[None,:])
inner_int_shells = np.array(inner_int_shells)
result_levin = 2/np.pi*np.trapezoid(inner_int_shells[:, None, :,:]* inner_int_shells[None, :, :,:],k_int, axis = -1)






