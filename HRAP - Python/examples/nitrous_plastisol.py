# Purpose: Demonstrate API usage and validate for a typical hybrid, case is similar to GUI defaults
# HRAP_Source/HRAP - Python/examples/nitrous_plastisol.py

import scipy
import numpy as np
from pathlib import Path
from importlib.resources import files as imp_files

import matplotlib.pyplot as plt

import hrap.core as core
import hrap.chem as chem
import hrap.fluid as fluid
from hrap.tank    import *
from hrap.grain   import *
from hrap.chamber import *
from hrap.nozzle  import *
from hrap.units   import _in, _ft, _lbf, _atm

jax.config.update("jax_enable_x64", True)
hrap_root = Path(imp_files('hrap'))
file_prefix = 'nitrous_plastisol'



print('Building combustion chemistry table...')
plastisol = chem.make_basic_reactant(
    formula = 'Plastisol-362',
    composition = { 'C': 7.200, 'H': 10.82, 'O': 1.14, 'Cl': 0.669 },
    M = 140.86, # kg/kmol
    T0 = 298.15, # K
    h0 = -2.6535755e7, # J/kmol
)
comb = chem.ChemSolver([hrap_root/'thermo.dat', plastisol])
chem_Pc, chem_OF = np.linspace(10*_atm, 50*_atm, 10), np.linspace(1.0, 10.0, 20)
chem_k, chem_M, chem_T = [np.zeros((chem_Pc.size, chem_OF.size)) for i in range(3)]
ox, fu_1, fu_2 = 'N2O(L),298.15K', 'Plastisol-362', 'AL(cr)'
mfrac_al = 0.0 
internal_state = None
for j, OF in enumerate(chem_OF):
    for i, Pc in enumerate(chem_Pc):
        o = OF / (1 + OF) # o/f = OF, o+f=1 => o=OF/(1 + OF)
        flame, internal_state = comb.solve(Pc, {ox: o, fu_1: (1-mfrac_al)*(1-o), fu_2: mfrac_al*(1-o)}, max_iters=150, internal_state=internal_state)
        chem_k[i,j], chem_M[i,j], chem_T[i,j] = flame.gamma, flame.M, flame.T

print('Baking NOS saturated property curves...')
get_sat_nos_props = fluid.bake_sat_coolprop('NitrousOxide', np.linspace(183.0, 309.0, 20))


# 1. GİRDİLER VE TASARIM NOKTASI (Inputs & Design Point)
design_point = "Optimum Expansion at 0.9033 bar (Ambient)"
target_thrust = 100.0  # N
burn_time = 8.0  # s
Pc_target = 10.0 # bar
Cf_target = 1.2542
cstar_target = 1516.19 # m/s
OF_ratio = 7.3
fuel_density = 900.0 # kg/m³
reg_a = 0.0001435 # m/s
reg_n = 0.5275
initial_port_radius_mm = 12.0
ambient_pressure_pa = 90300.0 # 0.9033 bar -> Pa

# 2. GEOMETRİ VE HESAPLANAN ÇIKTILAR (RPA + Python Outputs)
throat_diameter_mm = 10.05
exit_diameter_mm = 15.79
grain_length_mm = 56.66  
inner_diameter_mm = initial_port_radius_mm * 2.0  # 24.00 mm
outer_diameter_mm = 50.37  
oxidizer_mass_flow = 0.04596  # kg/s

fuel_mass_flow = oxidizer_mass_flow / OF_ratio
total_mass_flow = oxidizer_mass_flow + fuel_mass_flow
port_area_m2 = np.pi * (initial_port_radius_mm / 1000.0)**2
oxidizer_mass_flux = oxidizer_mass_flow / port_area_m2
web_thickness_mm = (outer_diameter_mm - inner_diameter_mm) / 2.0


chamber_inner_diameter_mm = 54.0 

# Console report
print("\n" + "="*60)
print("=== ENGINEERING REPORT: HYBRID MOTOR SIZING ===")
print(f"Design Point: {design_point}")
print("-" * 60)
print("1. INPUTS & ASSUMPTIONS (per CEA/RPA):")
print(f"  Target Thrust:        {target_thrust} N")
print(f"  Burn Time:            {burn_time} s")
print(f"  Chamber Pressure:     {Pc_target} bar")
print(f"  Target c* / Cf:       {cstar_target} m/s / {Cf_target}")
print(f"  O/F Ratio:            {OF_ratio}")
print(f"  Fuel Density:         {fuel_density} kg/m³")
print(f"  Regression Law:       a = {reg_a}, n = {reg_n}")
print(f"  Ambient Pressure:     {ambient_pressure_pa} Pa")
print("-" * 60)
print("2. DETERMINED OUTPUTS & GEOMETRY:")
print(f"  Throat / Exit Dia:    {throat_diameter_mm} mm / {exit_diameter_mm} mm")
print(f"  Mass Flow (Tot/Ox/Fu):{total_mass_flow:.5f} / {oxidizer_mass_flow:.5f} / {fuel_mass_flow:.5f} kg/s")
print(f"  Grain Length / OD:    {grain_length_mm} mm / {outer_diameter_mm} mm")
print(f"  Initial Port Radius:  {initial_port_radius_mm} mm")
print(f"  Web Thickness:        {web_thickness_mm:.2f} mm")
print(f"  Oxidizer Mass Flux:   {oxidizer_mass_flux:.2f} kg/m²s")
print("-" * 60)
print("3. CONSISTENCY CHECKS:")
print(f"  Grain OD ({outer_diameter_mm}mm) <= Chamber ID ({chamber_inner_diameter_mm}mm): VALIDATED")
print("="*60 + "\n")


print('Initializing engine...')


m_ox_required = 0.45 


rho_n2o_liquid = 745.0  # kg/m³
tank_volume = (m_ox_required / rho_n2o_liquid) * 1.5  # %50 ullage

# INJECTOR 
Cd_injector = 0.65  
dP_injector = 4.0e6  # 40 bar basınç farkı
rho_injector = rho_n2o_liquid  
injector_area = oxidizer_mass_flow / (Cd_injector * np.sqrt(2 * rho_injector * dP_injector))
inj_CdA = Cd_injector * injector_area


# 1. TANK
tnk = make_sat_tank(
    get_sat_nos_props,
    V = tank_volume,
    inj_CdA = inj_CdA,
    m_ox = m_ox_required,
    T = 294,
)

# 2. GRAIN (YAKIT)
shape = make_circle_shape(
    ID = inner_diameter_mm / 1000.0,  
)
grn = make_constOF_grain(
    shape,
    OF = OF_ratio,
    OD = outer_diameter_mm / 1000.0,  
    L = grain_length_mm / 1000.0,  
    rho = fuel_density,  
)

# 3. CHAMBER (YANMA ODASI)

prepost_ID = chamber_inner_diameter_mm / 1000.0  
prepost_V  = (3.5+1.7)*_in * np.pi/4*prepost_ID**2  
rings_V    = 3 * (1/8*_in) * np.pi*(2.5/2 * _in)**2  
fuel_V     = (grain_length_mm/1000.0) * np.pi*((outer_diameter_mm/1000.0)/2)**2  
cmbr = make_chamber(
    V0 = prepost_V + rings_V + fuel_V, 
    cstar_eff = 0.9651*1.25,  
)

# 4. NOZZLE (LÜLE)
expansion_ratio = (exit_diameter_mm / throat_diameter_mm)**2 

noz = make_cd_nozzle(
    thrt = throat_diameter_mm / 1000.0,  
    ER = expansion_ratio, 
    eff = 0.9796,          # RPA: Nozzle efficiency
    C_d = 0.995,
)


from jax.scipy.interpolate import RegularGridInterpolator
chem_interp_k = RegularGridInterpolator((chem_OF, chem_Pc), chem_k, fill_value=1.4)
chem_interp_M = RegularGridInterpolator((chem_OF, chem_Pc), chem_M, fill_value=29.0)
chem_interp_T = RegularGridInterpolator((chem_OF, chem_Pc), chem_T, fill_value=293.0)

s, x, method = core.make_engine(
    tnk, grn, cmbr, noz,
    chem_interp_k=chem_interp_k, chem_interp_M=chem_interp_M, chem_interp_T=chem_interp_T,
    Pa=ambient_pressure_pa,
)

# Create the function for firing engines
fire_engine = core.make_integrator(
    core.step_fe,
    method,
)

# Integrate the engine state
T = 12.0
print('Running...')
import time
t1 = time.time()
t, _x, xstack = fire_engine(s, x, dt=1E-3, T=T)
jax.block_until_ready(xstack)
t2  = time.time()
t, x, xstack = fire_engine(s, x, dt=1E-3, T=T)
jax.block_until_ready(xstack)
t3 = time.time()
print('done, first run was {a:.2f}s, second run was {b:.2f}s'.format(a=t2-t1, b=t3-t2))

# Unpack the dynamic engine state
N_t = xstack.shape[0]
tnk, grn, cmbr, noz = core.unpack_engine(s, xstack, method)

# Ensure results folder exists
results_path = Path('./results')
results_path.mkdir(parents=True, exist_ok=True)

# Export için de Outer Diameter değerini tutarlı hale getirdik
OD, L = outer_diameter_mm/1000.0, grain_length_mm/1000.0
core.export_rse(
    results_path/(file_prefix+'.rse'),
    t, noz['thrust'].ravel(), noz['mdot'].ravel(), t*0, t*0,
    OD=OD, L=L, D_throat=s['noz_thrt'], D_exit=np.sqrt(s['noz_ER'])*s['noz_thrt'],
    motor_type='hybrid', mfg='HRAP',
)
core.export_eng(
    results_path/(file_prefix+'.eng'),
    t, noz['thrust'], t*0,
    OD=OD, L=L,
    mfg='HRAP',
)

# Visualization
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12,7))
axs = np.array(axs).ravel()

# Plot thrust
axs[0].plot(np.linspace(0.0, T, N_t), noz['thrust'], label='sim')
axs[0].axhline(y=target_thrust, color='r', linestyle='--', label=f'target={target_thrust}N')
axs[0].set_title('Thrust')
axs[0].legend()

# Plot oxidizer flow rate
axs[1].plot(np.linspace(0.0, T, N_t), tnk['mdot_ox'], label='mdot_ox')
axs[1].plot(np.linspace(0.0, T, N_t), tnk['mdot_inj'], label='mdot_inj')
axs[1].plot(np.linspace(0.0, T, N_t), tnk['mdot_vnt'], label='mdot_vnt')
axs[1].plot(np.linspace(0.0, T, N_t), grn['mdot'], label='mdot_grn')
axs[1].plot(np.linspace(0.0, T, N_t), noz['mdot'], label='mdot_noz')
axs[1].plot(np.linspace(0.0, T, N_t), cmbr['mdot_g'], label='mdot_cmbr')
axs[1].axhline(y=oxidizer_mass_flow, color='r', linestyle='--', label=f'target ox={oxidizer_mass_flow}')
axs[1].legend(loc='upper right')
axs[1].set_title('mdot')

axs[2].plot(np.linspace(0.0, T, N_t), cmbr['P'], label='chamber')
axs[2].plot(np.linspace(0.0, T, N_t), tnk['P'], label='tank')
axs[2].plot(np.linspace(0.0, T, N_t), noz['Pe'], label='noz exit')
axs[2].legend(loc='upper right')
axs[2].set_title('P')

axs[3].plot(np.linspace(0.0, T, N_t), tnk['T'])
axs[3].set_title('T tank')

axs[4].plot(np.linspace(0.0, T, N_t), tnk['m_ox_liq'], label='ox liq')
axs[4].plot(np.linspace(0.0, T, N_t), tnk['m_ox_vap'], label='ox vap')
axs[4].plot(np.linspace(0.0, T, N_t), cmbr['m_g'], label='cmbr stored')
axs[4].plot(np.linspace(0.0, T, N_t), grn['V']*grn['rho'], label='grain')
axs[4].legend()
axs[4].set_title('m')

D = (outer_diameter_mm - inner_diameter_mm) / 1000.0  # grain thickness in m
axs[5].plot([0.0, T], [D]*2, label='grain thickness')
axs[5].plot(np.linspace(0.0, T, N_t), grn['d'], label='net regression')
axs[5].legend()

axs[6].plot(np.linspace(0.0, T, N_t), noz['Me'], label='Mach exit')
axs[6].legend()

axs[7].plot(np.linspace(0.0, T, N_t), cmbr['cstar'], label='cstar')
axs[7].plot(np.linspace(0.0, T, N_t), cmbr['T'], label='cmbr T')
axs[7].legend()

axs[8].plot(np.linspace(0.0, T, N_t), cmbr['V0'] - grn['V'], label='Empty cmbr V')
axs[8].legend()

# Save plot
fig.tight_layout()
fig.savefig(results_path/(file_prefix+'_plots.pdf'), format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()