#!/usr/bin/env python
# Authors: 
# 01/23 - 03/23: Nike Dattani, nike@hpqc.org
# 03/23 - 03/23: Sam Zhuang

import pyscf, numpy as np
from pyscf import gto, scf, ao2mo, fci, ci, cc
from pyscf.lib import logger
from scipy.constants import physical_constants

def stable_opt_internal(mf):
    log = logger.new_logger(mf)
    mo1, _, stable, _ = mf.stability(return_status=True)
    cyc = 0
    while (not stable and cyc < 10):
        log.note('Try to optimize orbitals until stable, attempt %d' % cyc)
        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf = mf.run(dm1)
        mo1, _, stable, _ = mf.stability(return_status=True)
        cyc += 1
    if not stable:
        log.note('Stability optimization failed after %d attempts' % cyc)
    return mf



# XYZ coordinates:
molecule_xyz = '''
O  0.9858 0.0000  2.0307
H -1.0241 0.0000  1.7361
C  0.0000 0.0000  1.3430
O  0.0000 0.0000  0.0000
H  0.9329 0.0000 -0.2728
'''


# Z-matrix:
R_CO1 = 1.343
R_CH = 1.097
R_CO2 = 1.202
R_OH = 0.972

A1 = 111
A2 = 106.3
A3 = 124.9

D1 = 180
D2 = 0

molecule_zmat = f'''
H1
C1 1 {R_CH}
O1 2 {R_CO1} 1 {A1}
H2 3 {R_OH}  2 {A2} 1 {D1}
O2 2 {R_CO2} 3 {A3} 4 {D2}
'''

if __name__ == '__main__':

    ## Constants
    VERBOSITY = 9
    CONV_TOL = 1e-7
    BASIS = 'def2-TZVPP'
    FROZEN = 3
    USE_ZMAT = True

    if USE_ZMAT:
        molecule = molecule_zmat
    else:
        molecule = molecule_xyz

    HARTREE_TO_EV_FACTOR = dict(physical_constants)['Hartree energy in eV'][0]

    # Energy calculation for neutral molecule
    neut_mol = pyscf.M(atom = molecule, basis = BASIS, verbose=VERBOSITY, spin=0, charge=0)
    #neut_mol.symmetry = False
    #neut_mhf = neut_mol.UHF().set(conv_tol=1e-10,max_cycle=999,direct_scf_tol=1e-14) # Hartree-Fock calculation, mean-field object created
    neut_mhf = scf.UHF(neut_mol).run()                          
    neut_mhf = stable_opt_internal(neut_mhf)                                         # Opimize the HF solution using stability analysis
    neut_mcc = cc.UCCSD(neut_mhf).set(conv_tol=CONV_TOL, frozen=FROZEN)                       # Post-Hartree-Fock
    neut_tup = neut_mcc.kernel()                                                     #
    neut_et  = neut_mcc.ccsd_t()                                                     #
    neut_E   = neut_mcc.e_tot + neut_et                                              #
    
    # Calculation for cation molecule
    cat_mol = pyscf.M(atom = molecule, basis = BASIS, verbose=VERBOSITY, spin=1, charge=1)
    #cat_mol.symmetry = False
    #cat_mhf = cat_mol.UHF().set(conv_tol=1e-10,max_cycle=999,direct_scf_tol=1e-14) # Hartree-Fock calculation, mean-field object created
    cat_mhf = scf.UHF(cat_mol).run()                          
    cat_mhf = stable_opt_internal(cat_mhf)                                           # Opimize the HF solution using stability analysis
    cat_mcc = cc.UCCSD(cat_mhf).set(conv_tol=CONV_TOL, frozen=FROZEN)                         # Post-Hartree-Fock
    cat_tup = cat_mcc.kernel()                                                       #
    cat_et  = cat_mcc.ccsd_t()                                                       #
    cat_E   = cat_mcc.e_tot + cat_et                                                 #
    
    E_diff  = cat_E - neut_E                                                         # Ionization energy
    
    # print("Neutral molecule SCF energy (hartrees) =", neut_)
    print("Neutral molecule UCCSD energy (hartrees) =", neut_mcc.e_tot)
    print("Neutral molecule UCCSD(T) energy (hartrees) =", neut_E)
    
    # print("Cation molecule SCF energy (hartrees) =", cat_)
    print("Cation molecule UCCSD energy from output file (hartrees) =", cat_mcc.e_tot)
    print("Cation molecule UCCSD(T) energy (hartrees) =", cat_E)
    
    print("UCCSD energy difference (hartrees)= ", cat_mcc.e_tot-neut_mcc.e_tot)
    print("UCCSD energy difference (eV) = ", HARTREE_TO_EV_FACTOR*(cat_mcc.e_tot-neut_mcc.e_tot)) # Use 27.211386245988(53) eV from 2018 CODATA
    
    print("UCCSD(T) energy difference (hartrees)= ", E_diff)
    print("UCCSD(T) energy difference (eV) = ", HARTREE_TO_EV_FACTOR*E_diff)                      # Use 27.211386245988(53) eV from 2018 CODATA
    
    
