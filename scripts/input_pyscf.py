# Pyscf calculation for CH2O2 (Formic acid)
import numpy as np
import pyscf
from pyscf import gto, scf, ao2mo, fci, ci, cc
from pyscf.lib import logger

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
        log.note('Stability Opt failed after %d attempts' % cyc)
    return mf

# Z-matrix definition
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
C1       1       {R_CH}
O1       2       {R_CO1}       1       {A1}
H2       3       {R_OH}        2       {A2}     1       {D1}
O2       2       {R_CO2}       3       {A3}     4       {D2}
'''

# XYZ format definition
molecule_xyz = '''
O  0.9858 0.0000  2.0307 
H -1.0241 0.0000  1.7361
C  0.0000 0.0000  1.3430
O  0.0000 0.0000  0.0000
H  0.9329 0.0000 -0.2728
'''

# Calculation for neutral molecule
neut_mol = pyscf.M(atom = molecule_xyz, basis = 'def2-TZVPP', verbose=9, spin=0, charge=0, output='out_neut_ccsd.txt')
#neut_mol.symmetry = False
neut_mhf = scf.UHF(neut_mol).run()                          # Hartree-Fock calculation, mean-field object created
neut_mhf = stable_opt_internal(neut_mhf)                    # Running stable_opt_internal loop
neut_mcc = cc.UCCSD(neut_mhf).set(conv_tol=1e-5, frozen=3)  # Configuration Interaction
neut_tup = neut_mcc.kernel()                                # Configuration energies
neut_E = neut_mcc.e_tot + neut_tup[0]                       # Total energies

# Calculation for cation molecule
cat_mol = pyscf.M(atom = molecule_xyz, basis = 'def2-TZVPP', verbose=9, spin=1, charge=1, output='out_cat_ccsd.txt')
#cat_mol.symmetry = False
cat_mhf = scf.UHF(cat_mol).run()                            # Hartree-Fock calculation, mean-field object created
cat_mhf = stable_opt_internal(cat_mhf)                      # Running stable_opt_internal loop
cat_mcc = cc.UCCSD(cat_mhf).set(conv_tol=1e-5, frozen=3)    # Configuration Interaction
cat_tup = cat_mcc.kernel()                                  # Configuration energies
cat_E = cat_mcc.e_tot + cat_tup[0]                          # Total energies

E_diff = cat_E - neut_E

print("Neutral molecule E(UCCSD) from output file (Hartrees) =", neut_mcc.e_tot)
print("Neutral molecule E_corr from output file (Hartrees) =", neut_mcc.e_corr)
print("Neutral molecule energy calculated from mcc.kernel() (Hartrees) =", neut_tup[0])
print("Neutral molecule total energy (Hartrees) =", neut_E)

print("Cation molecule E(UCCSD) from output file (Hartrees) =", cat_mcc.e_tot)
print("Cation molecule E_corr from output file (Hartrees) =", cat_mcc.e_corr)
print("Cation molecule energy calculated from mcc.kernel() (Hartrees) =", cat_tup[0])
print("Cation molecule total energy (Hartrees) =", cat_E)

print("UCCSD energy difference (Hartrees)= ", cat_mcc.e_tot-neut_mcc.e_tot)
print("UCCSD energy difference (eV) = ", 27.2114*(cat_mcc.e_tot-neut_mcc.e_tot))

print("UCCSD(T) energy difference (Hartrees)= ", E_diff)
print("UCCSD(T) energy difference (eV) = ", 27.2114*E_diff)