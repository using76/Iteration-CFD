! BFER2, the two-step reduced methane-air mechanism of:
!   B. Franzelli, E. Riber, L. Y. M. Gicquel and T. Poinsot, "Large Eddy
!   Simulation of combustion instabilities in a lean partially premixed
!   swirled flame", Combustion and Flame 159 (2012) 621-637,
!   DOI 10.1016/j.combustflame.2011.08.004.
! Read copy: the authors' manuscript, hosted by CERFACS at
!   https://www.cerfacs.fr/cantera/docs/mechanisms/methane-air/DOCS/BFER/CF_Preccinsta.pdf
!   sha256 42c230d28d2e25382928f43489497d8c5d727b746d86225fbb1bb780bd77f982,
!   read 2026-10-18 (SPEC-LIT 133.12). Eqs. (2)-(3) p. 8, eqs. (4)-(5) p. 9,
!   Table 3 p. 28, Table 4 p. 28, eqs. (6)-(7) p. 10.
!
! This file is the phi = 1 SPECIALISATION: the paper's phi-dependent
! corrections f1 (eq. (6)) and f2 (eq. (7)) are evaluated once at phi = 1 and
! folded into the pre-exponential factors, A1' = A1 f1(1) = 5.264154282E+09
! and A2' = A2 f2(1) = 4.454018570E+07. That is exact only for the case that
! uses this file (cases/flame1d_ch4_bfer.jsonc): every species there has the
! same diffusivity (one Le for all, SPEC-LIT 133.11's Lewis route), so the
! element mass fractions obey a source-free transport equation whose uniform
! inlet and initial values are both stoichiometric, and the local phi is 1
! everywhere at all times.
!
! Recorded inconsistency: CERFACS's own Cantera file 2S_CH4_BFER.cti
! (sha256 a7bf7580673e7eb1f7b8d915e472e4a07c57aca3595ceb9e75ff95408845cea5)
! carries beta2 = 0.7 and no f1/f2; the paper's Table 3 says 0.8. The paper
! is used (SPEC-LIT 133.12).
!
ELEMENTS O H C N END
SPECIES CH4 O2 CO H2O CO2 N2 END
REACTIONS
CH4 + 1.5O2 => CO + 2H2O   5.264154282E+09  0.0  35500.0
  FORD /CH4 0.5/
  FORD /O2 0.65/
CO + 0.5O2 <=> CO2   4.454018570E+07  0.8  12000.0
END
