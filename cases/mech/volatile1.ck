! volatile1.ck - the one-step volatile mechanism of a Phase B dust cloud
! (SPEC-LIT 159.10, equations (159.16)-(159.17); docs/17 DST-15, decision DD11).
!
! VOL is the corn-starch fuel formula C6H7.88O4.98 of SPEC-LIT 152.1 times 50,
! so that its element counts are whole (C300 H394 O249). Its NASA-7 record is
! 50 times the starch's per-mole record of (152.4)-(152.5): b1 = 50 a0,
! b2 = 50 a1, b6 = 50 a5 with a5 re-derived from H_f = -792.6 kJ/mol
! (SPEC-LIT 152.2), b3 = b4 = b5 = b7 = 0, one range 200-6000 K. Per kilogram
! it is the starch exactly. Source of a0, a1 and H_f: C. Huang, M. Bloching and
! A. N. Lipatnikov, J. Loss Prev. Process Ind. 75 (2022) 104707,
! DOI 10.1016/j.jlp.2021.104707, eqs. (1)-(2) and Table 1 (CC BY).
!
! THE RATE IS AUTHORED, UNCALIBRATED. No open source gives corn-starch volatile
! kinetics (SPEC-LIT 151.8). A and E were chosen so that the volatile's life
! 1/(k [O2]) is about 0.14 ms at 1500 K in air at 1 atm and the rate is
! negligible at room temperature. No gate depends on them (SPEC-LIT 159.10).
! The other species' thermo is NASA TM-4513 (cases/mech/tm4513_thermo.dat).
!
ELEMENTS C H O N END
SPECIES VOL O2 H2O CO2 N2 END
THERMO ALL
   200.000  1000.000  6000.000
VOL               DST15 C 300H 394O 249     G   200.000  6000.000 1000.00      1
-1.63630000E+02 5.02800000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00    2
-4.94108521E+06 0.00000000E+00-1.63630000E+02 5.02800000E+00 0.00000000E+00    3
 0.00000000E+00 0.00000000E+00-4.94108521E+06 0.00000000E+00                   4
END
REACTIONS
VOL + 274O2 => 300CO2 + 197H2O   1.0E+14  0.0  30000.0
  FORD /VOL 1.0/
  FORD /O2 1.0/
END
