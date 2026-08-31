! meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
! Source-available, not Open Source. Teaching and academic research are
! free; commercial and non-academic research require a licence.
! Enquiries: simul@msimul.com
! See LICENSE at the repository root.
! Provenance: see rust/PROVENANCE.md. No GPL-licensed source was consulted.
!
! Standalone driver for RADCAL (NIST TN 1402, US public domain; source
! reference/fds/Source/rcal.f90).  Computes the TOTAL EMISSIVITY of a
! homogeneous isothermal H2O/CO2/N2 column,
!     eps = 1 - INT tau_w B_w(T) dw / INT B_w(T) dw
! which is what SUB_RADCAL returns as 1 - TOTAL_TRANSMISSIVITY once the
! "wall" temperature is set to the gas temperature.
PROGRAM RADCAL_EMISSIVITY
USE PRECISION_PARAMETERS, ONLY: EB
USE RADCAL_VAR
USE RADCAL_CALC, ONLY: INIT_RADCAL, RCALLOC, RCDEALLOC, RCDEALLOC2, SUB_RADCAL
IMPLICIT NONE (TYPE,EXTERNAL)

REAL(EB) :: AMEAN, AP0, RADIANCE, TRANSMISSIVITY
REAL(EB) :: T, L, XH2O, XCO2, PA, MR
INTEGER  :: IT, IL, IM, NARG
CHARACTER(64) :: ARG

REAL(EB), PARAMETER :: TLIST(6) = [400._EB, 700._EB, 1000._EB, 1500._EB, 2000._EB, 2400._EB]
REAL(EB), PARAMETER :: PALIST(6) = [0.01_EB, 0.03_EB, 0.1_EB, 0.3_EB, 1.0_EB, 3.0_EB]
! Molar ratios: propane-air (1.333), methane-air (2.0), ethylene-air (1.0)
REAL(EB), PARAMETER :: MRLIST(3) = [1.0_EB, 1.3333333333333333_EB, 2.0_EB]
! Participating-gas partial pressure fraction of the mixture (atm), i.e.
! X_H2O + X_CO2. Stoichiometric propane-air products give 0.271; this is the
! composition every point below is evaluated at, the path length carrying
! p_a L.
REAL(EB) :: PAFRAC

PAFRAC = 0.271_EB
NARG = COMMAND_ARGUMENT_COUNT()
IF (NARG >= 1) THEN
   CALL GET_COMMAND_ARGUMENT(1, ARG)
   READ(ARG,*) PAFRAC
ENDIF

CALL RCALLOC
ALLOCATE(SEGMENT_LENGTH_M(1))
ALLOCATE(TOTAL_PRESSURE_ATM(1))
ALLOCATE(TEMP_GAS(1))
ALLOCATE(PARTIAL_PRESSURES_ATM(16,1))
NPT = 1
TOTAL_PRESSURE_ATM(1) = 1.0_EB
LAMBDAMIN = -1.1E+4_EB
LAMBDAMAX = -1.0E+4_EB
OMMIN = 50._EB
OMMAX = 10000._EB
CALL INIT_RADCAL

WRITE(*,'(A)') '# p_a_fraction = '
WRITE(*,'(F10.6)') PAFRAC
WRITE(*,'(A)') '# M_r  T_K  p_a_L_atm_m  L_m  X_H2O  X_CO2  eps_radcal  kappa_planck_1_per_m'
DO IM = 1, SIZE(MRLIST)
   MR   = MRLIST(IM)
   XCO2 = PAFRAC/(1._EB+MR)
   XH2O = PAFRAC - XCO2
   PA   = XH2O + XCO2
   DO IT = 1, SIZE(TLIST)
      T = TLIST(IT)
      DO IL = 1, SIZE(PALIST)
         L = PALIST(IL)/PA
         SEGMENT_LENGTH_M(1) = L
         TEMP_GAS(1) = T
         TWALL       = T
         PARTIAL_PRESSURES_ATM = 0._EB
         PARTIAL_PRESSURES_ATM(I_CO2,1) = XCO2
         PARTIAL_PRESSURES_ATM(I_H2O,1) = XH2O
         PARTIAL_PRESSURES_ATM(I_N2 ,1) = 1._EB - XCO2 - XH2O
         TRANSMISSIVITY = 1._EB
         CALL SUB_RADCAL(AMEAN,AP0,RADIANCE,TRANSMISSIVITY)
         WRITE(*,'(F8.4,1X,F8.1,1X,F10.4,1X,F10.4,1X,F9.5,1X,F9.5,1X,F12.7,1X,F14.7)') &
              MR, T, PALIST(IL), L, XH2O, XCO2, 1._EB-TRANSMISSIVITY, AP0*100._EB
      ENDDO
   ENDDO
ENDDO

CALL RCDEALLOC2
CALL RCDEALLOC
END PROGRAM RADCAL_EMISSIVITY
