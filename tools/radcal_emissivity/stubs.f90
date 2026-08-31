! meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
! Source-available, not Open Source. Teaching and academic research are
! free; commercial and non-academic research require a licence.
! Enquiries: simul@msimul.com
! See LICENSE at the repository root.
! Provenance: see rust/PROVENANCE.md. No GPL-licensed source was consulted.
!
! Minimal stand-ins for the three FDS modules RADCAL (rcal.f90) refers to.
! Values taken from FDS's own cons.f90 / radi.f90 (NIST, US public domain).

MODULE GLOBAL_CONSTANTS
USE PRECISION_PARAMETERS, ONLY: EB
IMPLICIT NONE (TYPE,EXTERNAL)
LOGICAL :: AEROSOL_AL2O3 = .FALSE.
REAL(EB), PARAMETER :: SIGMA = 5.670373E-8_EB   ! Stefan-Boltzmann, W/(m2 K4)
END MODULE GLOBAL_CONSTANTS

MODULE RADCONS
USE PRECISION_PARAMETERS, ONLY: EB, PI
USE GLOBAL_CONSTANTS, ONLY: SIGMA
IMPLICIT NONE (TYPE,EXTERNAL)
REAL(EB), PARAMETER :: RPI_SIGMA = SIGMA/PI
END MODULE RADCONS

MODULE COMP_FUNCTIONS
IMPLICIT NONE (TYPE,EXTERNAL)
CONTAINS
SUBROUTINE SHUTDOWN(MESSAGE,PROCESS_0_ONLY)
CHARACTER(*), INTENT(IN) :: MESSAGE
LOGICAL, INTENT(IN), OPTIONAL :: PROCESS_0_ONLY
WRITE(*,'(A)') 'RADCAL SHUTDOWN: '//TRIM(MESSAGE)
IF (PRESENT(PROCESS_0_ONLY)) CONTINUE
STOP 1
END SUBROUTINE SHUTDOWN
END MODULE COMP_FUNCTIONS
