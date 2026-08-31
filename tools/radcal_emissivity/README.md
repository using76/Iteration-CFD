# `radcal_emissivity` — the published reference SPEC-LIT §62.12's Gate 1-E uses

This is the standalone driver that produced `ofgpu::wsgg::RADCAL_EPS`: the 108
recorded total emissivities of an isothermal, homogeneous H2O/CO2/N2 column
that the WSGG model of SPEC-LIT §62 is gated against.

## Why it exists

§62.2 records that Bordbar's own emissivity table could not be obtained — the
paper is paywalled and every route to it returned HTTP 403 — so until this
driver existed the emissivity level in §62.12's Gate 1 was checked against a
hand-written band with no published number behind it. **RADCAL is a published
reference that is in this repository already and is US public domain**, so the
gate can quote real numbers, and anyone can regenerate them.

## What it is, and its licence

`reference/fds/Source/rcal.f90` is NIST's own implementation of RADCAL —

> W. L. Grosshandler, *RADCAL: A Narrow-Band Model for Radiation Calculations
> in a Combustion Environment*, NIST Technical Note 1402 (1993),
> DOI `10.6028/NIST.TN.1402`.

It ships inside FDS, which is **US public domain**
(`reference/fds/LICENSE.md`: *"software developed by NIST employees is not
subject to copyright protection within the United States"*). This directory
adds two files and compiles NIST's source unmodified:

| file | what it is |
|---|---|
| `stubs.f90` | the three FDS modules `rcal.f90` refers to (`GLOBAL_CONSTANTS`, `RADCONS`, `COMP_FUNCTIONS`), cut down to the four symbols it actually uses |
| `emissivity_driver.f90` | one segment, one call to `SUB_RADCAL` per grid point, printing `1 - TOTAL_TRANSMISSIVITY` |

**No GPL-licensed source was consulted.**

## What it computes

`SUB_RADCAL` returns `TOTAL_TRANSMISSIVITY`, the Planck-weighted mean
transmissivity of the path evaluated at the *wall* temperature. Setting the
wall temperature to the **gas** temperature turns that into the total
emissivity of an isothermal column, which is exactly the quantity Hottel's
charts plot and exactly the quantity (62.1) fits:

```
eps_win(T, p_a L, M_r) = INT_win [1 - tau_w] B_w(T) dw / INT_win B_w(T) dw
```

over RADCAL's own window `50 <= w <= 10000 1/cm`. The column is one segment at
1 atm total pressure with `X_H2O + X_CO2 = 0.271` (the stoichiometric
propane-air product mixture — pass a different fraction as `argv[1]`), the
balance nitrogen, no soot; the **path length** carries `p_a L`.

The window is not the whole spectrum at flame temperature — a seventh of a
2400 K blackbody radiates below 1 micron, where RADCAL does not look and the
gas does not absorb — so the crate multiplies each entry by
`ofgpu::wsgg::RADCAL_WINDOW_FRACTION`, and a unit test recomputes those six
numbers from the Planck function.

## Building and running

Any Fortran compiler will do; the numbers below were produced with Intel
`ifx` 2025.3 on Windows.

`reference/` is **git-ignored** (see `.gitignore` and `NOTICE`): FDS is 191 MB
and is not what this project distributes, so clone it separately before
building this.

```sh
cp ../../reference/fds/Source/prec.f90 .
cp ../../reference/fds/Source/rcal.f90 .
ifx /O2 prec.f90 stubs.f90 rcal.f90 emissivity_driver.f90 /exe:radcal_eps.exe
./radcal_eps.exe
# gfortran: gfortran -O2 -o radcal_eps prec.f90 stubs.f90 rcal.f90 emissivity_driver.f90
```

Columns are `M_r`, `T` (K), `p_a L` (atm m), `L` (m), `X_H2O`, `X_CO2`,
`1 - TOTAL_TRANSMISSIVITY`, and RADCAL's own Planck-mean absorption
coefficient (1/m).

## What `p_a L` does and does not fix

(62.1) makes the total emissivity a function of `p_a L`, `T` and `M_r` alone.
RADCAL's is not, because collision broadening depends on the partial pressure
and not only on the product — so the composition this driver is run at is part
of the answer. Run it three times to see the size of that:

```sh
./radcal_eps.exe 0.10   # thin plume
./radcal_eps.exe 0.271  # stoichiometric propane-air products (the recorded one)
./radcal_eps.exe 0.50   # richer than any real flame
```

At the SAME `p_a L`, `0.10` reads a mean of **−2.4 %** against `0.271` (worst
−7.1 %) and `0.50` reads **+2.4 %** (worst +7.2 %); the worst spread across the
three at a single point is **15.3 %**. That is a real limit on what Gate 1-E
can claim, it is (62.1)'s idealisation rather than RADCAL's shortcoming, and
SPEC-LIT §62.12 states it before the verdict rather than after.

## The check that says the driver is driven correctly

Run it at `p_a L = 1e-6` and the emissivity must collapse to the optically
thin limit `kappa_Planck * L`, with `kappa_Planck` the last column. It does,
to four significant figures: at 400 K, `7.3214619 * 3.690e-6 = 2.702e-5`
against a printed `2.70e-5`; at 700 K, `6.5578506 * 3.690e-6 = 2.420e-5`
against `2.42e-5`. That is an internal consistency check of RADCAL's two
outputs against each other, and it is what rules out a mis-set path length or
a mis-read return value.

## The verdict this produced

Recorded in SPEC-LIT §62.13 and printed live by `ofgpu-validate`: mean
`|d eps / eps|` of **11.4 %** over the 108 points, worst **30.5 %**, with the
signed bias falling **monotonically** with temperature from **+20.8 %** at
400 K to **−12.3 %** at 2400 K and crossing zero near Bordbar's own
`T_ref = 1200 K`. Gate 1-E's ±10 % bar is **missed** at 58 of 108 points.
Neither model is truth, and §62.13 says so in those words.
