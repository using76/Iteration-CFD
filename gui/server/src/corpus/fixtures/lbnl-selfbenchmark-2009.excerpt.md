EXCERPT transcribed for test use from: Mathew, Ganguly, Greenberg, Sartor,
Self-benchmarking Guide for Data Centers: Metrics, Benchmarks, Actions,
Lawrence Berkeley National Laboratory for NYSERDA, 13 July 2009.
Prepared under US Government sponsorship; no restrictive licence.
Source: https://www.osti.gov/servlets/purl/983248
Not the whole document: only the air-management and energy-ratio metrics.

A1 Delta-T, air management
A1 = dA2 - dA1, the difference between the return air temperature dA2 and the
supply air temperature dA1 at the cooling unit, in degrees C.
A low supply temperature together with a small supply-return differential
indicates an air management opportunity.

A2 Relative humidity delta
A2 = RH_return - RH_supply, in percentage points of relative humidity.

A3 Return temperature index
A3 = ((dA2 - dA1) / (dA6 - dA5)) * 100, where dA6 is the mean rack outlet
temperature and dA5 the mean rack inlet temperature. RTI below 100 % means
bypass air; RTI above 100 % means recirculation; 100 % is ideal.

A4 Airflow efficiency
A4 = dA7 * 1000 / dA8, where dA7 is the total fan power in kW and dA8 the total
fan airflow in cfm, so the metric has units of W/cfm. LBNL's better practice
threshold is 0.5 W/cfm.

B1 Data centre infrastructure efficiency
B1 = dE2 / (dE1 + (dE4 + dE5 + dE6) * 293), where dE1 is annual electrical kWh,
dE2 annual IT electrical kWh, and dE4, dE5, dE6 annual fuel, district steam and
district chilled water in MMBTU at 293 kWh per MMBTU. Reported as a percentage.

B2 Power usage effectiveness
B2 = (dE1 + (dE4 + dE5 + dE6) * 293) / dE2, the reciprocal of B1. It is a
facility annual energy ratio and is not computable from a room model.
