This is the data side of an economics research project attmepting to find correlation and effects of income and tax county emission data using panel data from 2010-2021.

_____________________________________________________________________________________________________________________________________________________


**Sources:**

https://vulcan.rc.nau.edu/

https://www.census.gov/data/datasets/time-series/demo/popest/2010s-counties-total.html

https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-total.html

https://www.ers.usda.gov/data-products/county-level-data-sets/county-level-data-sets-download-data/

https://apps.bea.gov/histdatacore/Regional_Accounts_new.html

https://www.census.gov/programs-surveys/stc/data/datasets.html?utm_source=chatgpt.com&text-list-4071d3d4f8%3Atab=all#text-list-4071d3d4f8
_____________________________________________________________________________________________________________________________________________________

**Guide:**

Code for basic regression and cleaning for cross-sectional data is in "main/Project_Code"

Code for Double Machine Learning for causal inference on panel data (2010-2021) will be in "main/ECONPROJECT Causal Inference Question Code (DML).Rmd"

Code for cleaning panel data is in "main/ECONPROJECT Data Cleaning for DML.R"

_____________________________________________________________________________________________________________________________________________________

**Notes:**

Not sure if the names of counties are accurate in the political or modern sense, could be under an old name and 
I am not intimately knowledgeable about every county in the US to know if they are correct; I apologize if any 
counties are under an improper name.

_____________________________________________________________________________________________________________________________________________________


**EMISSIONS VARIABLE NAMES:**


RES npt Coal (tC): Residential non-point source coal emissions (in tons of carbon)

RES npt Petrol (tC): Residential non-point source petroleum emissions (in tons of carbon)

RES npt NG (tC): Residential non-point source natural gas emissions (in tons of carbon)

COM npt Coal (tC): Commercial non-point source coal emissions (in tons of carbon)

COM npt Petrol (tC): Commercial non-point source petroleum emissions (in tons of carbon)

COM npt NG (tC): Commercial non-point source natural gas emissions (in tons of carbon)

IND npt Coal (tC): Industrial non-point source coal emissions (in tons of carbon)

IND npt Petrol (tC): Industrial non-point source petroleum emissions (in tons of carbon)

IND npt NG (tC): Industrial non-point source natural gas emissions (in tons of carbon)

NRD npt Petrol (tC): Natural resources non-point source petroleum emissions (in tons of carbon)

NRD npt NG (tC): Natural resources non-point source natural gas emissions (in tons of carbon)

RRD npt Coal (tC): Road-related non-point source coal emissions (in tons of carbon)

RRD npt Petrol (tC): Road-related non-point source petroleum emissions (in tons of carbon)

RRD npt NG (tC): Road-related non-point source natural gas emissions (in tons of carbon)

CMV npt Coal (tC): Commercial motor vehicle non-point source coal emissions (in tons of carbon)

CMV npt Petrol (tC): Commercial motor vehicle non-point source petroleum emissions (in tons of carbon)

CMV npt NG (tC): Commercial motor vehicle non-point source natural gas emissions (in tons of carbon)

AIR pt Coal (tC): Air point source coal emissions (in tons of carbon)

AIR pt Petrol (tC): Air point source petroleum emissions (in tons of carbon)

AIR pt NG (tC): Air point source natural gas emissions (in tons of carbon)

COM pt Coal (tC): Commercial point source coal emissions (in tons of carbon)

COM pt Petrol (tC): Commercial point source petroleum emissions (in tons of carbon)

COM pt NG (tC): Commercial point source natural gas emissions (in tons of carbon)

IND pt Coal (tC): Industrial point source coal emissions (in tons of carbon)

IND pt Petrol (tC): Industrial point source petroleum emissions (in tons of carbon)

IND pt NG (tC): Industrial point source natural gas emissions (in tons of carbon)

NRD pt Coal (tC): Natural resources point source coal emissions (in tons of carbon)

NRD pt Petrol (tC): Natural resources point source petroleum emissions (in tons of carbon)

NRD pt NG (tC): Natural resources point source natural gas emissions (in tons of carbon)

RRD pt Coal (tC): Road-related point source coal emissions (in tons of carbon)

RRD pt Petrol (tC): Road-related point source petroleum emissions (in tons of carbon)

RRD pt NG (tC): Road-related point source natural gas emissions (in tons of carbon)

ELC Coal (tC): Electricity coal emissions (in tons of carbon)

ELC Petrol (tC): Electricity petroleum emissions (in tons of carbon)

ELC NG (tC): Electricity natural gas emissions (in tons of carbon)

ONR Gasoline (tC): Off-road gasoline emissions (in tons of carbon)

ONR Diesel (tC): Off-road diesel emissions (in tons of carbon)

ONR NG (tC): Off-road natural gas emissions (in tons of carbon)

CMT (tC): Commercial maritime transportation emissions (in tons of carbon)

RES Total npt FFCO2 (tC): Total residential non-point source fossil-fuel CO2 emissions (in tons of carbon)

COM Total npt FFCO2 (tC): Total commercial non-point source fossil-fuel CO2 emissions (in tons of carbon)

IND Total npt FFCO2 (tC): Total industrial non-point source fossil-fuel CO2 emissions (in tons of carbon)

RRD Total npt FFCO2 (tC): Total road-related non-point source fossil-fuel CO2 emissions (in tons of carbon)

NRD Total npt FFCO2 (tC): Total natural resources non-point source fossil-fuel CO2 emissions (in tons of carbon)

COM Total pt FFCO2 (tC): Total commercial point source fossil-fuel CO2 emissions (in tons of carbon)

IND Total pt FFCO2 (tC): Total industrial point source fossil-fuel CO2 emissions (in tons of carbon)

NRD Total pt FFCO2 (tC): Total natural resources point source fossil-fuel CO2 emissions (in tons of carbon)

RRD Total pt FFCO2 (tC): Total road-related point source fossil-fuel CO2 emissions (in tons of carbon)

RES Total FFCO2 (tC): Total residential fossil-fuel CO2 emissions (in tons of carbon)

COM Total FFCO2 (tC): Total commercial fossil-fuel CO2 emissions (in tons of carbon)

IND Total FFCO2 (tC): Total industrial fossil-fuel CO2 emissions (in tons of carbon)

ELC Total FFCO2 (tC): Total electricity fossil-fuel CO2 emissions (in tons of carbon)

RRD Total FFCO2 (tC): Total road-related fossil-fuel CO2 emissions (in tons of carbon)

CMV Total FFCO2 (tC): Total commercial motor vehicle fossil-fuel CO2 emissions (in tons of carbon)

ONR Total FFCO2 (tC): Total off-road fossil-fuel CO2 emissions (in tons of carbon)

NRD Total FFCO2 (tC): Total natural resources fossil-fuel CO2 emissions (in tons of carbon)

AIR Total FFCO2 (tC): Total air fossil-fuel CO2 emissions (in tons of carbon)

CMT Total CO2 (tC): Total commercial maritime transportation CO2 emissions (in tons of carbon)

Total FFCO2 (tC): Total fossil-fuel CO2 emissions (in tons of carbon)

