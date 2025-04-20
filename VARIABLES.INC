--=========================================================================================================

-- Written by: Amirsaman Rezaeyan - June 2022

--=========================================================================================================

-- Decision Variables for MOO

DEFINES
-- parameter  base value   minimum value   maximum value (leave the unit of each variable after --)
-- Injectors: water injection rate, stb/day
'R_inje_1' 11236.203565420874 0 30000 REAL/ -- stb/day
'R_inje_2' 17959.7545259111 0 30000 REAL/ -- stb/day
'R_inje_3' 1742.5083650459837 0 30000 REAL/ -- stb/day

-- Producers under no gas lift: bottom hole pressure, psia
'BHP_prod_1' 1835.090722245057 950 2200 REAL/ -- psia
'BHP_prod_2' 1990.5533010005272 950 2200 REAL/ -- psia
'BHP_prod_3' 1179.2556373167922 950 2200 REAL/ -- psia
'BHP_prod_4' 1489.9312733026447 950 2200 REAL/ -- psia
'BHP_prod_5' 1124.3673258150523 950 2200 REAL/ -- psia

-- Producers under no gas lift: production rate, stb/day
'R_prod_1' 14560.69984217036 10000 20000 REAL/ -- stb/day
'R_prod_2' 15142.344384136115 10000 20000 REAL/ -- stb/day
'R_prod_3' 16075.448519014382 10000 20000 REAL/ -- stb/day
'R_prod_4' 19488.855372533333 10000 20000 REAL/ -- stb/day
'R_prod_5' 13046.137691733707 10000 20000 REAL/ -- stb/day

-- Producers under gas lift: tubing head pressure, psia
'THP_prod_1' 408.1067456177209 100 800 REAL/ -- psia
'THP_prod_2' 124.07196478065288 100 800 REAL/ -- psia
'THP_prod_3' 563.7655990477874 100 800 REAL/ -- psia
'THP_prod_4' 482.69719554029575 100 800 REAL/ -- psia
'THP_prod_5' 642.5929763527802 100 800 REAL/ -- psia

-- Producers under gas lift (conditional parameter): production rate, stb/day
'R_prodgl_1' 6381.099809299766 1000 10000 REAL/ -- stb/day
'R_prodgl_2' 2763.845761772307 1000 10000 REAL/ -- stb/day
'R_prodgl_3' 4498.095607205338 1000 10000 REAL/ -- stb/day
'R_prodgl_4' 4210.779940242303 1000 10000 REAL/ -- stb/day
'R_prodgl_5' 2268.3180247728637 1000 10000 REAL/ -- stb/day

-- Producers under gas lift: lift gas injection rate, Mscf/day
'R_glif_1' 9881.982429404656 1000 10000 REAL/ -- Mscf/day
'R_glif_2' 1049.6990541124217 1000 10000 REAL/ -- Mscf/day
'R_glif_3' 7561.0645123688855 1000 10000 REAL/ -- Mscf/day
'R_glif_4' 4226.191556898453 1000 10000 REAL/ -- Mscf/day
'R_glif_5' 6609.683141448021 1000 10000 REAL/ -- Mscf/day
/
------------------------------------------------------------------------