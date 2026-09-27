<!-- meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). Source-available, not Open Source. No GPL-licensed source was consulted. -->

# G-RULES — the AM-9 gate of rules.py

Date 2026-09-24 - binary sha256 `054bba67c8650082` - git HEAD `7e8e14ff866e`.

## Part 1 - PASS

- run A: exit 0, growth 1.265, h/t1 41.958, T/(cf*h) 0.999624, n_layers 8, full 1.0, mean 1.0, dropped 
- run B: exit 0, growth 1.266, h/t1 41.958, T/(cf*h) 1.003318, n_layers 8, full 0.0, mean 0.996693, dropped 
- run C: exit 1, growth 1.265, h/t1 60.06, T/(cf*h) 0.698331, G5 0.049950
- run D: exit 0, growth 1.265, h/t1 59.94, T/(cf*h) 0.699729, n_layers 0, full 0.0, mean 0.0, dropped patch "cube": the thickness fell below min_thickness * T = 1.071e-3
- run F: exit 0, growth 1.2, h/t1 44.0, T/(cf*h) 0.749958, n_layers 0, full 0.0, mean 0.0, dropped patch "cube": the thickness fell below min_thickness * T = 1.148e-3

## Part 2A - PASS

- 12 rows, 11 applied, 1 refused ['A-1-009']
- fired {'R-BUDGET': 8, 'R-CURV': 7, 'R-DOM': 12, 'R-FEAT': 12, 'R-WIN': 12, 'R-YP': 12}, wall levels {'4': 1, '5': 10}

## Part 2B - PASS

- 12 rows, 12 applied, 0 refused 
- fired {'R-BUDGET': 4, 'R-CURV': 8, 'R-DOM': 12, 'R-FEAT': 6, 'R-WIN': 12, 'R-YP': 12}, wall levels {'6': 12}

## Part 2D - PASS

- 12 rows, 12 applied, 0 refused 
- fired {'R-BUDGET': 6, 'R-CURV': 3, 'R-DOM': 12, 'R-FEAT': 8, 'R-PLANE': 4, 'R-WIN': 12, 'R-YP': 12}, wall levels {'4': 2, '5': 5, '6': 5}

## Part 2E - PASS

- 12 rows, 12 applied, 0 refused 
- fired {'R-BUDGET': 4, 'R-CURV': 1, 'R-DOM': 12, 'R-FEAT': 7, 'R-GAP': 7, 'R-WIN': 12, 'R-YP': 12}, wall levels {'4': 3, '5': 8, '6': 1}

## Part 2F - PASS

- 12 rows, 12 applied, 0 refused 
- fired {'R-DOM': 12, 'R-FEAT': 9, 'R-PLANE': 3, 'R-WIN': 12, 'R-YP': 12}, wall levels {'3': 2, '4': 4, '5': 6}

## Part 2 - PASS

- 60 rows, 59 applied, 1 refused

## Part 3 - PASS

- 35 commensurate rows, R-PLANE applied 34 (box_c 21/21, plate_c 8/8, lcorner_c 5/6), abstained ['F-1-009']
- live box_c (D-1-002): exit 0, snap max_over_h 2.404564606299065e-14, layers {'dropped': None, 'full_area_frac': 1.0, 'mean_frac': 1.0, 'n_layers': 8}
- live plate_c (F-1-006): exit 0, snap max_over_h 1.0371936281019459e-14, layers {'dropped': None, 'full_area_frac': 1.0, 'mean_frac': 1.0, 'n_layers': 8}
- live lcorner_c (F-1-003): exit 0, snap max_over_h 9.388759126347192e-13, layers {'dropped': None, 'full_area_frac': 1.0, 'mean_frac': 1.0, 'n_layers': 8}

On a box the delivered stack needs h/t1 <= about 42.9: runs D and F pass the §D.3 early check and still lose their layers, so R-PLANE uses 0.70 of the G5 edge.

