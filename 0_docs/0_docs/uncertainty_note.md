# Uncertainty Note (Template)

## Summary
State combined standard uncertainty for velocity measurements at key probe points.

## Components
- Repeatability: standard deviation from N independent image pairs.
- Algorithmic bias: RMSE from synthetic PIV using CFD ground truth.
- Seeding/illumination variability: measured by varying seeding density and illumination.

## Combined budget
u_combined = sqrt(u_repeat^2 + u_bias^2 + u_seeding^2)

## Reporting
Report mean velocity ± u_combined at selected probe points and include an error map.
