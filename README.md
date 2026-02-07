# Credit Risk Decision System


## EDA insights

- Default rate ≈ 8%
- Housing/property fields are sparse (Missing ≈ 70%)
- dataset mostly numeric (106 numeric, 16 cat)

### Drop Candidates

- Very high missing housing columns
- Duplicate signals (DAYS_BIRTH later)

### Transform Candidates

- Log income
- Log credit
- Sentinel replacement

### Impute Candidates

- Medium missing financial columns

### Feature Engineering Candidates

- Credit ratios
- Age
- Missing flags

### High Signal Features
 
- EXT_SOURCE
- Financial ratios
- Education