# Experiment Log

## EXP-001 — Basic Dataset Inspection
**Status:** ACCEPTED

### Findings
- Data contains timestamp plus four target variables:
  - X error
  - Y error
  - Z error
  - satellite clock error
- No missing values were found.
- Exact duplicate rows exist in the MEO datasets.
- Sampling is irregular and differs between datasets.

### Decision
- Normalize column names.
- Parse and preserve UTC timestamps.
- Remove exact duplicate rows during preprocessing.
- Do not assume a fixed sampling interval.

---

## EXP-002 — Automatic IQR Outlier Removal
**Status:** REJECTED

### Problem
IQR-based filtering flagged many observations, especially in GEO data.

### Why it was rejected
Large error values occurred in temporal clusters and across multiple variables. They may represent genuine error regimes/events rather than isolated bad measurements.

### Countermeasure
Do not automatically delete statistical outliers.
Treat unusual values as possible real regimes and let later modeling experiments determine their usefulness.

---

## EXP-003 — Fixed 30-Minute Segmentation
**Status:** REJECTED

### Problem
Segmenting whenever the time gap exceeded 30 minutes produced excessive segments.

Examples:
- GEO Train was split into many segments despite valid 1–2 hour sampling.
- MEO Train was almost completely fragmented.

### Why it was rejected
A large time gap is not automatically a data error or a new physical regime.

### Countermeasure
Preserve actual timestamps and elapsed time.
Do not use one arbitrary global gap threshold as a segmentation rule.

---

## EXP-004 — Sampling Pattern Analysis
**Status:** ACCEPTED

### Findings
Datasets have distinct native sampling regimes:

- GEO: mainly ~15 minute bursts plus longer gaps
- MEO: mainly ~60 minute sampling
- MEO2: mainly ~10 minute sampling

### Decision
The model must handle irregular sampling and must not globally resample all datasets to one fixed interval.

---

## EXP-005 — Contemporaneous Correlation Analysis
**Status:** ACCEPTED WITH CAUTION

### Findings
Some datasets show strong X/Y/Z/clock relationships, while others show weak relationships.

### Decision
Multivariate modeling is justified as a candidate approach.

### Caution
Contemporaneous correlation alone does not establish temporal predictability.

---

## EXP-006 — Row-Lag Autocorrelation
**Status:** REJECTED AS MODEL DESIGN BASIS

### Problem
Lag 1 means different elapsed times in different datasets.

Examples:
- MEO2 lag 1 is commonly ~10 minutes.
- MEO lag 1 is commonly ~60 minutes.

### Why it was rejected
Row order is not equivalent to elapsed time.

### Countermeasure
Replace row-based lag analysis with actual elapsed-time analysis.

---

## EXP-007 — Time-Aware Gap-Binned Autocorrelation
**Status:** ACCEPTED

### Findings
Temporal dependence differs by dataset:

- GEO shows strongest dependence around its ~15 minute sampling regime.
- MEO shows strong orbital dependence around its ~60 minute sampling regime.
- MEO2 shows strong short-term dependence around ~10 minutes and meaningful dependence at longer intervals.

### Decision
Temporal memory is dataset/regime dependent.
The model should explicitly represent elapsed time.

---

## EXP-008 — Time-Aware Cross-Variable Correlation
**Status:** ACCEPTED

### Findings
Past values of one error component can correlate with future values of other components.

Examples:
- GEO Train, ~15 min: strong cross-variable correlations across X/Y/Z/clock.
- MEO Train, ~60 min: strong orbital temporal persistence and meaningful cross-variable relationships.
- MEO2 Train, ~10 min: strong Y/Z persistence and several cross-variable relationships.

### Decision
Use a multivariate model rather than four completely independent models.

### Important limitation
Small test datasets produce unstable correlations. Training-set relationships should therefore drive architecture decisions; test-set correlations are used only as supporting evidence.

---

# Current EDA Architecture Requirements

The eventual model should:

1. Predict all four errors jointly.
2. Preserve real timestamps.
3. Include elapsed-time information.
4. Handle irregular sampling.
5. Avoid assuming a universal sampling interval.
6. Allow temporal relationships to differ between datasets/regimes.
7. Be evaluated using the competition's residual-based metrics, especially Shapiro-Wilk behavior, mean, standard deviation, and Q-Q plots.

## Current Model Hypothesis

**Multivariate + time-aware sequence model**

Transformer, GRU, LSTM, or another architecture will be compared experimentally rather than selected in advance.

## Next Phase

Model-development experiments will begin only after establishing the validation strategy and leakage-safe train/validation splits.
