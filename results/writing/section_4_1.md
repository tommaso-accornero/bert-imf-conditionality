## 4.1 BERT Classification Performance

The 13-category classifier achieves a macro-F1 score of 0.898 on the held-out test set, surpassing the 0.80 target. Overall accuracy is 0.939; however, this metric is reported only as a secondary measure. Due to a class imbalance of 43:1 between the most and least frequent categories, overall accuracy is disproportionately influenced by strong performance on majority classes and does not reflect the model's effectiveness on minority categories, which are most relevant to the mission creep analysis. Macro-F1, which assigns equal weight to each category regardless of size, is therefore the appropriate primary metric. For comparison, the best baseline, TF-IDF with Logistic Regression, achieves a macro-F1 of 0.774, while TF-IDF with Random Forest achieves 0.733. The BERT model's improvement of 12.4 macro-F1 points over the best baseline demonstrates a substantial performance gain.

Table 1 presents per-category F1 scores on the test set, sorted in descending order.

**Table 1. Per-category F1, 13-category model (test set)**

| Category | n_test | F1    |
|----------|--------|-------|
| DEB      | 921    | 0.989 |
| FIN      | 1,226  | 0.963 |
| EXT      | 411    | 0.954 |
| POV      | 77     | 0.953 |
| FP       | 942    | 0.943 |
| RTP      | 559    | 0.930 |
| OTH      | 39     | 0.921 |
| LAB      | 238    | 0.920 |
| PRI      | 166    | 0.902 |
| SOE      | 346    | 0.867 |
| ENV      | 29     | 0.831 |
| SP       | 55     | 0.796 |
| INS      | 140    | 0.699 |

DEB is the strongest category, with an F1 score of 0.989. Debt management conditions, such as borrowing ceilings, debt reporting obligations, and debt sustainability frameworks, employ formulaic language that is largely unique to this category, facilitating straightforward classification. INS is the weakest category, with an F1 score of 0.699. This lower performance reflects genuine semantic overlap: conditions involving fiscal transparency, civil service wage controls, and public enterprise governance use institutional language that spans the boundaries between INS, FP, LAB, and SOE. This is not a post-hoc rationalisation; the same ambiguity was documented in Section 3.3 as the basis for eight medium-confidence descriptor mappings during the construction of the MONA lookup labels. Manual validation also identified the FP/INS boundary as the most common source of human uncertainty in the 100-condition coding exercise.

Two minority categories, SP and ENV, which are particularly relevant to the non-core expansion finding, benefit substantially from the class-weighting correction applied during training. Without balanced weights, a model trained on 8,516 FIN examples and 55 SP examples would have limited incentive to learn the minority classes. With class-weighting correction, SP improves from a TF-IDF baseline of 0.394 to 0.796, and ENV from 0.439 to 0.831, representing improvements of approximately 40 points each. All 13 categories exceed the 0.55 threshold, below which per-category performance would be considered insufficient for the mission creep analysis.

The 7-category classifier achieves a macro-F1 score of 0.918, with all categories exceeding 0.80 (see Table 2). GOVERNANCE is the weakest category at 0.814, which is expected because this category combines INS, ENV, and OTH—three sub-components with distinct linguistic profiles—into a single label. The 7-category model is used for the composition analysis in Section 4.2, where the GOVERNANCE share serves as the primary mission creep indicator. The 13-category model underpins the regression outcome variable.

**Table 2. Per-category F1, 7-category model (test set)**

| Category      | n_test | F1    |
|---------------|--------|-------|
| FISCAL        | 2,422  | 0.966 |
| FINANCIAL     | 1,226  | 0.957 |
| EXTERNAL      | 411    | 0.954 |
| LABOUR        | 238    | 0.926 |
| PRIVATISATION | 512    | 0.914 |
| SOCIAL        | 132    | 0.893 |
| GOVERNANCE    | 208    | 0.814 |

Three validation exercises, each addressing a distinct generalization concern, provide converging evidence that the classifier is reliable for mission creep analysis. The first validation addresses the methodological boundary at 2019–2020: BERT was applied to 207 hand-coded IMF Monitor program-years from 2015–2019, and program-level scope scores were compared against the hand-coded ground truth. In this comparison, 99.0% of program-years had BERT-predicted and hand-coded scope scores that agreed within one policy area, with a mean absolute error of 0.097 and a bias of +0.077. The positive bias direction, indicating that BERT marginally overstates scope relative to human coders, is conservative for the mission creep test. The second validation addresses generalization to MONA’s taxonomy: BERT predictions on the full MONA 2020–2024 sample agree with the independent descriptor-based lookup labels in 76.9% of cases at the 7-category level, with disagreements concentrated in boundary cases previously identified as ambiguous in Section 3.3. The third validation examines agreement with human judgment on previously unseen MONA conditions: 100 MONA 2020–2024 conditions were manually coded by the author, and BERT agreed with the primary judgment in 86% of cases (89% when excluding the 12 conditions identified as genuinely ambiguous). No disagreements occurred at the GOVERNANCE/FISCAL boundary, which is the most consequential for the mission creep analysis. Across the full MONA 2020–2024 inference set, mean prediction confidence is 0.978, with only 2.5% of predictions falling below the 0.70 review threshold.
