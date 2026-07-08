# Interpretable Machine Learning for Early Diabetes Risk Prediction Using Clinical Health Indicators

**A Technical Work Sample**

---

## Abstract

Type 2 diabetes affects hundreds of millions of people worldwide, and delayed diagnosis is
associated with substantially worse long-term outcomes, yet many at-risk individuals are not
flagged until symptoms are already advanced. This project develops and evaluates a set of
machine learning models — Logistic Regression, Random Forest, and XGBoost — to predict
diabetes risk from eight routinely collected clinical indicators, using the publicly available
Pima Indians Diabetes Dataset (768 patients). Beyond predictive performance, the project's
central emphasis is on *trustworthy* machine learning for healthcare: leakage-safe
preprocessing, transparent handling of missing data, model explainability via SHAP
(SHapley Additive exPlanations) and feature importance, and an explicit discussion of bias,
distribution shift, uncertainty, and human oversight. On a held-out test set (20%, n = 154),
Logistic Regression achieved the best overall balance (accuracy 0.792, F1 0.729, ROC-AUC
0.846), narrowly outperforming Random Forest (ROC-AUC 0.829) and XGBoost (ROC-AUC 0.833). We
argue that in a clinical screening context, recall on the positive (diabetic) class and
interpretability of individual predictions matter at least as much as raw accuracy, and we
report both alongside a discussion of why a simpler, transparent model can be preferable to
a marginally more accurate black box. The full pipeline, figures, and analysis are designed
to be reproducible from a single fixed random seed and are organized as an open,
documented repository suitable for technical review.

---

## 1. Introduction

Type 2 diabetes is a chronic condition whose early stages are frequently asymptomatic. By the
time classic symptoms (excessive thirst, fatigue, blurred vision) prompt a clinical visit,
a patient may already have experienced years of elevated blood glucose and attendant
vascular damage. Early identification — ideally before a formal diagnosis — creates a window
for lifestyle intervention, closer monitoring, and pharmacological management that can delay
or prevent the onset of complications such as retinopathy, nephropathy, and cardiovascular
disease.

Machine learning is a natural fit for this problem: diabetes risk is known to depend on a
combination of measurable factors (glucose levels, body mass index, age, family history,
blood pressure) whose *joint* pattern is more informative than any single threshold. However,
healthcare is a domain where a wrong or opaque prediction carries real consequences — a
missed high-risk patient (false negative) can mean a delayed diagnosis, while a model that
cannot explain *why* it flagged a patient is difficult for a clinician to trust or act on.

This project therefore treats predictive accuracy as necessary but not sufficient. Its actual
goal is to demonstrate a complete, *responsible* machine learning workflow: careful data
cleaning that respects the clinical meaning of the data, evaluation metrics chosen for a
screening use case (not just accuracy), explainability at both the global (population) and
local (individual patient) level, and an explicit accounting of where this kind of model
could fail or be misused if deployed carelessly. That combination — practical ML competence
plus a genuine safety/trustworthiness lens — is the point of the exercise, not simply
achieving a high leaderboard score on a well-worn dataset.

## 2. Related Work

Machine learning approaches to diabetes prediction using the Pima Indians Diabetes Dataset
are well represented in the applied ML literature, with logistic regression, decision trees,
support vector machines, and ensemble methods all reported at accuracies in the roughly
75–90% range depending on preprocessing choices, evaluation protocol, and — importantly — how
rigorously train/test leakage is avoided (Smith et al., 1988; a substantial body of later work
builds on the same dataset with varying preprocessing). Random Forest and gradient boosting
methods (e.g., XGBoost) have generally been reported to match or modestly exceed linear
baselines on this dataset (Chen & Guestrin, 2016, describe the XGBoost algorithm itself).

Two related threads inform this project more directly. First, explainable AI (XAI) methods —
particularly SHAP, which unifies several prior feature-attribution methods under a
game-theoretic framework (Lundberg & Lee, 2017) — have become a standard tool for making
tree-based and other complex models auditable, and are increasingly expected in clinical ML
work rather than treated as optional. Second, a growing literature on trustworthy and
responsible AI in healthcare emphasizes that predictive performance alone is an incomplete
success criterion: fairness across subgroups, robustness to distribution shift between
training and deployment populations, calibrated uncertainty, and the preservation of
meaningful human oversight are treated as first-class requirements rather than afterthoughts
(Obermeyer et al., 2019, is a widely cited example of a deployed healthcare model that
produced racially biased outcomes despite strong aggregate accuracy — a cautionary precedent
directly relevant to this project's design choices). This project draws on both threads:
the modeling and explainability methodology is standard applied ML practice, and the
"AI Safety Discussion" section below is written in the spirit of the second literature.

## 3. Dataset Description

The **Pima Indians Diabetes Dataset**, originally collected by the National Institute of
Diabetes and Digestive and Kidney Diseases and widely redistributed via the UCI Machine
Learning Repository and Kaggle, contains 768 records of female patients of Pima Indian
heritage, all at least 21 years old. Each record has 8 clinical features and one binary
outcome label:

| Feature | Description | Unit |
|---|---|---|
| Pregnancies | Number of times pregnant | count |
| Glucose | Plasma glucose concentration, 2-hour oral glucose tolerance test | mg/dL |
| BloodPressure | Diastolic blood pressure | mmHg |
| SkinThickness | Triceps skinfold thickness | mm |
| Insulin | 2-hour serum insulin | mu U/mL |
| BMI | Body mass index | kg/m² |
| DiabetesPedigreeFunction | A function scoring likelihood of diabetes based on family history | unitless score |
| Age | Age | years |
| **Outcome** | 1 = diabetes diagnosed within 5 years, 0 = not diagnosed | binary label |

The dataset is imbalanced: 500 negative cases (65.1%) and 268 positive cases (34.9%), a
ratio of roughly 1.87:1. This is an important consideration for both model training and
evaluation metric choice (see Section 6).

**A known and important limitation of this dataset for this project's own AI Safety
discussion:** the population is a single ethnic and sex-specific group (Pima Indian women).
Any predictive model trained here is not automatically valid for other populations — this
point is treated in depth in Section 10.

## 4. Data Preprocessing

The raw data contains a data-quality quirk common to many real-world clinical datasets:
several fields use the numeral `0` to represent a *missing* measurement rather than a true
zero reading, because `0` is not a physiologically plausible value for these quantities:

| Field | Zero count | % of dataset | Physiological interpretation of 0 |
|---|---|---|---|
| Insulin | 374 | 48.7% | Not survivable — treated as missing |
| SkinThickness | 227 | 29.6% | Not survivable — treated as missing |
| BloodPressure | 35 | 4.6% | Not survivable — treated as missing |
| BMI | 11 | 1.4% | Not survivable — treated as missing |
| Glucose | 5 | 0.7% | Not survivable — treated as missing |

Treating these as literal zeros (as a naive pipeline sometimes does) would badly distort the
learned relationship between these features and diabetes risk — for example, a model would
otherwise learn that an "Insulin" reading of 0 is common and largely uninformative, when in
fact nearly half of those readings are simply missing data, concentrated in ways that are not
necessarily random (missingness itself can correlate with which clinic or era the data was
collected in).

**Pipeline steps, in order (see `src/preprocessing.py`):**

1. **Replace implausible zeros with `NaN`** for Glucose, BloodPressure, SkinThickness,
   Insulin, and BMI.
2. **Train/test split first** (80/20, stratified on the outcome label, fixed
   `random_state=42`) — performed *before* any imputation or scaling.
3. **Median imputation, fit on the training split only**, then applied to both the training
   and test splits. Fitting imputation statistics on the full dataset (including test data)
   is a common and easy-to-miss form of data leakage that inflates reported test performance;
   this pipeline explicitly avoids it.
4. **Feature scaling** (`StandardScaler`) fit on the training split only, used for Logistic
   Regression (tree-based models do not require feature scaling).

This "fit-on-train-only" discipline is applied consistently and is one of the more
easily-overlooked sources of over-optimistic results in applied ML work — flagging and
avoiding it is itself part of this project's trustworthiness argument.

## 5. Exploratory Data Analysis

Key findings from EDA (full figures in `figures/`):

- **Class imbalance** (`figures/class_balance.png`): 65% negative / 35% positive. This
  motivates using `class_weight="balanced"` (Logistic Regression, Random Forest) and
  `scale_pos_weight` (XGBoost) during training, and prioritizing recall/F1/ROC-AUC over raw
  accuracy during evaluation, since a trivial "always predict negative" classifier would
  already score 65% accuracy while being clinically useless.
- **Correlation structure** (`figures/correlation_heatmap.png`): Glucose shows the strongest
  positive correlation with the diabetes outcome, consistent with its direct clinical role
  in diagnosis. BMI, Age, and Pregnancies show moderate positive correlations. Skin
  Thickness and Insulin are moderately correlated with each other and with BMI, suggesting
  some redundancy among adiposity-related measurements.
- **Feature distributions by outcome** (`figures/feature_distributions.png`): Glucose shows
  the clearest separation between the two outcome classes, followed by BMI and Age. Blood
  Pressure shows the weakest separation, suggesting it is a comparatively weak standalone
  predictor in this dataset.
- **Outlier inspection** (`figures/outlier_boxplots.png`): Insulin and DiabetesPedigreeFunction
  show pronounced right-skew with high-value outliers; these were retained (rather than
  removed) since they represent plausible extreme physiological measurements, and tree-based
  models are robust to this kind of skew.

## 6. Feature Engineering

All engineered features were chosen for **clinical plausibility**, in keeping with the
project's interpretability goal — the aim is not an unconstrained polynomial or automated
feature-search expansion, but a small set of derived variables a clinician could recognize
and sanity-check (see `src/feature_engineering.py`):

- **BMI category** (Underweight / Normal / Overweight / Obese) — standard WHO BMI thresholds.
- **Glucose category** (Normal / Prediabetic / Diabetic range) — standard ADA screening
  thresholds for fasting/OGTT glucose.
- **Age group** (<30 / 30–44 / 45–59 / 60+) — coarse risk-band bucketing aligned with typical
  diabetes screening age brackets.
- **Glucose × BMI interaction** — motivated by the known joint role of hyperglycemia and
  adiposity in insulin resistance.
- **Age × Pregnancies interaction** — gestational diabetes history compounds with age-related
  risk.

After one-hot encoding the categorical buckets, the feature set grows from 8 raw clinical
variables to **18 engineered features**. Notably, in the fitted Random Forest, the
**Glucose × BMI interaction term emerged as the single most important feature** (importance
≈ 0.248, ahead of raw Glucose at ≈ 0.188) — a result consistent with the clinical intuition
that motivated including it, and a useful sanity check that the engineered features are
capturing real signal rather than noise.

## 7. Machine Learning Models

Three models spanning a spectrum of interpretability and expressiveness were trained,
each addressing the ~1.87:1 class imbalance directly rather than via synthetic
oversampling (to keep every training example traceable to a real patient record):

- **Logistic Regression** (`class_weight="balanced"`, `max_iter=2000`) — the transparency
  baseline. Every prediction can be decomposed into a weighted sum of standardized features,
  and coefficients are directly interpretable as log-odds contributions.
- **Random Forest** (`n_estimators=300, max_depth=6, min_samples_leaf=5,
  class_weight="balanced"`) — a bagged ensemble of decision trees, capable of capturing
  nonlinearities and interactions without manual specification, while still supporting exact
  SHAP value computation and built-in Gini-based feature importance.
- **XGBoost** (`n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
  colsample_bytree=0.8, scale_pos_weight` set to the train-set class ratio) — gradient-boosted
  trees, generally the strongest baseline for structured/tabular data in the applied ML
  literature.

Hyperparameters were kept deliberately modest (shallow trees, moderate ensemble size) given
the dataset's small size (614 training examples) to reduce overfitting risk rather than
chase marginal accuracy gains through aggressive tuning.

## 8. Model Evaluation

Evaluated on the held-out test set (n = 154, stratified to preserve the ~65/35 class split):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.792** | 0.672 | **0.796** | **0.729** | **0.846** |
| Random Forest | 0.747 | 0.612 | 0.759 | 0.678 | 0.829 |
| XGBoost | 0.747 | 0.642 | 0.630 | 0.636 | 0.833 |

*(See `figures/confusion_matrix_*.png`, `figures/roc_curves_comparison.png`,
`figures/metrics_comparison.png`, and `reports/metrics_summary.csv` for full detail.)*

**Why recall matters most here:** in a screening context, a false negative (a diabetic
patient predicted as non-diabetic) means a missed opportunity for early intervention, while a
false positive typically leads to a confirmatory follow-up test — a comparatively low-cost
error. Logistic Regression's higher recall (0.796 vs. 0.759 for Random Forest and 0.630 for
XGBoost) is therefore a meaningful advantage for this use case, not just a marginal
statistic. Combined with its full transparency (Section 9), this is why we would recommend
Logistic Regression as the leading candidate among the three for a screening-support role in
this project, despite the somewhat greater theoretical flexibility of the tree ensembles.

That said, none of these models is close to good enough for unsupervised clinical
decision-making — recall of 0.796 still means roughly one in five diabetic patients in this
test set would be missed by the model alone. This is addressed directly in Sections 10–11.

## 9. Explainability

### 9.1 Global explanations: feature importance and SHAP

- **Random Forest feature importance** (`figures/rf_feature_importance.png`): the
  Glucose × BMI interaction, raw Glucose, BMI, Age, and the "Diabetic-range Glucose" category
  dominate, in that order — consistent with clinical knowledge and with the correlation
  analysis in Section 5.
- **SHAP summary (beeswarm) plot** (`figures/shap_summary_beeswarm.png`): shows both the
  magnitude and direction of each feature's effect across all test patients. High Glucose
  values (red) push predictions toward higher diabetes risk (positive SHAP value), while low
  Glucose values (blue) push toward lower risk — the expected direction, visible directly in
  the plot rather than asserted.
- **SHAP bar plot** (`figures/shap_feature_importance_bar.png`): a population-level ranking of
  mean absolute SHAP value per feature, corroborating the Random Forest's built-in importance
  ranking via an independent (game-theoretic) attribution method.
- **Logistic Regression coefficients** (`figures/logreg_coefficients.png`): directly
  interpretable log-odds contributions per standardized feature — the most transparent
  explanation available among the three models, since no post-hoc approximation is needed.

### 9.2 Local explanation: a single patient

`figures/shap_local_decision_plot_patient0.png` shows a SHAP decision plot for one individual
test-set patient, tracing how each feature moved that specific patient's predicted risk away
from the model's baseline (expected) prediction. This is the level of explanation a clinician
would actually need at the point of care — not "what matters on average," but "why did the
model flag *this* patient."

### 9.3 Why interpretability matters in healthcare

A model that cannot explain an individual prediction places a clinician in an uncomfortable
position: either trust an opaque score, or ignore it. Interpretability serves several
concrete purposes here: it lets a clinician **sanity-check** a prediction against their own
reasoning (e.g., "yes, this patient's risk is glucose-driven, that's consistent with what I
observed"); it supports **actionable follow-up** (a Glucose-driven risk score suggests an
HbA1c or repeat OGTT, while a BMI-driven score suggests different counseling); and it creates
an **audit trail** for cases where the model is later found to have been wrong, which is a
precondition for iterative improvement and accountability rather than a black-box result
patients and clinicians are simply asked to accept.

## 10. AI Safety Discussion

This section is the core of the project's trustworthiness argument, and is treated with the
same rigor as the modeling work above.

### 10.1 Bias and fairness

The training data is drawn entirely from **female patients of Pima Indian heritage, age
21+**. This is a critical limitation, not a footnote: applying this model, unmodified, to
other populations (different sex, ethnicity, age range) would be a misuse of the model, not
merely a reduction in accuracy. Diabetes risk factors and their relative importance can
differ meaningfully across populations, and a model trained on one narrow demographic group
offers no guarantee of fair or even correct performance elsewhere. Any real deployment would
require, at minimum, revalidation on the target population's own data, and ideally a fairness
audit disaggregating performance (recall, precision, calibration) across relevant subgroups —
directly motivated by documented real-world cases of healthcare algorithms that were accurate
in aggregate but systematically biased for specific subpopulations (Obermeyer et al., 2019).

### 10.2 Distribution shift

Clinical measurement practices are not static: which glucose assay is used, how blood
pressure is measured, or which population is being screened can all shift between when a
model was trained and when it is deployed, or over time within a single deployment. Such
shift can silently degrade a model's calibration and accuracy even if the model's code is
completely unchanged. This risk is compounded by the small size of the training data (614
examples) — a model trained on this little data is likely to be more sensitive to shift than
one trained on a larger, more diverse corpus. A responsible deployment would include ongoing
monitoring for shift (e.g., tracking the distribution of incoming feature values against the
training distribution) rather than treating the model as a "train once, deploy forever"
artifact.

### 10.3 Model uncertainty

None of the three models' reported metrics tell a clinician how *confident* the model is
about any single patient's prediction — a raw predicted probability from `predict_proba` is
not automatically a well-calibrated probability. Two concrete steps that a genuinely
deployment-ready version of this project would add: (1) a calibration curve
(reliability diagram) checking whether, e.g., patients assigned a 70% predicted risk are
diabetic roughly 70% of the time; and (2) surfacing prediction confidence alongside the
prediction itself, so that low-confidence cases can be routed differently (e.g., flagged for
additional testing rather than treated as a confident risk score). Both are natural next
steps and are called out explicitly in Section 12 (Future Work) rather than treated as solved
by this project.

### 10.4 Human oversight

This model — and any model built the same way — should be positioned as a
**decision-support tool that augments clinical judgment, not a diagnostic authority that
replaces it.** A predicted high-risk score should route a patient toward a confirmatory
diagnostic test (HbA1c, fasting glucose, oral glucose tolerance test), not toward an automatic
diagnosis, treatment change, or insurance/administrative decision. Meaningful human oversight
also means clinicians need a channel to flag cases where a model's prediction (or SHAP
explanation) doesn't match their own clinical reasoning — both to protect individual patients
and to surface systematic model errors over time.

### 10.5 Responsible deployment in clinical settings

Beyond the model itself, responsible deployment would require: a documented **intended use**
statement (which population, which decision the model is meant to support, and — just as
importantly — which decisions it is explicitly *not* validated for); a **monitoring plan**
for performance and distribution drift over time; **informed consent and transparency**
with patients about when and how an algorithmic risk score is used in their care; and
**regulatory awareness** — a tool used to inform real clinical decisions in the US would
likely fall under FDA Software as a Medical Device (SaMD) considerations, which impose
requirements well beyond what an academic-style benchmarking exercise like this one
satisfies. This project is explicitly a research prototype and technical work sample, not a
clinical tool, and should not be treated as validated for any deployment use.

## 11. Limitations

- **Single, narrow, small dataset.** 768 patients, one demographic group, collected some
  decades ago — not representative of a modern, diverse patient population.
- **Substantial missingness in key fields** (nearly half of Insulin values were imputed),
  which, despite careful handling, is still a meaningfully different signal than genuinely
  observed measurements.
- **Median imputation is simple** and does not model uncertainty introduced by imputation;
  more sophisticated approaches (e.g., multiple imputation, or model-based imputation) could
  better represent that uncertainty, at the cost of added complexity.
- **No external validation set** from a different data source or time period — all reported
  performance is on a held-out split from the *same* dataset, which cannot detect distribution
  shift by construction.
- **No calibration analysis** was performed in this iteration (see Section 10.3); reported
  probabilities should not yet be interpreted as well-calibrated risk percentages.
- **Modest overall performance.** None of these models exceed ~0.85 ROC-AUC; state-of-the-art
  results on this exact dataset in the literature are in a similar range, reflecting the
  dataset's inherent size and noise limitations rather than an implementation shortfall, but
  this ceiling should be stated plainly rather than implied to be higher than it is.

## 12. Future Work

- Add probability calibration analysis (reliability diagrams, Brier score) and, if needed,
  post-hoc recalibration (e.g., Platt scaling or isotonic regression).
- Evaluate subgroup performance once a more demographically diverse dataset is available, to
  directly test the fairness concerns raised in Section 10.1 rather than only flag them.
- Explore uncertainty-aware modeling (e.g., conformal prediction) to accompany each
  prediction with a statistically grounded confidence interval rather than a single point
  probability.
- Add a lightweight external validation step against a second, independently collected
  diabetes dataset to probe distribution-shift robustness directly.
- Build a minimal clinician-facing interface presenting the SHAP local explanation alongside
  the prediction, to evaluate (qualitatively, with clinical input) whether the explanations
  are actually useful for real decision-making rather than only theoretically interpretable.

## 13. Conclusion

This project set out to demonstrate that a machine learning system for a sensitive domain
like healthcare should be judged on more than a single accuracy number. Working with the
Pima Indians Diabetes Dataset, we built a full, reproducible pipeline — leakage-safe
preprocessing, clinically motivated feature engineering, three models spanning the
interpretability/expressiveness spectrum, evaluation metrics chosen for a screening use case,
and both global and local explainability via SHAP. Logistic Regression, the most transparent
of the three models, achieved the best recall (0.796) and ROC-AUC (0.846) on held-out data,
reinforcing that interpretability and strong performance are not necessarily in tension. Just
as importantly, the AI Safety Discussion (Section 10) treats bias, distribution shift,
uncertainty, human oversight, and responsible deployment as integral parts of the technical
work, not an appendix — reflecting the view that a model's trustworthiness is inseparable
from its predictive performance when the domain is healthcare.

---

## References

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of
the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*
(pp. 785–794). Association for Computing Machinery. https://doi.org/10.1145/2939672.2939785

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions.
In *Advances in Neural Information Processing Systems 30* (pp. 4765–4774). Curran Associates.

Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an
algorithm used to manage the health of populations. *Science, 366*(6464), 447–453.
https://doi.org/10.1126/science.aax2342

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M.,
Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D.,
Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python.
*Journal of Machine Learning Research, 12*, 2825–2830.

Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using
the ADAP learning algorithm to forecast the onset of diabetes mellitus. In *Proceedings of the
Annual Symposium on Computer Application in Medical Care* (pp. 261–265). American Medical
Informatics Association.

UCI Machine Learning Repository. (n.d.). *Pima Indians Diabetes Database*. National Institute
of Diabetes and Digestive and Kidney Diseases. Retrieved 2026, from
https://archive.ics.uci.edu/

World Health Organization. (2016). *Global report on diabetes*. World Health Organization.
https://www.who.int/publications/i/item/9789241565257
