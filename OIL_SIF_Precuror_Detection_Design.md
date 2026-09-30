# OIL SIF Precursor Detection
## Technical Solution Design

### Purpose

This document describes a system for analysing Oil India Limited (OIL) unsafe-act and unsafe-condition observations, near-miss reports, and incident reports. It explains how narrative text can be converted into structured safety information, assessed for serious-injury-and-fatality (SIF) potential, and presented to HSE teams for investigation and decision support.

The system is intended to help experts analyse reports at scale, not replace their judgement. Every prediction and pattern should link back to the report evidence behind it. The original report remains the source of truth; AI-generated fields are stored separately, shown with their evidence, and made available for human review.

## 1. Executive Overview

A report often contains more useful safety information than its final incident category or outcome suggests. In a few sentences, a worker may describe an activity, a hazardous energy source, a control that should have prevented exposure, a failure in that control, and a possible serious consequence. A near miss can therefore have high SIF potential even when no one was injured.

The system makes those details easier to find and compare. It processes each report, extracts relevant safety factors, estimates SIF potential, identifies applicable Life-Saving Rules, and groups recurring precursor patterns. It then presents the results through site, activity, and time-based analysis in an HSE dashboard.

The overall flow is:

**Safety report → quality checks → NLP extraction → structured safety factors → SIF and Life-Saving Rule predictions → precursor analysis → probabilistic analysis → HSE review → dashboard analytics**

The aim is not simply to assign a label to a piece of text. It is to make important information in a large collection of reports searchable, measurable, and reviewable.

## 2. System Inputs

The primary inputs are OIL's unsafe-act and unsafe-condition observations, near-miss reports, and incident reports. A report may include free-text narrative, structured fields, or both.

Useful metadata may include:

- Report date and time
- Site and location
- Activity or work type
- Report category
- Equipment or asset
- Existing hazard, consequence, or control fields
- Reporter or review status, where access and privacy rules permit

The system should accept variation in the completeness and quality of reports. Missing information should remain unknown rather than being guessed by the model.

## 3. End-to-End Architecture

The solution is organised into the following layers:

1. Data ingestion and stable report identification
2. Quality checks and terminology normalization
3. Domain-specific NLP
4. Safety ontology and taxonomy
5. SIF-potential prediction
6. Life-Saving Rule mapping
7. Precursor-pattern discovery
8. Probabilistic barrier analysis
9. Uncertainty handling and HSE review
10. Site, activity, and temporal analytics
11. Interactive HSE dashboard

Each layer should produce versioned, inspectable outputs so that a result can be traced from the dashboard back to the report and model version that produced it.

## 4. Data Ingestion and Preservation

Assign every report a stable identifier when it enters the system. Keep the source narrative unchanged. Store cleaned text, extracted information, predictions, and review decisions in separate fields rather than overwriting the source.

A report record could contain:

- `report_id`
- `timestamp`, `site`, `location`, `activity`, and `report_type`
- Original structured fields and raw narrative
- Normalized narrative used for analysis
- Extracted safety factors and their evidence spans
- Model predictions, probabilities, and model versions
- Review status and any HSE corrections

This separation makes the system auditable. An HSE user should be able to move from a prediction to the exact report text that supports it, while retaining the original wording for context.

## 5. Data Quality and Terminology Normalization

Industrial reports use abbreviations, spelling variations, shorthand, and local terminology. The system should map equivalent phrases to consistent concepts without erasing the original wording.

For example, “LOTO”, “lock out/tag out”, and “lockout/tagout” can all map to the internal concept `ENERGY_ISOLATION`. The original phrase should remain available alongside the normalized concept.

Initial quality checks should identify:

- Duplicate reports
- Empty or very short narratives
- Invalid or inconsistent dates
- Inconsistent report categories
- Missing site, activity, or other important metadata
- Potential leakage between training and test data

Quality issues should be recorded and surfaced for review. They should not be silently corrected when the correction could change the safety meaning.

## 6. Domain NLP: Understanding the Narrative

Natural language processing (NLP) turns report text into information the rest of the system can use. A Transformer model can help interpret context, including distinctions such as “the equipment was isolated” versus “the equipment was not isolated.”

The system should produce two complementary outputs:

- A semantic representation of the report as a whole, useful for classification and similarity search
- Structured safety information extracted from the narrative

Extraction targets include activity, hazard, energy source, barrier, barrier failure, exposure, location, equipment, potential consequence, and evidence relating to a Life-Saving Rule.

Where possible, each extracted item should include the words or sentence that support it. If the report does not provide enough information, the system should return an unknown or uncertain value rather than inventing one.

## 7. Safety Ontology and Taxonomy

A safety ontology is a controlled vocabulary that defines concepts and how they relate. It lets the system recognize different phrases as referring to the same safety idea and supports consistent analysis across sites and report types.

A useful starting chain is:

**Activity → Hazard → Energy → Barrier → Barrier failure → Exposure → Potential consequence**

For example:

**Maintenance → unexpected energization → electrical energy → LOTO → verification not performed → worker exposure → potential serious or fatal electrical injury**

The supplied research paper describes a related taxonomy for process-safety reports, including technical, operational, and organizational factors. It also discusses extracting vocabulary, n-grams, and keywords from free text to map reports into a taxonomy. These ideas can inform the design, while OIL's own HSE experts define and approve the vocabulary used in production.

## 8. Labelling and Expert Annotation

The SIF model depends on a clear and consistently applied target label. OIL's HSE and domain experts should first define what qualifies as SIF potential. The label should describe the potential for a serious or fatal outcome, not simply the injury severity that happened to occur.

A practical annotation process is to:

1. Write guidelines with positive, negative, and ambiguous examples.
2. Select a representative sample across report types, sites, and activities.
3. Have multiple experts label a portion of the sample and measure agreement.
4. Use weak supervision or LLM-assisted pre-annotation only to help identify candidates.
5. Have experts review the validation and test sets.
6. Preserve an uncertain or review-needed category for reports that cannot be labelled reliably.

The annotation guidelines and label history should be versioned. Changes to the definition should be reflected in subsequent training and evaluation datasets.

## 9. SIF-Potential Prediction

The primary classifier estimates:

**P(SIF potential | narrative, metadata, extracted safety factors)**

A near miss can receive a high SIF-potential score even if it caused no injury. The model should assess the credible potential described by the report, rather than infer risk from outcome severity alone.

A staged model development path is recommended:

1. Establish a transparent baseline using TF-IDF with Logistic Regression or an SVM.
2. Evaluate a Transformer text classifier for contextual understanding.
3. Add structured safety factors and compare the results.

The final interface should show the score together with the supporting evidence. For example: electrical energy, worker exposure, an isolation barrier, and a verification failure. A score without inspectable evidence is not sufficient for HSE decision support.

## 10. Life-Saving Rule Mapping

A report may relate to more than one Life-Saving Rule, so rule mapping should be treated as a multi-label classification task. The model estimates a score for each rule in the approved catalogue.

An example output might be:

- Energy Isolation: 0.94
- Line of Fire: 0.71
- Hot Work: 0.08

The catalogue must come from the approved IOGP or OIL reference selected for the project. The model should only map reports to those approved rules; it must not create new rules. Each mapping should include the text evidence that supports it and should be available for HSE review.

## 11. Precursor-Pattern Discovery

Supervised models can find known categories, but they cannot identify every recurring situation in advance. Unsupervised analysis can help HSE teams discover patterns that have not yet been formally defined.

One approach is to represent each report with a semantic embedding and group similar reports. HDBSCAN is a reasonable starting point because it can identify dense groups without forcing every report into a cluster; reports that do not fit a clear group can remain unclustered.

For example, reports mentioning a wrong breaker being isolated, LOTO not being verified, an isolation point being misidentified, or unexpected energization may form a recurring pattern. An HSE expert can review the examples and give the pattern a meaningful label such as “energy-isolation verification failure.”

A useful pattern record should include its reviewed label, frequency, SIF-potential distribution, related activities and sites, common barrier failures, and links to member reports. A cluster number on its own is not a useful operational result.

## 12. Barrier-Failure Representation

A barrier is a control intended to prevent an unwanted event or reduce its consequences. The system should represent the control separately from the way it failed.

For example:

- Barrier: LOTO
- Failure mode: verification not performed

Potential failure states include missing, bypassed, incorrectly applied, failed, not verified, and inadequate. OIL HSE experts should confirm the terminology and definitions before these categories are used in production. The system should allow unknown or multiple failure modes when the narrative supports them.

## 13. Probabilistic Safety Analysis

A Bayesian Network can represent dependencies between safety factors as nodes and directed links. A candidate structure is:

**Activity → Hazard → Energy → Barrier → Barrier failure → Exposure → SIF potential**

Once the network is populated with suitable data, it can estimate conditional quantities such as:

**P(SIF potential | activity, hazard, barrier failure, exposure)**

The supplied paper applies Bayesian Networks to process-safety reports and describes dependencies between causes, event types, consequences, and influencing factors. That is a useful methodological reference for this layer.

The network should be treated as a probabilistic representation of observed data. Reports alone do not prove that one factor caused another. Any causal interpretation requires additional evidence and appropriate expert review.

## 14. Why Add a Probabilistic Layer?

A classifier estimates whether a report resembles known SIF-potential examples. A probabilistic model can also show how combinations of structured safety factors relate to model outcomes in the observed data.

For example, HSE teams could examine which barrier failures are frequently associated with high SIF-potential scores for a particular activity. This offers a structured way to explore relationships, while keeping the distinction between observed association and proven causation clear.

## 15. Uncertainty and Human Review

Model scores should be calibrated before users interpret them as probabilities or confidence. Reports with uncertain predictions should be routed to a review queue rather than presented as settled conclusions.

A typical workflow is:

1. Highlight high-confidence candidates for timely attention.
2. Send uncertain or ambiguous cases to HSE review.
3. Store reviewer corrections alongside the original model output.
4. Consider reviewed examples as candidates for future model updates.

Methods to consider include temperature scaling, ensemble uncertainty, selective prediction, and conformal prediction. The chosen method should be evaluated against actual review capacity and the cost of missed SIF precursors.

## 16. Site and Activity SIF-Precursor Density

Raw report totals can be misleading because larger sites or busier activities may produce more reports. A normalized measure helps compare them more fairly.

For a site, define expected SIF-precursor density as:

$$
D(\text{site}) = \frac{\sum_i P(\text{SIF}_i)}{N_{\text{site}}}
$$

Here, the numerator is the sum of SIF-potential probabilities for reports from that site, and the denominator is the number of reports in the selected period. The same calculation can be applied to activities.

The dashboard should show the denominator and reporting period with the density measure. Low report volume can make comparisons unstable, so users should be able to see the underlying counts and drill down to the reports.

## 17. Temporal Analytics

The system should track changes over time in SIF-potential density, Life-Saving Rule distribution, precursor frequency, and barrier-failure patterns. Results can be grouped by month or quarter and filtered by site, activity, or report type.

The supplied paper describes periodically updating its Bayesian model as new process-safety reports arrive and using the updated model to examine trends over time. OIL can use a similar update cycle, with governance checks before a new model version is released.

A later extension can monitor for distribution drift in report language, activities, and sites. A detected shift should prompt investigation; it should not trigger an automatic model update without review.

## 18. HSE Dashboard

The dashboard should support the questions HSE teams ask during routine review and deeper investigation.

**Overview**

Show total reports, SIF-potential rate, site and activity density, common Life-Saving Rules, and leading precursor patterns.

**Report Explorer**

Show the original narrative, SIF-potential score, extracted factors, mapped rules, supporting evidence, uncertainty, and review status. Let users open the evidence in context and correct extracted information where their role permits.

**Precursor Intelligence**

Show the reviewed pattern label, number of reports, associated SIF scores, sites, activities, and barrier failures. Provide access to the reports behind the pattern.

**Site and Activity Analysis**

Compare normalized density and time trends, with report volume visible alongside each measure. Allow drill-down from a site or activity to the contributing reports.

**Life-Saving Rules**

Show rule distribution and its links to hazards, activities, and precursor patterns, with report-level evidence available.

## 19. Example Report Journey

**Report text:** “During maintenance, the technician entered the area while the electrical panel was still energized. LOTO was not verified.”

The system might extract:

- Activity: Maintenance
- Energy: Electrical
- Hazard: Unexpected energization
- Barrier: LOTO
- Barrier failure: Verification not performed
- Exposure: Worker

From there, it can estimate SIF potential, suggest Energy Isolation as a relevant Life-Saving Rule, identify a related precursor pattern, and include the report in site and activity analytics. The HSE user should be able to inspect the exact phrases supporting each result and correct errors through the review workflow.

## 20. Model Evaluation

Evaluation should cover predictive performance and whether the results are useful for HSE review.

For SIF classification, measure precision, recall, F1, AUROC, and AUPRC. Because SIF-potential reports may be uncommon, AUPRC and recall deserve particular attention alongside the other measures.

For Life-Saving Rule mapping, report per-rule precision, recall, and F1, as well as suitable multi-label metrics.

For information extraction, measure entity-level precision, recall, and F1 for activity, hazard, energy, barrier, barrier failure, and exposure.

For precursor discovery, combine clustering diagnostics with expert assessment of coherence and operational usefulness.

For triage, use Recall@K: how many confirmed SIF precursors are recovered when HSE reviewers examine only the top K% of reports? Also measure calibration, uncertainty quality, and the resulting review workload.

## 21. Experimental Design

Do not rely on a single random train/test split. Use several evaluation settings to understand what the system learns and where it generalizes.

| Experiment | Purpose |
| --- | --- |
| Classical baseline | Establish a transparent reference point. |
| Transformer model | Test whether contextual language understanding improves results. |
| Structured safety factors | Measure the value added by extracted factors. |
| Probabilistic layer | Test whether it improves interpretation or calibration. |
| Temporal holdout | Test performance on reports created later than the training data. |
| Site holdout | Test generalization to a site not represented in training, where feasible. |
| Ablation | Identify which components contribute useful value. |
| Calibration | Check whether predicted probabilities match observed outcomes. |

## 22. Ablation Study

Compare progressively richer system variants:

- A: Narrative text only
- B: Text plus metadata
- C: Text plus extracted safety factors
- D: Text, safety factors, and probabilistic analysis

This helps determine which components improve prediction and which mainly improve interpretation. A component should not be retained merely because it makes the architecture more elaborate; its contribution should be demonstrated or its operational value stated clearly.

## 23. Distribution-Shift Testing

Use a temporal holdout by training on earlier reports and testing on later ones. Where data permits, also hold out one site and evaluate the model on that site's reports.

These tests help show whether the model has learned safety concepts that transfer across time and locations, rather than memorizing local wording or site-specific patterns. Performance should be reviewed by report type, activity, and site where sample sizes allow.

## 24. Technology Options

A possible technology stack includes:

- **ML and NLP:** Python, PyTorch, Hugging Face Transformers, and scikit-learn
- **Pattern discovery:** sentence-transformers and HDBSCAN
- **Probabilistic analysis:** pgmpy or an equivalent Bayesian Network library
- **Backend:** FastAPI
- **Data storage:** PostgreSQL, with pgvector if vector search is required
- **Dashboard:** Streamlit for an initial prototype, or a dedicated web frontend for a production application

These are candidate technologies, not fixed requirements. Final choices should reflect OIL's approved infrastructure, security requirements, operational support, and integration needs.

## 25. Deployment Flow and Traceability

A typical deployment flow is:

**HSSE export → ingestion → preprocessing → NLP inference → safety-factor extraction → predictions and analytics → database and optional vector index → dashboard**

For each generated output, store the model version, creation time, extracted factors, probability or score, supporting evidence, and review status. Retain the original report and enough processing metadata to reproduce or investigate a result.

Model updates should follow a controlled release process. Keep prior outputs available so reviewers can distinguish results from different model versions.

## 26. Security and Governance

Safety reports may contain sensitive operational information. The system should use approved processing environments, role-based access controls, audit logging, and controlled model versions.

External AI services should receive report data only when explicitly permitted by the applicable data-security requirements. Access to narratives and reviewer details should follow OIL policy. Retention, anonymization, and deletion rules should be established before production deployment.

## 27. Failure Modes and Controls

| Potential issue | Control |
| --- | --- |
| Ambiguous narrative | Mark the result as uncertain and route it for HSE review. |
| Missing context | Return unknown rather than inventing information. |
| Rare or local terminology | Maintain an approved domain vocabulary and use contextual models. |
| Class imbalance | Use suitable training methods and evaluation measures. |
| Distribution shift | Test across time and sites; monitor drift. |
| Incorrect extraction | Show evidence and allow HSE correction. |
| Causal overinterpretation | Describe observational relationships accurately and avoid unsupported causal claims. |
| Model or taxonomy change | Version changes and retain the history of outputs and reviews. |

## 28. Development Roadmap

1. Define the data schema and quality pipeline.
2. Agree SIF and Life-Saving Rule annotation guidelines and build a reviewed labelled dataset.
3. Train classical NLP baselines.
4. Evaluate Transformer models.
5. Add safety-factor extraction and evidence display.
6. Explore precursor clustering with HSE review.
7. Develop the probabilistic analysis layer.
8. Add uncertainty handling and the review workflow.
9. Build site, activity, and temporal analytics in the dashboard.
10. Complete expert validation, temporal and site testing, ablations, and operational readiness checks.

The phases can overlap where dependencies allow, but the labelled data, definitions, and traceability requirements should be established early.

## 29. Evaluation Matrix

| System component | Primary measures |
| --- | --- |
| SIF classification | Precision, recall, F1, AUROC, AUPRC |
| Life-Saving Rules | Per-rule precision, recall, and F1; multi-label metrics |
| Safety information extraction | Entity-level precision, recall, and F1 |
| Precursor discovery | Clustering diagnostics and expert validation |
| Operational triage | Recall@K and review workload |
| Uncertainty | Calibration and reliability analysis |

## 30. Final System Blueprint

```text
OIL SAFETY REPORTS
        ↓
DATA QUALITY AND DOMAIN NORMALIZATION
        ↓
NLP AND SAFETY INFORMATION EXTRACTION
        ↓
SAFETY ONTOLOGY AND TAXONOMY
        ↓
SIF-POTENTIAL PREDICTION  |  LIFE-SAVING RULE MAPPING  |  PRECURSOR DISCOVERY
        ↓
STRUCTURED BARRIER REPRESENTATION
        ↓
PROBABILISTIC SAFETY ANALYSIS
        ↓
UNCERTAINTY HANDLING AND HSE REVIEW
        ↓
SITE, ACTIVITY, AND TEMPORAL ANALYTICS
        ↓
INTERACTIVE HSE DASHBOARD
```

## 31. Methodological Basis

The supplied research paper argues that process-safety narratives contain useful information about causes, influencing factors, and consequences. It presents an approach combining NLP, a taxonomy, and Bayesian Networks, including periodic updates as new reports become available.

For this design, the relevant ideas are the structured taxonomy, explicit representation of barriers, extraction of information from free text, and probabilistic analysis of relationships. The paper provides methodological context; OIL-specific definitions, data, and expert validation remain necessary before applying these methods operationally.

## 32. Intended Outcome

The finished system should help an HSE user answer, from one interface:

- Which reports describe potential for a serious or fatal outcome?
- Which Life-Saving Rules may be involved?
- What precursor patterns are recurring?
- Which barrier failures appear repeatedly?
- Which sites and activities have higher SIF-precursor density?
- How uncertain are the model's results?
- Which report evidence supports each result?

The system is designed to make safety analysis more consistent and scalable while keeping the original report visible, the AI's reasoning inspectable, and expert review central to the process.
