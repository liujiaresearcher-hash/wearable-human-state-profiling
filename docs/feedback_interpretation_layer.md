# Feedback Interpretation Layer

## Why Interpretation Matters

Classification metrics are useful for checking whether wearable features can distinguish protocol labels, but they do not explain how a person-facing system should communicate uncertainty, ambiguity, or individual differences. For human-computer interaction research, the output should be understandable enough to support reflection without overstating what the wearable signals mean.

This repository adds an HCI interpretation layer on top of the raw WESAD Empatica E4 baseline/stress pipeline. The layer keeps the original modeling task, then adds local summaries that describe feature-group patterns, subject-level baseline/stress distinctions, ambiguous model output, and feedback-card examples.

The wording is intentionally cautious. The outputs refer to patterns under the WESAD protocol, psychophysiological strain indicators, and the baseline/stress distinction. They are not clinical conclusions, diagnosis tools, medical advice, or deployable stress-monitoring claims.

## From Classification To Subject Profile

The original demo trains baseline classifiers for baseline versus stress windows. The interpretation layer adds subject-level profiles generated from the same local window-level feature table.

Each profile summarizes:

- Number of baseline and stress windows.
- Standardized ACC / movement-related feature means.
- Standardized BVP / cardiovascular-related feature means.
- Standardized EDA / electrodermal-related feature means.
- Standardized TEMP / skin-temperature-related feature means.
- Stress-minus-baseline descriptive differences for each feature group.
- Top changed features within the subject.
- A cautious interpretation sentence.

The profile is descriptive. It does not say that a subject is in a real-world stress state. It says that, under the WESAD protocol, selected wearable feature groups changed between baseline and protocol stress windows. Those changes may reflect psychophysiological strain indicators within the experimental task.

## From Feature Contribution To User-Facing Explanation

The explanation layer separates model-based and descriptive components:

- Model-based: logistic-regression coefficients are summarized as model-associated feature weights after imputation and standard scaling.
- Descriptive: within-subject stress-minus-baseline feature differences are summarized as local descriptive changes.
- Ambiguity: logistic-regression probabilities near 0.5 are flagged as ambiguous model output.

This separation matters because a model-associated feature weight is not causal evidence, and a descriptive difference is not proof that a feature caused a state. Both are useful for making the pipeline more inspectable, but neither should be treated as clinical, causal, or medical advice.

## Feedback Styles

### Direct Alert

A direct alert is the most explicit style. It might say that body-signal patterns suggest a higher-strain moment under the research protocol. This style is easy to notice, but it can feel intrusive or overly certain if used without uncertainty language.

### Reflective Self-Check Prompt

A reflective prompt invites the user to interpret the moment themselves, for example by asking whether they want to pause and check how they feel. This style may better preserve agency because the system does not present the model output as a final judgment.

### Low-Interruption Visual Cue

A low-interruption cue uses a subtle interface change, such as color, rhythm, or icon state. This style may reduce interruption cost, but it requires careful design so users understand what the cue means and can ignore it when it is not useful.

## Limitations

The current layer is an offline research-demo extension. It uses WESAD protocol labels and local derived features. It does not validate real-world stress monitoring, does not evaluate live interaction, does not provide medical advice, and does not establish that users trust or benefit from these feedback styles.

The feature summaries are intentionally lightweight. They use simple statistical features, feature groups, logistic-regression weights, descriptive differences, and probability bands. More advanced signal processing may be useful later, but the current version prioritizes transparency and reproducibility.

Generated profile tables, ambiguity reports, and feedback cards are derived from local WESAD subject data and should remain under `outputs/`.

## Future User Study Ideas

Future HCI work could evaluate how people understand and respond to these feedback styles. Possible study dimensions include:

- Clarity: whether users understand what the feedback is saying.
- Trust: whether users see the feedback as appropriately uncertain and bounded.
- Perceived usefulness: whether the feedback helps reflection or task management.
- Comfort: whether the feedback feels supportive rather than intrusive.
- Agency: whether users feel able to accept, ignore, or reinterpret the cue.
- Actionability: whether the feedback suggests a useful next step without overclaiming.

Possible methods include interviews, diary studies, or mixed-method lab sessions where participants compare direct alerts, reflective self-check prompts, and low-interruption visual cues. This repository does not evaluate those questions yet; it only prepares reproducible examples that could support such a future study.
