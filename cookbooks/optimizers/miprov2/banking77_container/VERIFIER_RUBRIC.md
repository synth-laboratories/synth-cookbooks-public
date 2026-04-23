# Banking77 Verifier Rubric

Score each rollout on a `0.0` to `1.0` scale.

Return strict JSON with:

- `score`
- `summary`
- `criteria`
- `notes`

## What counts as success

The rollout is successful when:

1. the container returns a valid rollout response
2. exactly one predicted intent is present
3. the predicted intent is a valid Banking77 label
4. the predicted intent exactly matches the expected intent

## Core failure rules

Assign `0.0` if any of these are true:

- the rollout response is malformed
- the prediction is missing
- the prediction is not one Banking77 label
- the prediction is outside the Banking77 taxonomy

Assign at most `0.4` if:

- the response is valid but the predicted label is incorrect

## Criteria and weights

1. `prediction_present` weight `0.20`
2. `prediction_in_taxonomy` weight `0.20`
3. `prediction_exact_match` weight `0.60`
