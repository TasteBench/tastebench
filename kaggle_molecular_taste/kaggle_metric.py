"""Kaggle scoring metric — macro-averaged F1 for Molecular Taste Classification.

Paste this entire file into a Kaggle metric notebook (Create > Notebook,
mark it as a metric), publish it, and attach the notebook as the
competition's scoring metric.

The metric is equivalent to:

    sklearn.metrics.f1_score(y_true, y_pred, average='macro',
                             labels=['sweet','bitter','sour','umami','undefined'])

Macro averaging weights every class equally regardless of support, which is
the right choice for this dataset's heavy imbalance (umami n=6 vs. sweet
n=1473 in test). Plain accuracy would be dominated by sweet; macro-F1 forces
models to do well on minority classes too.
"""

import pandas as pd
from sklearn.metrics import f1_score

VALID_LABELS = ("sweet", "bitter", "sour", "umami", "undefined")


class ParticipantVisibleError(Exception):
    """Raised for errors that should be shown to participants in submission feedback."""


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
) -> float:
    """Macro-averaged F1 across the 5 taste classes.

    Parameters
    ----------
    solution
        Ground-truth dataframe with columns [row_id_column_name, "taste"].
    submission
        Participant submission with columns [row_id_column_name, "taste"].
    row_id_column_name
        Name of the id column (this competition uses "id").

    Returns
    -------
    float
        Macro-F1 score in [0, 1].
    """
    if "taste" not in submission.columns:
        raise ParticipantVisibleError("Submission must have a 'taste' column.")

    invalid = set(submission["taste"]) - set(VALID_LABELS)
    if invalid:
        raise ParticipantVisibleError(
            f"Invalid taste labels in submission: {sorted(invalid)}. "
            f"Allowed labels: {list(VALID_LABELS)} (case-sensitive)."
        )

    merged = solution.merge(
        submission, on=row_id_column_name, suffixes=("_true", "_pred")
    )
    if len(merged) != len(solution):
        raise ParticipantVisibleError(
            f"Submission is missing predictions for some test ids. "
            f"Expected {len(solution)} matched rows, got {len(merged)}."
        )

    return f1_score(
        merged["taste_true"],
        merged["taste_pred"],
        average="macro",
        labels=list(VALID_LABELS),
    )
