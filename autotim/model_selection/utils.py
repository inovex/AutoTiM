"""Utils for evaluation and model selection."""


def is_better(metric: str, first_value: float, second_value: float):
    """
    Defines if first_value is better than second_value for the given metric.
    """
    if metric in ['accuracy', 'balanced_accuracy', 'recall_score', 'precision_score']:
        return first_value > second_value

    # else (for metrics that are better when they are minimal)
    return first_value < second_value
