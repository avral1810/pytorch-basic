from __future__ import annotations


def get_visible_tests():
    return [
        ("returns a float accuracy", test_returns_float),
        ("learns the classification task", test_good_accuracy),
    ]


def get_hidden_tests():
    return [
        ("is deterministic for the same seed", test_deterministic),
        ("accuracy stays in range", test_accuracy_range),
    ]


def test_returns_float(solution_module):
    actual = solution_module.train_full_logistic_regression(42)
    assert isinstance(actual, float), f"Expected a float, got {type(actual)}"


def test_good_accuracy(solution_module):
    actual = solution_module.train_full_logistic_regression(42)
    assert actual >= 0.9, f"Expected logistic regression accuracy >= 0.9, got {actual:.3f}"


def test_deterministic(solution_module):
    first = solution_module.train_full_logistic_regression(42)
    second = solution_module.train_full_logistic_regression(42)
    assert abs(first - second) < 1e-8, f"Expected deterministic accuracy, got {first} and {second}"


def test_accuracy_range(solution_module):
    actual = solution_module.train_full_logistic_regression(7)
    assert 0.0 <= actual <= 1.0, f"Expected accuracy in [0, 1], got {actual}"
