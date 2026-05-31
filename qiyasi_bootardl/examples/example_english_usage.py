# -*- coding: utf-8 -*-
"""English usage example for qiyasi_bootardl."""
from qiyasi_bootardl import bootstrap_ardl_test
from qiyasi_bootardl.datasets import load_macro_example


def main():
    data = load_macro_example()

    result = bootstrap_ardl_test(
        data,
        yvar="الناتج",
        xvar=["الاستثمار", "الانفتاح"],
        case=3,
        max_lag=4,
        ardl_ic="AIC",
        n_boot=500,
        random_state=2024,
        progress=True,
    )

    print(result.summary())
    print("\n=== Statistics ===")
    print(result.statistics_table())
    print("\n=== Bootstrap critical values ===")
    print(result.bootstrap_table())
    print("\n=== Decision ===")
    print(result.decision_label())
    for w in result.warnings():
        print(w)


if __name__ == "__main__":
    main()
