import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from .evaluator import Evaluator, EvaluatorResult


class CallbackPlotter:
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator

        # Storage for history
        columns = list(EvaluatorResult.__annotations__.keys())
        self.history = pd.DataFrame(columns=columns)

        # Initialize plot
        plt.ion()
        self.fig, self.axes = plt.subplots(6, 1, figsize=(10, 10), sharex=True)

    def callback(self, xk, state=None):
        result = self.evaluator.evaluate(xk)
        self.history = pd.concat([self.history, pd.DataFrame([result.__dict__])], ignore_index=True)

        # Store history
        for i, column in enumerate(["E", "B_mag", "grad_val", "trap_depth", "x", "y", "z"]):
            values = self.history[column].values.tolist()
            ax = self.axes[i]
            ax.clear()
            ax.plot(values, label=column)
            ax.set_ylabel(column)
            ax.set_xlabel("Iteration")

        clear_output(wait=True)
        plt.draw()
        plt.tight_layout()
        plt.pause(0.05)
