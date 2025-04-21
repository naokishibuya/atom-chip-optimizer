import logging
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from .evaluator import Evaluator


class CallbackPlotter:
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator

        # For result history
        self.history_df = None

        # Initialize plot
        plt.ion()
        self.fig, self.axes = plt.subplots(7, 1, figsize=(10, 10), sharex=True)

    def callback(self, xk, state=None):
        result = self.evaluator.evaluate(xk)
        logging.info(result)
        result_df = pd.DataFrame([result.__dict__])
        if self.history_df is None:
            self.history_df = result_df
        else:
            self.history_df = pd.concat([self.history_df, result_df], ignore_index=True)

        # Store history
        for i, column in enumerate(["E", "B_mag", "grad_val", "trap_depth", "x", "y", "z"]):
            values = self.history_df[column].values.tolist()
            ax = self.axes[i]
            ax.clear()
            ax.plot(values, label=column)
            ax.set_ylabel(column)
            ax.set_xlabel("Iteration")

        clear_output(wait=True)
        plt.draw()
        plt.tight_layout()
        plt.pause(0.05)
