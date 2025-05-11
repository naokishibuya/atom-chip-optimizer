import os
import threading
import queue
import signal
import sys
import uvicorn
from fastapi import FastAPI, Request
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
import atom_chip as ac


RUN_DIR = os.path.dirname(os.path.abspath(__file__))


# fmt: off
options = ac.potential.AnalysisOptions(
    search = dict(
        x0      = [0.0, 0.0, 0.5],  # Initial guess
        bounds  = [(-0.5, 0.5), (-0.5, 0.5), (0.0, 2.0)],
        method  = "Nelder-Mead",
        options = dict(
            xatol   = 1e-10,
            fatol   = 1e-10,
            maxiter = int(1e5),
            maxfev  = int(1e5),
            disp    = True,
        ),
    ),
    hessian = dict(
        method = "jax",
        # method = "finite-difference",
        # hessian_step = 1e-5,  # Step size for Hessian calculation
    ),
    # for the trap analayis (not used for field analysis)
    total_atoms=1e5,
    condensed_atoms=1e5,
)
# fmt: on

# Build the atom chip
# fmt: off
prototype = ac.AtomChip(
    name        = "Atom Chip Analyzer",
    atom        = ac.rb87,
    components  = [],
    bias_fields = ac.field.ZERO_BIAS_FIELD,
)
# fmt: on


# Loading visualizer configuration
visualizer = ac.visualization.Visualizer(os.path.join(RUN_DIR, "visualization.yaml"))

layout_queue = queue.Queue()


def process_job():
    if layout_queue.empty():
        return
    print("Processing simulation job...")
    layout = layout_queue.get()
    try:
        atom_chip = prototype.from_json(layout)
        analysis = atom_chip.analyze(options)
        visualizer.update(atom_chip, analysis)
    except Exception as e:
        print(f"Error processing simulation job: {e}")


app = FastAPI()


@app.post("/simulate")
async def simulate(request: Request):
    print("Received simulation request:")
    try:
        layout = await request.json()
        layout_queue.queue.clear()
        layout_queue.put(layout)
    except Exception as e:
        print(f"Error processing request: {e}")
        return {"status": "error", "message": str(e)}
    return {"status": "received"}


def server_thread():
    uvicorn.run(app, host="127.0.0.1", port=8000)


def signal_handler(signum, frame):
    print(f"Signal {signum} received, shutting down...")
    QApplication.quit()


signal.signal(signal.SIGINT, signal_handler)


def main():
    # Start FastAPI server in a separate thread
    threading.Thread(target=server_thread, daemon=True).start()

    # Qt application setup
    app = QApplication(sys.argv)

    # Create a timer to periodically check for new jobs
    timer = QTimer()
    timer.timeout.connect(process_job)
    timer.start(100)  # Check every 100 ms

    # Start the event loop
    app.exec_()


if __name__ == "__main__":
    main()
