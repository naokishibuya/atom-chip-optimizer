import os
import threading
import queue
import uvicorn
from fastapi import FastAPI, Request
import atom_chip as ac


RUN_DIR = os.path.dirname(os.path.abspath(__file__))


# fmt: off
options = ac.potential.AnalysisOptions(
    search = dict(
        x0      = [0.0, 0.0, 0.5],  # Initial guess
        bounds  = [(-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0)],
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
    condensed_atoms=5e4,
    verbose = True,
)
# fmt: on

# TODO
bias_fields = ac.field.BiasFields(
    coil_factors=[-1.068, 1.8, 3.0],  # [G/A]
    currents=[17.4, 44.3, 0.0],  # [A]
    stray_fields=[3.5, -0.1, 0.0],  # [G]
)

bias_fields = ac.field.BiasFields(
    coil_factors=[-1.068, 1.8, 3.0],  # [G/A]
    currents=[0.0, 0.0, 0.0],  # [A]
    stray_fields=[1.0, 1.0, 0.0],  # [G]
)

# Build the atom chip
# fmt: off
atom_chip = ac.AtomChip(
    name        = "Atom Chip Analyzer",
    atom        = ac.rb87,
    components  = [],
    bias_fields = bias_fields,
)
# fmt: on


# Loading visualizer configuration
visualizer = ac.visualization.Visualizer(os.path.join(RUN_DIR, "atom_chip_server.yaml"))

layout_queue = queue.Queue()
condition = threading.Condition()


def main_loop():
    while visualizer.is_alive:
        with condition:
            condition.wait(timeout=1)
        try:
            layout = layout_queue.get_nowait()
        except queue.Empty:
            continue  # check again

        print("Processing simulation job...")
        try:
            atom_chip.process_json(layout)
            atom_chip.analyze(options)
            visualizer.update(atom_chip)
        except Exception as e:
            print(f"Error processing simulation job: {e}")
            continue


app = FastAPI()


@app.post("/simulate")
async def simulate(request: Request):
    print("Received simulation request:")
    try:
        layout = await request.json()
        with condition:
            layout_queue.queue.clear()
            layout_queue.put(layout)
            condition.notify()
    except Exception as e:
        print(f"Error processing request: {e}")
        return {"status": "error", "message": str(e)}
    return {"status": "received"}


def server_thread():
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    threading.Thread(target=server_thread, daemon=True).start()
    main_loop()
