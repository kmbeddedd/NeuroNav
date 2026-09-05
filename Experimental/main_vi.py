#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys
import time
import webbrowser

# ============================================================
# PROJECT PATHS
# ============================================================
# Resolves to the /home/sage/NeuroNav/ directory
ROOT = Path(__file__).resolve().parent

# Target the scripts and assets nested inside the 'visualization' folder
COMPARE_SCRIPT = ROOT / "visualization" / "scripts" / "compare_predictions.py"
FRONTEND_DIR = ROOT / "visualization" / "frontend"
GENERATED_DIR = ROOT / "visualization" / "generated"

# ============================================================
# SERVER CONFIGURATION
# ============================================================
HOST = "127.0.0.1"
PORT = 8000

# The URL maps directly to the structure hosted from the ROOT directory
URL = f"http://{HOST}:{PORT}/visualization/frontend/index.html"


# ============================================================
# RUN PREDICTION COMPARISON
# ============================================================
def update_comparisons():
    print("=" * 60)
    print("NEURONAV — UPDATING PREDICTION COMPARISONS")
    print("=" * 60)

    if not COMPARE_SCRIPT.exists():
        print(
            f"ERROR: Comparison script not found at expected path:\n"
            f"{COMPARE_SCRIPT}\n\n"
            f"Please ensure 'main_vi.py' is placed directly inside '/home/sage/NeuroNav/'."
        )
        sys.exit(1)

    # Execute the comparison script inside the context of the ROOT path
    result = subprocess.run(
        [sys.executable, str(COMPARE_SCRIPT)],
        cwd=ROOT
    )

    if result.returncode != 0:
        print("\nERROR: Prediction comparison telemetry script failed.")
        sys.exit(result.returncode)

    print("\nPrediction telemetry compilation completed successfully.")


# ============================================================
# START FRONTEND SERVER
# ============================================================
def start_server():
    print("=" * 60)
    print("NEURONAV — STARTING VISUALIZATION SERVER")
    print("=" * 60)

    if not FRONTEND_DIR.exists():
        print(
            f"ERROR: Frontend UI directory not found:\n"
            f"{FRONTEND_DIR}"
        )
        sys.exit(1)

    # Launching the server from ROOT allows the web browser to access
    # both the frontend code and the generated output folder seamlessly.
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            str(PORT),
            "--bind",
            HOST
        ],
        cwd=ROOT
    )

    return server


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================
def main():
    # Proactively create the destination data folder if it doesn't exist yet
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Compile the telemetry metrics from your data files
    update_comparisons()

    # 2. Spin up the background web server
    server = start_server()

    print()
    print("=" * 60)
    print("NEURONAV — SUBSYSTEMS ONLINE")
    print("=" * 60)
    print(f"Interactive Dashboard: {URL}")
    print("Press Ctrl+C at any time to terminate the environment.")
    print("=" * 60)

    # Provide the HTTP socket server a brief moment to initialize and bind
    time.sleep(1)

    # 3. Fire up the system's default browser directly to the dashboard
    webbrowser.open(URL)

    try:
        # Maintain process lifeline while the dashboard is running
        server.wait()
    except KeyboardInterrupt:
        print("\nStopping NeuroNav application ecosystem...")
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
