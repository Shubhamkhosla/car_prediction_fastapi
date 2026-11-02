
import subprocess
import sys
import time

# Start FastAPI
backend = subprocess.Popen([
    sys.executable,
    "-m",
    "uvicorn",
    "app.main:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8000"
])

# Give FastAPI time to start
time.sleep(3)

# Start Streamlit
frontend = subprocess.Popen([
    sys.executable,
    "-m",
    "streamlit",
    "run",
    "streamlit_app.py",
    "--server.address",
    "127.0.0.1",
    "--server.port",
    "8501"
])

try:
    frontend.wait()
except KeyboardInterrupt:
    print("\nStopping both applications...")

finally:
    frontend.terminate()
    backend.terminate()

    frontend.wait()
    backend.wait()

    print("Both applications stopped.")
