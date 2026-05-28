import os
import sys
import webbrowser
import time
from threading import Timer

# Add parent directory to path to ensure backend imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

try:
    from backend.app import app
except ImportError as e:
    print(f"Import Error: {e}")
    print("Attempting to run direct script setup...")
    import subprocess
    # Run setup if packages are missing
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    from backend.app import app

def open_browser():
    """Waits for server to start, then automatically opens the web browser."""
    try:
        time.sleep(1.5)
        url = "http://localhost:8000"
        print(f"\n[SYSTEM] Automatically launching browser to: {url}")
        webbrowser.open(url)
    except Exception as e:
        print(f"[WARNING] Could not open browser automatically: {e}")

if __name__ == '__main__':
    # Colorful Banner
    print("\n" + "="*70)
    print("  \033[1;36mCrediTrust XAI - AI-Based Loan Default Risk Prediction\033[0m")
    print("  \033[1;35mExplainable Credit Decision Support System\033[0m")
    print("  " + "-"*66)
    print("  Institution:  Kyungdong University Global, South Korea")
    print("  Department:   Department of Artificial Intelligence")
    print("  Authors:      Shardul Narvekar & Krishna Sevak")
    print("  " + "-"*66)
    print("  \033[1;32mSTATUS: ML Core Engine Activated.\033[0m")
    print("  \033[1;32mActive Models: Logistic Regression | Random Forest | XGBoost\033[0m")
    print("="*70 + "\n")
    
    # Spawn browser trigger in the background
    Timer(1.5, open_browser).start()
    
    # Run Flask Web Server
    try:
        app.run(host='0.0.0.0', port=8000, debug=False)
    except KeyboardInterrupt:
        print("\n[SYSTEM] Server stopped by user request. Exiting.")
