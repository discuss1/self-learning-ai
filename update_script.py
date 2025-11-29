import subprocess
import os

def update_docs():
    subprocess.run(["python", "app/scraper/python.py"], cwd="/app")
    subprocess.run(["python", "app/scraper/javascript.py"], cwd="/app")

def fine_tune_models():
    subprocess.run(["python", "-c", "from app.main import fine_tune; fine_tune('python')"], cwd="/app")
    subprocess.run(["python", "-c", "from app.main import fine_tune; fine_tune('javascript')"], cwd="/app")

if __name__ == "__main__":
    update_docs()
    fine_tune_models()

