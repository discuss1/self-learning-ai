from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from github import Github
import os
import subprocess
from pathlib import Path

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load Model
model_name = "facebook/incoder-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
codegen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Generate Code
def generate_code(prompt: str, language: str = "python") -> str:
    model_path = f"./models/{language}_fine-tuned"
    if os.path.exists(model_path):
        pipe = pipeline("text-generation", model=model_path, tokenizer=tokenizer, device=0)
    else:
        pipe = codegen_pipeline
    result = pipe(prompt, max_length=500, num_return_sequences=1)
    return result[0]["generated_text"]

# GitHub Commit
def commit_to_github(filename: str, code: str):
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo("your-username/self-learning-ai")
    try:
        repo.create_file(
            f"generated/{filename}",
            f"AI-generated {filename} (official docs)",
            code
        )
        return f"✅ Committed {filename} to GitHub!"
    except Exception as e:
        return f"❌ Error: {e}"

# Self-Update
def self_update():
    subprocess.run(["git", "pull"], cwd="/app")
    subprocess.run(["python", "update_script.py"], cwd="/app")

# API Endpoints
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
def generate(request: Request, prompt: str = Form(...), language: str = Form(...)):
    return {"code": generate_code(prompt, language)}

@app.post("/commit")
def commit(request: Request, code: str = Form(...), filename: str = Form(...)):
    return {"status": commit_to_github(filename, code)}

@app.post("/update")
def update(background_tasks: BackgroundTasks):
    background_tasks.add_task(self_update)
    return {"status": "Self-update triggered!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

