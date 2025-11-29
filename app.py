# app.py
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from github import Github
import os
import subprocess
from pathlib import Path

# --- Load Model ---
model_name = "facebook/incoder-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
codegen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# --- Generate Code (Multi-Language) ---
def generate_code(prompt: str, language: str = "python") -> str:
    if language == "django":
        prompt = (
            "You are a Django expert. Generate production-ready Django code. "
            "Follow Django 4.2 best practices and official documentation. "
            f"Task: {prompt}"
        )
    else:
        prompt = f"Generate {language} code for: {prompt}"

    result = codegen_pipeline(prompt, max_length=500, num_return_sequences=1)
    return result[0]["generated_text"]

# --- Scrape Docs ---
def scrape_docs(language: str):
    if language == "django":
        subprocess.run(["python", "app/scraper/django.py"])
    elif language == "python":
        subprocess.run(["python", "app/scraper/python.py"])
    return f"✅ Scraped {language} documentation!"

# --- Fine-Tune Model ---
def fine_tune(language: str):
    # Placeholder: Replace with actual fine-tuning logic
    return f"🔄 Fine-tuning on {language} docs (simulated)."

# --- Commit to GitHub ---
def commit_to_github(filename: str, code: str):
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo("your-username/self-learning-ai")
    try:
        repo.create_file(
            f"generated/{filename}",
            f"AI-generated {filename}",
            code
        )
        return f"✅ Committed {filename} to GitHub!"
    except Exception as e:
        return f"❌ Error: {e}"

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Self-Learning AI (Multi-Language + Django)")

    with gr.Tabs():
        # --- Code Generation Tab ---
        with gr.TabItem("Generate Code"):
            with gr.Row():
                language = gr.Dropdown(
                    ["python", "django", "javascript", "c++"],
                    value="python",
                    label="Language"
                )
                prompt = gr.Textbox(
                    label="Prompt (e.g., 'Write a function to...')",
                    placeholder="Write a Django view to list blog posts..."
                )
            with gr.Row():
                generate_btn = gr.Button("Generate Code")
                output = gr.Code(label="Generated Code")
            with gr.Row():
                filename = gr.Textbox(label="Filename (e.g., views.py)")
                commit_btn = gr.Button("Commit to GitHub")
                commit_status = gr.Textbox(label="GitHub Status")

            generate_btn.click(
                fn=generate_code,
                inputs=[prompt, language],
                outputs=output
            )
            commit_btn.click(
                fn=commit_to_github,
                inputs=[filename, output],
                outputs=commit_status
            )

        # --- Training Tab ---
        with gr.TabItem("Train Model"):
            with gr.Row():
                train_language = gr.Dropdown(
                    ["python", "django", "javascript"],
                    value="python",
                    label="Language to Train"
                )
                scrape_btn = gr.Button("Scrape Documentation")
                fine_tune_btn = gr.Button("Fine-Tune Model")
            with gr.Row():
                scrape_status = gr.Textbox(label="Scrape Status")
                fine_tune_status = gr.Textbox(label="Fine-Tune Status")

            scrape_btn.click(
                fn=scrape_docs,
                inputs=train_language,
                outputs=scrape_status
            )
            fine_tune_btn.click(
                fn=fine_tune,
                inputs=train_language,
                outputs=fine_tune_status
            )

# --- Deploy ---
demo.launch()

