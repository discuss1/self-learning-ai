import gradio as gr
import torch
import logging
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from github import Github
import os
import subprocess
from tqdm import tqdm
from pathlib import Path

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"./logs/training_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

# --- Generate Code ---
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

# --- Fine-Tune Model (with Progress Tracking) ---
def fine_tune(language: str):
    try:
        # Load dataset
        dataset = load_dataset("text", data_files=f"./data/{language}/*.md")["train"]

        # Tokenize
        def tokenize(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

        dataset = dataset.map(tokenize, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./models/{language}_fine-tuned",
            per_device_train_batch_size=2,
            num_train_epochs=1,
            save_steps=500,
            logging_dir="./logs",
            logging_steps=10,
            report_to="tensorboard",
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        # Train with progress
        logger.info(f"🔄 Starting fine-tuning for {language}...")
        trainer.train()
        logger.info(f"💾 Model saved to ./models/{language}_fine-tuned")

        # Get final loss
        final_loss = trainer.state.log_history[-1]["loss"]
        return f"✅ Fine-tuned {language} model! Final loss: {final_loss:.4f}"

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return f"❌ Training failed: {str(e)}"

# --- Get Training Logs ---
def get_logs():
    log_file = sorted(Path("./logs").glob("training_*.log"))[-1]
    with open(log_file, "r") as f:
        return f.read()

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
                    label="Prompt",
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
            with gr.Row():
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

        # --- Logs Tab ---
        with gr.TabItem("Logs"):
            logs = gr.Textbox(label="Training Logs", interactive=False, lines=20)
            refresh_btn = gr.Button("Refresh Logs")
            refresh_btn.click(fn=get_logs, outputs=logs)

# --- Deploy ---
demo.launch()

