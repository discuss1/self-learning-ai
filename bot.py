from github import Github
import os

def commit_to_github(filename: str, code: str):
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo("your-username/self-learning-ai")
    try:
        repo.create_file(
            f"generated/{filename}",
            f"AI-generated {filename}",
            code
        )
        return True
    except Exception as e:
        print(f"GitHub commit failed: {e}")
        return False

