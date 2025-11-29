# Self-Learning AI (Dedicated Server + Self-Updating)

## 🚀 Deployment

### 1. Server Setup (Ubuntu 22.04)
```bash
# Install Docker
sudo apt update
sudo apt install docker.io docker-compose git
sudo systemctl enable docker
sudo systemctl start docker

# Clone the repo
git clone https://github.com/your-username/self-learning-ai.git
cd self-learning-ai

# Set GitHub token
echo "GITHUB_TOKEN=your_github_token" >> .env

# Build and run
docker-compose up --build -d

