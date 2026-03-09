# 🚌 Bus Distance Calculator

A machine learning web app that predicts how far a bus can travel based on fuel, load, speed, and route — containerised with Docker and deployed with a CI/CD pipeline.

🔗 **Live App**: https://bus-distance-calculator.streamlit.app/

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| App | Python, Streamlit, Scikit-learn |
| Containerisation | Docker, Docker Compose |
| Reverse Proxy | Nginx |
| CI/CD | GitHub Actions |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

\`\`\`
Bus-Distance-Calculator/
├── backend/                  # Streamlit ML app
│   ├── bus.py                # main app file
│   ├── main.py               # ML training logic
│   ├── fuel_dataset.csv      # training data
│   ├── requirements.txt      # Python dependencies
│   └── Dockerfile            # containerises the app
├── frontend/
│   └── nginx.conf            # reverse proxy config
├── scripts/
│   └── deploy.sh             # one-command local deploy
├── .github/
│   └── workflows/
│       └── docker-build.yml  # CI/CD pipeline
├── docker-compose.yml
└── .gitignore
\`\`\`

---

## 🚀 How to Run Locally

### Option 1 — with the deploy script
\`\`\`bash
git clone https://github.com/mohitjain2306/Bus-Distance-Calculator.git
cd Bus-Distance-Calculator
bash scripts/deploy.sh
\`\`\`

### Option 2 — with Docker Compose
\`\`\`bash
docker compose up --build
\`\`\`

### Option 3 — plain Python
\`\`\`bash
cd backend
pip install -r requirements.txt
streamlit run bus.py
\`\`\`

---

## ⚙️ CI/CD Pipeline

Every push to \`main\` triggers GitHub Actions to:
1. Build the Docker image
2. Push it to DockerHub as \`latest\`

Add these secrets in your GitHub repo settings:
- \`DOCKERHUB_USERNAME\`
- \`DOCKERHUB_TOKEN\`

---

## 👤 Author

**Mohit Jain** — Fresher | DevOps & Backend

- GitHub: [@mohitjain2306](https://github.com/mohitjain2306)
- Live App: [Bus Distance Calculator](https://bus-distance-calculator.streamlit.app/)
- DockerHub: [mohitjain2306](https://hub.docker.com/u/mohitjain2306)
