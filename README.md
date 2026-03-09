# 🚌 Bus Distance Calculator
### ML-Powered Bus Range Prediction & Fleet Efficiency System

A web application that predicts how far a bus can travel based on fuel level, load, speed, temperature, and route type — with efficiency insights for fleet management.

> **Live Demo:** https://bus-distance.streamlit.app/

---

## What It Does

Enter details like:
```
Fuel: 75%, Load: 1200kg, Speed: 65kmph, Temp: 22°C, Route: Highway
```
And the system will:
- Predict the distance the bus can travel
- Score the fuel efficiency
- Analyze load and temperature impact
- Give route optimization recommendations
- Support bulk fleet analysis via CSV upload

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| App | Python, Streamlit |
| ML Model | Scikit-learn, Random Forest |
| Containerization | Docker, Docker Compose |
| Reverse Proxy | Nginx |
| CI/CD | GitHub Actions |
| Deployment | Streamlit Cloud |

---

## Project Structure
```
Bus-Distance-Calculator/
├── .github/
│   └── workflows/
│       └── docker-build.yml   # CI/CD pipeline
├── backend/
│   ├── bus.py                 # main Streamlit app
│   ├── main.py                # ML training logic
│   ├── fuel_dataset.csv       # training dataset
│   ├── requirements.txt       # Python dependencies
│   └── Dockerfile             # containerises the app
├── frontend/
│   └── nginx.conf             # reverse proxy config
├── scripts/
│   └── deploy.sh              # one-command local deploy
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## Run Locally

### Without Docker
```bash
git clone https://github.com/mohitjain2306/Bus-Distance-Calculator.git
cd Bus-Distance-Calculator
pip install -r backend/requirements.txt
cd backend
streamlit run bus.py
```
Then open http://localhost:8501

### With Docker
```bash
git clone https://github.com/mohitjain2306/Bus-Distance-Calculator.git
cd Bus-Distance-Calculator
docker compose up --build
```
- Frontend (nginx): http://localhost:80
- Backend (streamlit): http://localhost:8501

---

## DevOps Setup

### Docker
The backend is containerized using Docker. The frontend is served using Nginx.
```bash
docker build -t bus-distance-calculator ./backend
docker run -p 8501:8501 bus-distance-calculator
```

### Docker Compose
Runs both frontend and backend together with one command:
```bash
docker compose up
```

### CI/CD Pipeline
Every push to main automatically:
1. Builds the Docker image
2. Pushes it to DockerHub

DockerHub image: `mohitjain2306/bus-distance-calculator`

---

## Input Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| Fuel Level | Current fuel percentage | 75% |
| Vehicle Load | Total load in kilograms | 1200 kg |
| Speed | Average speed | 65 kmph |
| Temperature | Outside temperature | 22°C |
| Route Type | Highway, Urban, or Rural | Highway |

---

## Features

- **Distance Prediction** — ML model predicts range based on current conditions
- **Efficiency Scoring** — rates fuel efficiency and highlights issues
- **Load Impact Analysis** — shows how weight affects performance
- **Route Comparison** — highway vs urban vs rural performance
- **Bulk Fleet Analysis** — upload CSV to analyse entire fleet at once

---

## Author

**Mohit Jain**
- GitHub: [@mohitjain2306](https://github.com/mohitjain2306)
- Live App: [bus-distance.streamlit.app](https://bus-distance.streamlit.app/)
- DockerHub: [mohitjain2306](https://hub.docker.com/u/mohitjain2306)
