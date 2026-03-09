# start from an official Python image (slim = smaller size)
FROM python:3.11-slim

# set the working directory inside the container
WORKDIR /app

# copy requirements first (Docker caches this layer if requirements don't change)
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the app files
COPY . .

# expose the port Streamlit runs on
EXPOSE 8501

# tell Docker how to run the app
CMD ["streamlit", "run", "bus.py", "--server.port=8501", "--server.address=0.0.0.0"]
