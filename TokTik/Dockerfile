FROM --platform=linux/amd64 python:3.9

# Install libraries
COPY ./requirements.txt ./

RUN apt-get update && apt-get install -y git; apt install -y libgl1; apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

RUN pip install -r requirements.txt && \
    rm ./requirements.txt

# Setup container directories
RUN mkdir /app

# Copy local code to the container
COPY ./app /app

# launch server with gunicorn
WORKDIR /app
EXPOSE 8080
CMD ["gunicorn", "main:app", "--timeout=0", "--preload", \
     "--workers=1", "--threads=4", "--bind=0.0.0.0:8080"]