# syntax=docker/dockerfile:1.2
FROM python:latest
# put you docker configuration here

# Step 2. Copy local code to the container image.
ENV APP_HOME /challenge
WORKDIR $APP_HOME
COPY . ./

# Step 3. Install production dependencies.
RUN pip install -r requirements.txt

# Step 4: Run the web service on container startup using gunicorn webserver.
CMD exec gunicorn --bind :8080 --workers 1 --worker-class uvicorn.workers.UvicornWorker  --threads 8 api:app