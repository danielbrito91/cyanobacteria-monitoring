# Base image
FROM python:3.10-slim

# Install depedencies
WORKDIR /cyanobateria-monitoring
COPY requirements/prod_req.txt requirements.txt
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
RUN python3 -m pip install --upgrade pip\
    && pip install -r requirements.txt

#COPY .dvc .dvc
COPY src/ src/
COPY app.py app.py
#COPY data/ data/
COPY reports/metrics.json reports/metrics.json
COPY params.yaml params.yaml
#COPY dvc.yaml dvc.yaml

RUN dvc init --no-scm -f
RUN dvc remote modify storage access_key_id ${AWS_ACCESS_KEY_ID}
RUN dvc remote modify storage secret_access_key ${AWS_SECRET_ACCESS_KEY}
RUN dvc pull

#
#RUN dvc fetch
#RUN dvc remote add -d storage data
#RUN dvc pull
#RUN dvc repro

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]