# Base image
FROM python:3.10

# Install depedencies
WORKDIR /cyanobateria-monitoring
COPY requirements.txt requirements.txt
COPY .dvc .dvc
RUN python3 -m pip install --upgrade pip\
    && pip install -r requirements.txt

COPY src src
COPY app.py app.py
COPY params.yaml params.yaml
COPY dvc.yaml dvc.yamld


RUN dvc remote modify cyanomonit access_key_id ${aws_access_key_id}
RUN dvc remote modify cyanomonit secret_access_key ${aws_secret_access_key}
RUN dvc fetch
RUN dvc pull --remote cyanomonit
RUN dvc repro

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]