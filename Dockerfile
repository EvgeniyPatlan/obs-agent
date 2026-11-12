# One image for every Python container: bench services, load generator,
# trace sink and the lab (training, evaluation, reports).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RESULTS_DIR=/work/results
RUN pip install --no-cache-dir numpy==1.26.4 matplotlib==3.8.4
WORKDIR /work
COPY obsagent /work/obsagent
COPY bench /work/bench
