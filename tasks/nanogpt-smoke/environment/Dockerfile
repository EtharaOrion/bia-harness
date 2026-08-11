FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir "torch>=2.6"

WORKDIR /app/environment
RUN mkdir -p /logs/artifacts/runs /logs/verifier
