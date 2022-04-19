FROM python:3.8.13-buster

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    TZ=Asia/Tokyo

WORKDIR /app

COPY poetry.lock pyproject.toml ./

RUN apt-get update && apt-get -y install vim xsel

RUN pip install poetry

RUN poetry config virtualenvs.create false \
  && poetry install
