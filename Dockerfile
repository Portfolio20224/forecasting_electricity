FROM python:3.13-slim

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_VERSION=1.8.2 \
    PYTHONPATH=/app

RUN pip install poetry==$POETRY_VERSION

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install --without dev --no-root

COPY . .

EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]