FROM python:3.8-slim
WORKDIR /app

# install openjdk 17 jre
RUN apt-get update \
    && apt-get install --no-install-recommends --no-install-suggests -y software-properties-common \
    && apt-get install -y ca-certificates-java; \
    update-ca-certificates -f; \
    apt-get install -y default-jre; \
    apt-cache policy git;\
    apt-get install --no-install-recommends --no-install-suggests -y build-essential=12.9 gcc git; \
    apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies RUN python -m pip install --upgrade pip
RUN pip install poetry && pip cache purge

COPY autotim autotim
COPY main.py main.py
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
COPY data data

RUN poetry install --no-interaction --no-dev --no-root

# set PYTHONPATH
ENV PYTHONPATH /
ENV PYTHONUNBUFFERED true

#CMD poetry run python main.py
COPY wrapper_script.sh wrapper_script.sh
CMD poetry run bash wrapper_script.sh
