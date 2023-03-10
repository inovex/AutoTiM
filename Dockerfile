FROM python:3.8-slim
WORKDIR /app

# install openjdk 11 jre
RUN apt-get update \
    && apt-get install --no-install-recommends --no-install-suggests -y software-properties-common=0.96.24.32.14 \
    && apt-get update \
    && add-apt-repository ppa:openjdk-r/ppa;\
    apt-get install --no-install-recommends --no-install-suggests -y build-essential=12.9 gcc=4:10.2.1-1;\
    apt-get install --no-install-recommends --no-install-suggests -y openjdk-11-jre=11.0.18+10-1~deb11u1; \
    apt-cache policy git;\
    apt-get install --no-install-recommends --no-install-suggests -y git=1:2.30.2-1\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY autotim autotim

# set PYTHONPATH
ENV PYTHONPATH /
ENV FLASK_APP autotim/app/app.py
ENV PYTHONUNBUFFERED true

# expose Flask App
EXPOSE 5004

# Run the application
CMD ["gunicorn", "-t", "0", "-b" , "0.0.0.0:5004", "-R", "--log-level", "debug", "autotim.app.app:app"]
