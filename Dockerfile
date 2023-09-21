FROM python:3.8-slim
WORKDIR /app
# install openjdk 17 jre
RUN apt-get update \
    && apt-get install --no-install-recommends --no-install-suggests -y software-properties-common>=0.96 \
    && apt-get update;\
    apt-get install --no-install-recommends --no-install-suggests -y ca-certificates-java>=2000000;\
    update-ca-certificates -f;\
    apt-get install --no-install-recommends --no-install-suggests -y openjdk-17-jre=17.*;\
    apt-get install --no-install-recommends --no-install-suggests -y build-essential=12.9 gcc;\
    apt-cache policy git;\
    apt-get install --no-install-recommends --no-install-suggests -y git>=1:2\
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
CMD ["gunicorn", "-t", "0", "-b" , "0.0.0.0:5004", "-R", "-w", "4", "--log-level", "debug", "autotim.app.app:app"]
