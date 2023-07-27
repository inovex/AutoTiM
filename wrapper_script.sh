#!/bin/bash

# Start the first process
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./artifacts --host 0.0.0.0 &

# Start the second process
poetry run python main.py &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
