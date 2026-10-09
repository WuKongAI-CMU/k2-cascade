# Production sandbox: one container per rollout. The in-process Workspace confines paths but not the
# shell itself; run this image with --network none --read-only --tmpfs /tmp for real isolation.
FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml ./
COPY k2_tasks_env ./k2_tasks_env
COPY tasks ./tasks
RUN pip install --no-cache-dir pytest
EXPOSE 8000
CMD ["python", "-m", "k2_tasks_env.server", "--port", "8000"]
