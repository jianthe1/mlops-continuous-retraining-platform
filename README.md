graph TD
    A[Google Colab] -->|1. Log Model & Metrics| B(MLflow Registry)
    B -->|2. Trigger Build| C[GitHub Actions CI/CD]
    C -->|3. Containerize & Push| D(DockerHub)
    D -->|4. Helm Deploy| E[Kubernetes Cluster]
    E -->|5. Metrics & Drift Check| F[Evidently AI / Prometheus]
    F -->|6. Alert Drift| C
