# zuntenium_cloud/__init__.py
"""
Zuntenium Cloud Deployment Platform — Phase 4
===============================================
A production-grade, cloud-native quantum computing platform that orchestrates
the entire Zuntenium stack across distributed workers with multi-tenancy,
job queuing, observability, and horizontal scaling.

Architecture layers:
  API Gateway (FastAPI)     — REST + JWT auth + rate limiting
  Job Manager               — submission, validation, lifecycle
  Task Queue (Redis + RQ)   — distributed job dispatch
  Execution Workers         — stateless Dockerized Zuntenium runners
  Backend Connectors        — unified QuantumBackend registry
  Storage Layer             — PostgreSQL (jobs/users) + Redis (queue/cache)
  Observability             — Prometheus metrics + structured logging
"""

__version__ = "4.0.0"
