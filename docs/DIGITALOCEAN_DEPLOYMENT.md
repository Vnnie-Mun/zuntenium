# Zuntenium — DigitalOcean Production Deployment Guide

This guide walks through deploying the entire **Zuntenium Quantum SDK, Cloud Platform, and Next.js Workspace** to DigitalOcean with automated CI/CD via **GitHub Actions**.

---

## 1. Architecture Overview

```
User Browser / CLI / IDE
          │
          ▼  (Port 80 / 443 with Auto-SSL)
    ┌───────────────┐
    │     Caddy     │  Reverse Proxy & Automatic Let's Encrypt TLS
    └───┬───────┬───┘
        │       │
   (UI Routes)  │  (/auth, /jobs, /backends, /docs, /health)
        │       ▼
        │   ┌─────────────────────┐
        │   │  FastAPI Gateway    │ (Port 8000)
        │   └──────────┬──────────┘
        │              │
        │       ┌──────┴──────┐
        ▼       ▼             ▼
  ┌──────────┐ ┌─────────┐   ┌───────────────────────┐
  │ Next.js  │ │  Redis  │──►│ 4 RQ Distributed     │
  │ Visual   │ └─────────┘   │ Quantum Workers       │
  │ QCanvas  │       ▲       │ (High, Default, Low,  │
  └──────────┘       │       │  MHOS Hardware)       │
                     ▼       └───────────────────────┘
               ┌──────────┐
               │ Postgres │ (UUID-typed persistent DB)
               └──────────┘
```

The stack includes:
1. **Caddy**: Reverse proxy with automatic Let's Encrypt HTTPS and HTTP/3 support.
2. **Next.js 15 UI (`zuntenium-ui`)**: Standalone interactive QCanvas quantum workspace.
3. **FastAPI Cloud Gateway (`zuntenium_cloud`)**: Authentication, quota management, backend discovery, and job submission.
4. **Distributed RQ Worker Cluster**:
   - `worker-high`: SLA priority quantum jobs.
   - `worker-std`: Standard variational & simulation workflows.
   - `worker-low`: High-qubit batch simulation.
   - `worker-hw`: Physical quantum processors (IBM Quantum, IonQ, AWS Braket, Rigetti).
5. **Real-Time Collab Server (`zuntenium-collab-server`)**: WebSocket CRDT multi-user circuit canvas.
6. **PostgreSQL 15 & Redis 7**: Persistent state and task broker.

---

## 2. DigitalOcean Droplet Preparation

### Recommended Droplet Specifications
- **Operating System**: Ubuntu 22.04 LTS or 24.04 LTS
- **Plan**: Basic or Regular Droplet (Minimum 2GB RAM / 1 vCPU, Recommended: **4GB RAM / 2 vCPUs** for optimal container performance).
- **Authentication**: SSH Key.

### Initial Server Bootstrap (One Command)

SSH into your Droplet as root and execute the automated setup script:

```bash
ssh root@YOUR_DROPLET_IP
curl -fsSL https://raw.githubusercontent.com/Vnnie-Mun/zuntenium/main/scripts/setup_droplet.sh | bash
```

This script automatically:
1. Installs Docker CE & Docker Compose plugin.
2. Configures UFW firewall to allow SSH (22), HTTP (80), and HTTPS (443).
3. Clones the repository to `/opt/zuntenium`.
4. Creates `/opt/zuntenium/.env` with cryptographically secure random secrets.
5. Builds and starts the production container stack.

---

## 3. GitHub Actions Continuous Deployment Setup

Every push to the `main` branch can automatically build, test, and deploy to your Droplet.

### Configure GitHub Secrets

Navigate to your GitHub repository:
**Settings** ➔ **Secrets and variables** ➔ **Actions** ➔ **New repository secret**

Add the following three secrets:

| Secret Name | Value | Description |
|---|---|---|
| `DO_HOST` | `143.198.xxx.xxx` (or `zuntenium.yourdomain.com`) | Public IP address or domain of your Droplet |
| `DO_USER` | `root` | SSH user |
| `DO_SSH_KEY` | `-----BEGIN OPENSSH PRIVATE KEY-----...` | Private SSH key authorized on your Droplet |
| `DO_SSH_PORT` *(Optional)* | `22` | SSH port (defaults to 22 if omitted) |

> [!TIP]
> **Generating a dedicated deployment SSH key pair:**
> ```bash
> ssh-keygen -t ed25519 -C "github-actions-deploy" -f ~/.ssh/zuntenium_deploy
> ```
> Add the public key (`~/.ssh/zuntenium_deploy.pub`) to your Droplet's `/root/.ssh/authorized_keys`.
> Paste the private key (`~/.ssh/zuntenium_deploy`) into the GitHub Secret `DO_SSH_KEY`.

---

## 4. Custom Domain & HTTPS Setup

1. In your DNS provider (DigitalOcean DNS, Cloudflare, Namecheap, etc.), create an **A Record**:
   - Host: `@` or `quantum`
   - Value: `YOUR_DROPLET_IP`

2. On your Droplet, edit `/opt/zuntenium/.env`:
   ```bash
   nano /opt/zuntenium/.env
   ```
   Change:
   ```ini
   DOMAIN="quantum.yourdomain.com"
   ```

3. Restart the reverse proxy:
   ```bash
   cd /opt/zuntenium
   docker compose -f docker-compose.prod.yml restart caddy
   ```
   Caddy will automatically provision a free Let's Encrypt SSL certificate within seconds!

---

## 5. Verifying Deployment Health

Check container status:
```bash
docker compose -f docker-compose.prod.yml ps
```

All services should display `healthy` or `running`:
```
NAME                    STATUS                  PORTS
zuntenium-caddy         Up (healthy)            0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp
zuntenium-ui            Up (healthy)            3000/tcp
zuntenium-api           Up (healthy)            8000/tcp
zuntenium-collab        Up (healthy)            3001/tcp
zuntenium-worker-high   Up (running)            
zuntenium-worker-std    Up (running)            
zuntenium-worker-low    Up (running)            
zuntenium-worker-hw     Up (running)            
zuntenium-postgres      Up (healthy)            5432/tcp
zuntenium-redis         Up (healthy)            6379/tcp
```

### Smoke Test Endpoints

```bash
# 1. API Health Check
curl -s http://localhost/health
# Output: {"status":"ok","version":"4.0.0","timestamp":"..."}

# 2. Public Backends Discovery
curl -s http://localhost/backends

# 3. Next.js Web UI
curl -I http://localhost/
# Output: HTTP/1.1 200 OK
```

---

## 6. Maintenance Commands

| Task | Command |
|---|---|
| View live logs | `docker compose -f docker-compose.prod.yml logs -f` |
| View API logs only | `docker compose -f docker-compose.prod.yml logs -f api` |
| View Worker logs | `docker compose -f docker-compose.prod.yml logs -f worker-std` |
| Restart services | `docker compose -f docker-compose.prod.yml restart` |
| Clean shutdown | `docker compose -f docker-compose.prod.yml down` |
| Pull and rebuild | `docker compose -f docker-compose.prod.yml up -d --build` |
| Start monitoring (Prometheus + Grafana) | `docker compose -f docker-compose.prod.yml --profile monitoring up -d` |
