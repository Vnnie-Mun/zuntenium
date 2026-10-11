# ZQL-14 | Workspace Guide

## Learning Objectives
- Deploy the Zuntenium Cloud backend microservices.
- Launch the QCanvas Next.js interface.
- Install and configure the VS Code extension.

---

## 1. Zuntenium Cloud Microservices

Zuntenium isn't just an SDK; it includes a full distributed cloud execution engine built on FastAPI, Redis, and PostgreSQL.

### Architecture
- **API Server:** A FastAPI application (`zuntenium_cloud/api.py`) accepting REST and WebSocket requests.
- **Queue System:** Redis is used with Python-RQ for distributed job queuing.
- **Database:** PostgreSQL stores job results, telemetry, and user profiles.
- **Workers:** RQ Workers execute quantum circuits asynchronously.

### Deployment via Docker Compose

To spin up the entire cloud platform locally:
```bash
cd zuntenium_cloud
docker-compose up -d
```
*This starts the Postgres database, Redis queue, FastAPI server (port 8000), and background execution workers.*

---

## 2. QCanvas UI (Next.js)

The interactive web portal, **QCanvas**, allows for visual circuit building, job monitoring, and result visualization.

### Running QCanvas Locally

```bash
cd zuntenium-ui
npm install
npm run dev
```

The portal will be available at `http://localhost:3000`. It automatically communicates with your local Zuntenium Cloud API.

### Key Features
1. **Drag-and-Drop Circuit Builder:** Visually place gates on the canvas.
2. **Real-time Bloch Spheres:** Visualize statevector evolution interactively.
3. **Job Dashboard:** Monitor RQ job queues and retrieve execution results from PostgreSQL.

---

## 3. VS Code Extension

For developers who prefer IDE integration, Zuntenium provides a VS Code extension that bridges the gap between text editors and the quantum execution engine.

### Installation
1. Locate the `.vsix` file in the `zuntenium-vscode` directory.
2. Open VS Code and navigate to the Extensions tab.
3. Click the `...` menu at the top right of the extensions pane and select **Install from VSIX...**
4. Select `zuntenium-vscode-0.1.0.vsix`.

### Usage
- Open any `.qasm` (OpenQASM 2.0) file.
- Right-click in the editor.
- Select **Zuntenium: Run QASM Circuit**.
- Results and transpilation telemetry will stream directly into the VS Code Output panel.

## Next Steps
- Dive into advanced quantum physics and algorithmic workflows in [ZQL-05 | Variational and Hybrid](ZQL-05-variational-hybrid.md).
