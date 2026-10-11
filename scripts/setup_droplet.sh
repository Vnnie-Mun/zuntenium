#!/usr/bin/env bash
# ==============================================================================
# scripts/setup_droplet.sh — DigitalOcean Droplet Provisioning & Setup Script
# ==============================================================================
# Run on your DigitalOcean Droplet (Ubuntu 22.04 or 24.04 LTS):
#   curl -fsSL https://raw.githubusercontent.com/Vnnie-Mun/zuntenium/main/scripts/setup_droplet.sh | bash
# Or execute locally on the droplet:
#   sudo bash scripts/setup_droplet.sh
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}====================================================================${NC}"
echo -e "${BLUE}       ZUNTENIUM QUANTUM PLATFORM — DIGITALOCEAN DROPLET SETUP      ${NC}"
echo -e "${BLUE}====================================================================${NC}"

# 1. Require root or sudo
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}[ERROR] This script must be run as root or with sudo.${NC}"
   exit 1
fi

APP_DIR="/opt/zuntenium"
REPO_URL="https://github.com/Vnnie-Mun/zuntenium.git"

# 2. System update & prerequisites
echo -e "\n${YELLOW}[1/6] Updating system packages...${NC}"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    ufw \
    openssl

# 3. Install Docker & Docker Compose if missing
echo -e "\n${YELLOW}[2/6] Verifying Docker Engine & Docker Compose...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "Installing official Docker repository and packages..."
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    systemctl enable --now docker
    echo -e "${GREEN}Docker installed successfully.${NC}"
else
    echo -e "${GREEN}Docker is already installed ($(docker --version)).${NC}"
fi

# Ensure docker compose plugin exists
if ! docker compose version &> /dev/null; then
    echo -e "${RED}[ERROR] Docker Compose plugin could not be verified.${NC}"
    exit 1
fi

# 4. Configure Firewall
echo -e "\n${YELLOW}[3/6] Configuring UFW Firewall...${NC}"
ufw allow 22/tcp comment 'SSH' || true
ufw allow 80/tcp comment 'HTTP' || true
ufw allow 443/tcp comment 'HTTPS' || true
ufw allow 443/udp comment 'HTTP3' || true
ufw --force enable || true
echo -e "${GREEN}Firewall active for ports 22, 80, 443.${NC}"

# 5. Clone or update repository
echo -e "\n${YELLOW}[4/6] Setting up project repository in ${APP_DIR}...${NC}"
if [ -d "${APP_DIR}/.git" ]; then
    echo "Updating existing repository at ${APP_DIR}..."
    cd "${APP_DIR}"
    git fetch origin main
    git reset --hard origin/main
else
    echo "Cloning repository to ${APP_DIR}..."
    mkdir -p "${APP_DIR}"
    git clone "${REPO_URL}" "${APP_DIR}"
    cd "${APP_DIR}"
fi

# 6. Generate production .env if not exists
echo -e "\n${YELLOW}[5/6] Checking production environment secrets...${NC}"
if [ ! -f "${APP_DIR}/.env" ]; then
    echo "Generating secure .env configuration..."
    cp "${APP_DIR}/.env.production.example" "${APP_DIR}/.env"

    # Generate cryptographic random passwords
    JWT_SECRET=$(openssl rand -hex 32)
    DB_PASSWORD=$(openssl rand -hex 24)

    sed -i "s|ZUNTENIUM_JWT_SECRET=\"generate-a-strong-secret-in-production-with-openssl\"|ZUNTENIUM_JWT_SECRET=\"${JWT_SECRET}\"|g" "${APP_DIR}/.env"
    sed -i "s|POSTGRES_PASSWORD=\"generate-a-secure-db-password-with-openssl\"|POSTGRES_PASSWORD=\"${DB_PASSWORD}\"|g" "${APP_DIR}/.env"

    chmod 600 "${APP_DIR}/.env"
    echo -e "${GREEN}Created ${APP_DIR}/.env with random secrets.${NC}"
else
    echo -e "${GREEN}${APP_DIR}/.env already exists.${NC}"
fi

# 7. Start full production stack
echo -e "\n${YELLOW}[6/6] Launching Zuntenium production containers...${NC}"
cd "${APP_DIR}"
docker compose -f docker-compose.prod.yml pull || true
docker compose -f docker-compose.prod.yml up -d --build --remove-orphans

echo -e "\n${GREEN}====================================================================${NC}"
echo -e "${GREEN}   ZUNTENIUM PLATFORM DEPLOYED SUCCESSFULLY ON DIGITALOCEAN!       ${NC}"
echo -e "${GREEN}====================================================================${NC}"
echo -e "Platform Services:"
echo -e "  • Web IDE & QCanvas:    http://$(curl -s ifconfig.me || echo 'YOUR_DROPLET_IP')/"
echo -e "  • REST API Gateway:     http://$(curl -s ifconfig.me || echo 'YOUR_DROPLET_IP')/health"
echo -e "  • Interactive API Docs: http://$(curl -s ifconfig.me || echo 'YOUR_DROPLET_IP')/docs"
echo -e "  • Real-Time Collab:     http://$(curl -s ifconfig.me || echo 'YOUR_DROPLET_IP')/collab"
echo -e ""
echo -e "Helpful commands:"
echo -e "  cd ${APP_DIR}"
echo -e "  docker compose -f docker-compose.prod.yml ps          # Check container status"
echo -e "  docker compose -f docker-compose.prod.yml logs -f     # Follow container logs"
echo -e "  docker compose -f docker-compose.prod.yml restart     # Restart all services"
echo -e "===================================================================="
