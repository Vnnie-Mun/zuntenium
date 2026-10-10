/**
 * ZUNTENIUM SDK MARKETPLACE — API Server
 * All REST endpoints: publish, search, install, reviews, analytics,
 * developer portal, trust/verification, dependency resolution
 *
 * Stack: Node.js + http (production: Express/FastAPI)
 * Port: 3002
 */
"use strict";

const http   = require("http");
const crypto = require("crypto");
const { registry, sandbox, SDK_CATEGORIES, PERMISSIONS_CATALOG } = require("./sdk-registry");

const PORT = process.env.MARKETPLACE_PORT || 3002;

// ── Minimal router ────────────────────────────────────────────────────
function createRouter() {
  const routes = [];
  const add = (method, path, handler) => routes.push({ method, path: new RegExp("^" + path.replace(/:([^/]+)/g, "(?<$1>[^/]+)") + "(?:\\?.*)?$"), handler, raw: path });
  return {
    get:    (p, h) => add("GET",    p, h),
    post:   (p, h) => add("POST",   p, h),
    put:    (p, h) => add("PUT",    p, h),
    delete: (p, h) => add("DELETE", p, h),
    handle: async (req, res) => {
      for (const route of routes) {
        if (req.method !== route.method) continue;
        const url   = req.url.split("?")[0];
        const match = url.match(route.path);
        if (!match) continue;
        req.params = match.groups || {};
        req.query  = parseQuery(req.url);
        try { await route.handler(req, res); } catch (e) { sendError(res, 500, e.message); }
        return;
      }
      sendError(res, 404, "Route not found");
    },
  };
}

const router = createRouter();

// ── Middleware helpers ────────────────────────────────────────────────
function parseQuery(url) {
  const idx = url.indexOf("?");
  if (idx < 0) return {};
  return Object.fromEntries(new URLSearchParams(url.slice(idx + 1)));
}

function json(res, data, status = 200) {
  res.writeHead(status, { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" });
  res.end(JSON.stringify(data, null, 2));
}
function sendError(res, status, message) {
  json(res, { error: true, message, status }, status);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", c => { body += c; if (body.length > 5_000_000) reject(new Error("Payload too large")); });
    req.on("end", () => { try { resolve(body ? JSON.parse(body) : {}); } catch { reject(new Error("Invalid JSON")); } });
    req.on("error", reject);
  });
}

// Basic API key auth (production: JWT verification against Cloud Platform)
function requireAuth(req) {
  const key = req.headers["x-api-key"] || req.headers["authorization"]?.replace("Bearer ", "");
  if (!key) throw new Error("Authentication required");
  // Find developer by apiKey
  const dev = Array.from(registry.developers.values()).find(d => d.apiKey === key);
  if (!dev) throw new Error("Invalid API key");
  return dev;
}

function requireAdminAuth(req) {
  const key = req.headers["x-api-key"];
  if (key !== (process.env.ADMIN_KEY || "zt_admin_key")) throw new Error("Admin access required");
}

// ════════════════════════════════════════════════════════════════════
// STEP 1: HEALTH & CATALOG
// ════════════════════════════════════════════════════════════════════

router.get("/health", (req, res) => {
  json(res, {
    status:      "ok",
    version:     "1.0.0",
    packages:    registry.packages.size,
    developers:  registry.developers.size,
    categories:  SDK_CATEGORIES,
    uptime:      process.uptime(),
    timestamp:   Date.now(),
  });
});

router.get("/catalog", (req, res) => {
  json(res, {
    categories:  SDK_CATEGORIES,
    permissions: PERMISSIONS_CATALOG,
    featured:    registry.getFeatured(),
    stats: {
      totalPackages:   registry.packages.size,
      totalDevelopers: registry.developers.size,
      totalDownloads:  Array.from(registry.packages.values()).reduce((s, p) => s + p.totalDownloads, 0),
    },
  });
});

// ════════════════════════════════════════════════════════════════════
// STEP 2: DEVELOPER PORTAL
// ════════════════════════════════════════════════════════════════════

// Register new developer account
router.post("/developers/register", async (req, res) => {
  const body = await readBody(req);
  const { userId, name, email, bio, website } = body;
  if (!userId || !name || !email) return sendError(res, 400, "userId, name, email required");
  try {
    const dev = registry.registerDeveloper({ userId, name, email, bio, website });
    json(res, { success: true, developer: { ...dev, apiKey: dev.apiKey /* shown once */ } }, 201);
  } catch (e) { sendError(res, 409, e.message); }
});

// Get developer profile
router.get("/developers/:userId", (req, res) => {
  const dev = registry.getDeveloper(req.params.userId);
  if (!dev) return sendError(res, 404, "Developer not found");
  const { apiKey: _, ...safe } = dev; // never expose apiKey in GET
  json(res, { developer: safe, packages: registry.getDeveloperAnalytics(req.params.userId) });
});

// Get authenticated developer's own profile + analytics
router.get("/developers/me/profile", (req, res) => {
  try {
    const dev = requireAuth(req);
    json(res, {
      developer: dev,
      packages:  registry.getDeveloperAnalytics(dev.userId),
      installs:  registry.getUserInstalls(dev.userId),
    });
  } catch (e) { sendError(res, 401, e.message); }
});

// Admin: verify developer
router.post("/developers/:userId/verify", async (req, res) => {
  try {
    requireAdminAuth(req);
    const dev = registry.verifyDeveloper(req.params.userId);
    if (!dev) return sendError(res, 404, "Developer not found");
    json(res, { success: true, developer: dev });
  } catch (e) { sendError(res, 403, e.message); }
});

// ════════════════════════════════════════════════════════════════════
// STEP 3+4: PUBLISH SDK
// ════════════════════════════════════════════════════════════════════

// Publish (or update version of) an SDK
router.post("/packages/publish", async (req, res) => {
  let dev;
  try { dev = requireAuth(req); } catch (e) { return sendError(res, 401, e.message); }

  const body = await readBody(req);
  const { manifest, code } = body;
  if (!manifest || !code) return sendError(res, 400, "manifest and code required");

  // Force authorId to match authenticated developer
  manifest.authorId = dev.userId;
  manifest.author   = manifest.author || dev.name;

  try {
    const result = registry.publish(manifest, code, dev.userId);
    json(res, { success: true, ...result, message: `Published ${manifest.name}@${manifest.version}` }, 201);
  } catch (e) { sendError(res, 422, e.message); }
});

// Deprecate a version
router.post("/packages/:pkgId/versions/:version/deprecate", async (req, res) => {
  let dev;
  try { dev = requireAuth(req); } catch (e) { return sendError(res, 401, e.message); }
  const pkg = registry.getPackageRaw(req.params.pkgId);
  if (!pkg) return sendError(res, 404, "Package not found");
  if (pkg.authorId !== dev.userId) return sendError(res, 403, "Not package owner");
  const vEntry = pkg.versions[req.params.version];
  if (!vEntry) return sendError(res, 404, "Version not found");
  vEntry.deprecated = true;
  json(res, { success: true, message: `Version ${req.params.version} deprecated` });
});

// ════════════════════════════════════════════════════════════════════
// STEP 5: DISCOVERY & SEARCH
// ════════════════════════════════════════════════════════════════════

router.get("/packages", (req, res) => {
  const { q = "", category = "", tags = "", sortBy = "downloads", page = "1", limit = "20" } = req.query;
  const result = registry.search({
    query:    q,
    category,
    tags:     tags ? tags.split(",").filter(Boolean) : [],
    sortBy,
    page:     parseInt(page)  || 1,
    limit:    Math.min(parseInt(limit) || 20, 100),
  });
  json(res, result);
});

router.get("/packages/featured", (req, res) => {
  json(res, { packages: registry.getFeatured() });
});

router.get("/packages/category/:category", (req, res) => {
  json(res, { category: req.params.category, packages: registry.getByCategory(req.params.category) });
});

router.get("/packages/:pkgId", (req, res) => {
  const pkg = registry.getPackage(req.params.pkgId);
  if (!pkg) return sendError(res, 404, "Package not found");
  const reviews = registry.getReviews(req.params.pkgId);
  const analytics = registry.getAnalytics(req.params.pkgId);
  json(res, { package: pkg, reviews, analytics });
});

router.get("/packages/:pkgId/versions", (req, res) => {
  const pkg = registry.getPackageRaw(req.params.pkgId);
  if (!pkg) return sendError(res, 404, "Package not found");
  json(res, {
    pkgId: pkg.pkgId,
    latestVersion: pkg.latestVersion,
    versions: Object.entries(pkg.versions).map(([v, e]) => ({
      version:      v,
      publishedAt:  e.publishedAt,
      publishedBy:  e.publishedBy,
      deprecated:   e.deprecated,
      downloadCount:e.downloadCount,
      checksum:     e.checksum,
      securityScore:e.securityScan?.score,
    })),
  });
});

// ════════════════════════════════════════════════════════════════════
// STEP 6: INSTALLATION SYSTEM
// ════════════════════════════════════════════════════════════════════

router.post("/packages/:pkgId/install", async (req, res) => {
  const body = await readBody(req);
  const { userId, version = "latest" } = body;
  if (!userId) return sendError(res, 400, "userId required");
  try {
    const result = registry.install(req.params.pkgId, version, userId);
    json(res, { success: true, installed: result });
  } catch (e) { sendError(res, 422, e.message); }
});

router.delete("/packages/:pkgId/uninstall", async (req, res) => {
  const body = await readBody(req);
  const { userId } = body;
  if (!userId) return sendError(res, 400, "userId required");
  registry.uninstall(req.params.pkgId, userId);
  json(res, { success: true, message: `Uninstalled ${req.params.pkgId}` });
});

router.get("/users/:userId/installs", (req, res) => {
  json(res, { userId: req.params.userId, installed: registry.getUserInstalls(req.params.userId) });
});

// Resolve dependency tree before install
router.post("/packages/:pkgId/resolve", async (req, res) => {
  const body = await readBody(req);
  const { version = "latest" } = body;
  const pkg = registry.getPackageRaw(req.params.pkgId);
  if (!pkg) return sendError(res, 404, "Package not found");
  const v = version === "latest" ? pkg.latestVersion : version;
  const vEntry = pkg.versions[v];
  if (!vEntry) return sendError(res, 404, "Version not found");
  const deps = registry["_resolveDependencies"](vEntry.manifest.dependencies || {});
  json(res, { pkgId: req.params.pkgId, version: v, dependencies: deps });
});

// ════════════════════════════════════════════════════════════════════
// STEP 7: SANDBOX EXECUTION
// ════════════════════════════════════════════════════════════════════

router.post("/packages/:pkgId/execute", async (req, res) => {
  const body = await readBody(req);
  const { userId, version = "latest", hook = "onExecute", context = {} } = body;
  if (!userId) return sendError(res, 400, "userId required");

  const installs = registry.getUserInstalls(userId);
  const install  = installs.find(i => i.pkgId === req.params.pkgId);
  if (!install) return sendError(res, 403, `Package ${req.params.pkgId} not installed for user ${userId}`);

  const result = await sandbox.execute({
    pkgId:   req.params.pkgId,
    version: install.version,
    hook,
    context,
    userId,
    timeoutMs: 8000,
  });
  json(res, result);
});

// ════════════════════════════════════════════════════════════════════
// STEP 8: SECURITY
// ════════════════════════════════════════════════════════════════════

router.get("/packages/:pkgId/security/:version", (req, res) => {
  const key  = `${req.params.pkgId}@${req.params.version}`;
  const scan = registry.scans.get(key);
  if (!scan) return sendError(res, 404, "Security scan not found");
  json(res, { pkgId: req.params.pkgId, version: req.params.version, scan });
});

// Admin: manually block a package
router.post("/packages/:pkgId/block", async (req, res) => {
  try { requireAdminAuth(req); } catch (e) { return sendError(res, 403, e.message); }
  const pkg = registry.getPackageRaw(req.params.pkgId);
  if (!pkg) return sendError(res, 404, "Package not found");
  pkg.blocked = true;
  pkg.blockedAt = Date.now();
  json(res, { success: true, message: `Package ${req.params.pkgId} blocked` });
});

// Admin: feature a package
router.post("/packages/:pkgId/feature", async (req, res) => {
  try { requireAdminAuth(req); } catch (e) { return sendError(res, 403, e.message); }
  const pkg = registry.getPackageRaw(req.params.pkgId);
  if (!pkg) return sendError(res, 404, "Package not found");
  pkg.featured = true;
  json(res, { success: true, message: `Package ${req.params.pkgId} featured` });
});

// ════════════════════════════════════════════════════════════════════
// STEP 12: ANALYTICS
// ════════════════════════════════════════════════════════════════════

router.get("/packages/:pkgId/analytics", (req, res) => {
  json(res, registry.getAnalytics(req.params.pkgId));
});

router.get("/developers/:userId/analytics", (req, res) => {
  json(res, { packages: registry.getDeveloperAnalytics(req.params.userId) });
});

router.get("/analytics/global", (req, res) => {
  const packages = Array.from(registry.packages.values());
  json(res, {
    totalPackages:   packages.length,
    totalDevelopers: registry.developers.size,
    totalDownloads:  packages.reduce((s, p) => s + p.totalDownloads, 0),
    topPackages:     packages.sort((a, b) => b.totalDownloads - a.totalDownloads).slice(0, 10).map(p => ({ pkgId: p.pkgId, name: p.displayName, downloads: p.totalDownloads, rating: p.rating })),
    byCategory:      SDK_CATEGORIES.reduce((acc, cat) => {
      acc[cat] = packages.filter(p => p.category === cat).length;
      return acc;
    }, {}),
  });
});

// ════════════════════════════════════════════════════════════════════
// STEP 13: REVIEWS & TRUST
// ════════════════════════════════════════════════════════════════════

router.post("/packages/:pkgId/reviews", async (req, res) => {
  const body = await readBody(req);
  const { userId, rating, comment } = body;
  if (!userId || !rating) return sendError(res, 400, "userId and rating required");
  try {
    const review = registry.submitReview(req.params.pkgId, userId, { rating: parseInt(rating), comment });
    json(res, { success: true, review }, 201);
  } catch (e) { sendError(res, 422, e.message); }
});

router.get("/packages/:pkgId/reviews", (req, res) => {
  json(res, { pkgId: req.params.pkgId, reviews: registry.getReviews(req.params.pkgId) });
});

// ════════════════════════════════════════════════════════════════════
// HTTP SERVER
// ════════════════════════════════════════════════════════════════════

const server = http.createServer(async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type,X-Api-Key,Authorization");
  if (req.method === "OPTIONS") { res.writeHead(204); res.end(); return; }
  await router.handle(req, res);
});

server.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════════════════════════╗
║  ZUNTENIUM SDK MARKETPLACE API v1.0                      ║
║  REST: http://localhost:${PORT}                            ║
║  Health: http://localhost:${PORT}/health                   ║
║  Catalog: http://localhost:${PORT}/catalog                 ║
║  Packages: ${registry.packages.size} seeded · ${registry.developers.size} developers            ║
╚══════════════════════════════════════════════════════════╝
  `);
});

module.exports = { server };
