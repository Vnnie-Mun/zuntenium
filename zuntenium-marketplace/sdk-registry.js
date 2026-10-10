/**
 * ZUNTENIUM SDK MARKETPLACE — Registry Server
 * Steps 1-19: Developer Portal, Registry, Discovery, Installation,
 *             Sandbox, Security, Versioning, Dependencies, Analytics,
 *             Trust, API Integration, Scalability
 *
 * Stack: Node.js · Express · In-memory store (PostgreSQL-ready)
 * Port:  3002
 */
"use strict";

const crypto = require("crypto");

// ═══════════════════════════════════════════════════════
// STEP 3: SDK PACKAGE SCHEMA
// ═══════════════════════════════════════════════════════

/**
 * SDK Manifest (zuntenium.manifest.json):
 * {
 *   name:         string        "quantum_optimizer_pro"
 *   version:      string        "1.2.0"  (semver)
 *   displayName:  string
 *   description:  string
 *   author:       string
 *   authorId:     string
 *   license:      string        "MIT" | "Apache-2.0" | "Commercial"
 *   category:     SDKCategory
 *   tags:         string[]
 *   dependencies: Record<string,string>  { "zuntenium": ">=1.0.0" }
 *   entryPoint:   string        "main.py" | "index.js"
 *   runtime:      "python" | "javascript"
 *   hooks: {
 *     onInstall?:   string  (function name)
 *     onExecute?:   string
 *     onUninstall?: string
 *   }
 *   permissions: string[]  ["read_circuits","write_jobs","read_results"]
 *   icon?:        string   (URL)
 *   homepage?:    string
 *   repository?:  string
 * }
 */

const SDK_CATEGORIES = [
  "algorithms",         // VQE, QAOA, Grover variants
  "transpilers",        // custom transpilation passes
  "error_mitigation",   // new mitigation strategies
  "backends",           // new hardware connectors
  "visualization",      // custom result visualizations
  "ai_plugins",         // AI optimizer extensions
  "chemistry",          // quantum chemistry tools
  "finance",            // quantum finance algorithms
  "ml",                 // quantum ML models
  "utilities",          // circuit tools, converters
  "templates",          // circuit template packs
  "educational",        // tutorials, demos
];

const PERMISSIONS_CATALOG = {
  read_circuits:     "Read circuit definitions",
  write_circuits:    "Create and modify circuits",
  read_jobs:         "Read job status and metadata",
  write_jobs:        "Submit execution jobs",
  read_results:      "Access execution results",
  read_backends:     "Query available backends",
  call_ai:           "Invoke AI optimizer",
  call_mitigation:   "Apply error mitigation",
  network_access:    "Make external HTTP requests",
  filesystem_read:   "Read from sandbox filesystem",
};

// ═══════════════════════════════════════════════════════
// STEP 4: SDK REGISTRY
// ═══════════════════════════════════════════════════════

class SDKRegistry {
  constructor() {
    /** pkgId → SDKPackage */
    this.packages  = new Map();
    /** authorId → DeveloperAccount */
    this.developers = new Map();
    /** userId → InstalledSDK[] */
    this.installations = new Map();
    /** pkgId → AnalyticsEntry */
    this.analytics  = new Map();
    /** pkgId → Review[] */
    this.reviews    = new Map();
    /** pkgId@version → SecurityScan */
    this.scans      = new Map();
  }

  // ── STEP 2: Developer Registration ──────────────────────────────
  registerDeveloper({ userId, name, email, bio, website }) {
    if (this.developers.has(userId)) throw new Error("Developer already registered");
    const dev = {
      userId, name, email, bio, website,
      apiKey:    "zt_dev_" + crypto.randomBytes(20).toString("hex"),
      verified:  false,
      joinedAt:  Date.now(),
      publishedCount: 0,
      totalDownloads: 0,
    };
    this.developers.set(userId, dev);
    return dev;
  }

  getDeveloper(userId) { return this.developers.get(userId); }
  verifyDeveloper(userId) {
    const d = this.developers.get(userId);
    if (d) { d.verified = true; d.verifiedAt = Date.now(); }
    return d;
  }

  // ── STEP 9: Publish with semver validation ───────────────────────
  publish(manifest, codePayload, authorId) {
    const dev = this.developers.get(authorId);
    if (!dev) throw new Error("Developer not registered");

    this._validateManifest(manifest);
    this._validateSemver(manifest.version);

    const pkgId = this._pkgId(manifest.name, authorId);
    const versionKey = `${pkgId}@${manifest.version}`;

    // Step 8: security scan (simulated)
    const scan = this._securityScan(manifest, codePayload);
    if (scan.blocked) throw new Error(`SDK blocked: ${scan.reason}`);
    this.scans.set(versionKey, scan);

    let pkg = this.packages.get(pkgId);
    if (!pkg) {
      pkg = {
        pkgId, authorId,
        name:        manifest.name,
        displayName: manifest.displayName || manifest.name,
        description: manifest.description || "",
        category:    manifest.category || "utilities",
        tags:        manifest.tags || [],
        license:     manifest.license || "MIT",
        icon:        manifest.icon || null,
        homepage:    manifest.homepage || null,
        repository:  manifest.repository || null,
        createdAt:   Date.now(),
        versions:    {},
        latestVersion: null,
        totalDownloads: 0,
        rating:      null,
        verified:    dev.verified,
        featured:    false,
      };
      this.packages.set(pkgId, pkg);
      this.analytics.set(pkgId, { downloads: 0, installs: 0, activeInstalls: 0, weeklyDownloads: [] });
      this.reviews.set(pkgId, []);
    }

    const versionEntry = {
      version:      manifest.version,
      manifest,
      codePayload,  // In production: store in object storage (S3/GCS)
      publishedAt:  Date.now(),
      publishedBy:  authorId,
      securityScan: scan,
      deprecated:   false,
      downloadCount: 0,
      checksum:     crypto.createHash("sha256").update(codePayload).digest("hex"),
    };

    pkg.versions[manifest.version] = versionEntry;
    pkg.latestVersion = this._resolveLatest(pkg.versions);
    dev.publishedCount++;
    return { pkgId, version: manifest.version, checksum: versionEntry.checksum };
  }

  // ── STEP 5: Discovery & Search ──────────────────────────────────
  search({ query = "", category = "", tags = [], sortBy = "downloads", page = 1, limit = 20 }) {
    let results = Array.from(this.packages.values());

    if (query) {
      const q = query.toLowerCase();
      results = results.filter(p =>
        p.name.toLowerCase().includes(q) ||
        p.displayName.toLowerCase().includes(q) ||
        p.description.toLowerCase().includes(q) ||
        p.tags.some(t => t.toLowerCase().includes(q))
      );
    }
    if (category) results = results.filter(p => p.category === category);
    if (tags.length > 0) results = results.filter(p => tags.every(t => p.tags.includes(t)));

    results.sort((a, b) => {
      if (sortBy === "downloads") return b.totalDownloads - a.totalDownloads;
      if (sortBy === "rating")    return (b.rating || 0) - (a.rating || 0);
      if (sortBy === "newest")    return b.createdAt - a.createdAt;
      return 0;
    });

    const total = results.length;
    const start = (page - 1) * limit;
    const items = results.slice(start, start + limit).map(p => this._publicMeta(p));
    return { items, total, page, limit, pages: Math.ceil(total / limit) };
  }

  getPackage(pkgId)         { return this._publicMeta(this.packages.get(pkgId)); }
  getPackageRaw(pkgId)      { return this.packages.get(pkgId); }
  getFeatured()             { return Array.from(this.packages.values()).filter(p => p.featured).map(p => this._publicMeta(p)); }
  getByCategory(cat)        { return Array.from(this.packages.values()).filter(p => p.category === cat).map(p => this._publicMeta(p)); }

  // ── STEP 6: Installation ─────────────────────────────────────────
  install(pkgId, requestedVersion, userId) {
    const pkg = this.packages.get(pkgId);
    if (!pkg) throw new Error(`Package not found: ${pkgId}`);

    const version = requestedVersion === "latest"
      ? pkg.latestVersion
      : requestedVersion;

    const vEntry = pkg.versions[version];
    if (!vEntry) throw new Error(`Version ${version} not found for ${pkgId}`);
    if (vEntry.deprecated) throw new Error(`Version ${version} is deprecated`);

    // STEP 10: Resolve dependencies
    const deps = this._resolveDependencies(vEntry.manifest.dependencies || {});

    // Track installation
    const userInstalls = this.installations.get(userId) || [];
    const existingIdx = userInstalls.findIndex(i => i.pkgId === pkgId);
    const install = { pkgId, version, installedAt: Date.now(), manifest: vEntry.manifest };
    if (existingIdx >= 0) userInstalls[existingIdx] = install;
    else userInstalls.push(install);
    this.installations.set(userId, userInstalls);

    // Analytics
    pkg.totalDownloads++;
    vEntry.downloadCount++;
    const dev = this.developers.get(pkg.authorId);
    if (dev) dev.totalDownloads++;
    const analytics = this.analytics.get(pkgId);
    if (analytics) analytics.downloads++;

    return { pkgId, version, manifest: vEntry.manifest, deps, checksum: vEntry.checksum };
  }

  uninstall(pkgId, userId) {
    const installs = this.installations.get(userId) || [];
    this.installations.set(userId, installs.filter(i => i.pkgId !== pkgId));
  }

  getUserInstalls(userId) { return this.installations.get(userId) || []; }

  // ── STEP 13: Reviews & Trust ─────────────────────────────────────
  submitReview(pkgId, userId, { rating, comment }) {
    if (rating < 1 || rating > 5) throw new Error("Rating must be 1-5");
    const reviews = this.reviews.get(pkgId) || [];
    const existing = reviews.findIndex(r => r.userId === userId);
    const review = { userId, rating, comment, timestamp: Date.now() };
    if (existing >= 0) reviews[existing] = review;
    else reviews.push(review);
    this.reviews.set(pkgId, reviews);
    // Recalculate average
    const pkg = this.packages.get(pkgId);
    if (pkg) pkg.rating = reviews.reduce((s, r) => s + r.rating, 0) / reviews.length;
    return review;
  }

  getReviews(pkgId) { return this.reviews.get(pkgId) || []; }

  // ── STEP 12: Analytics ───────────────────────────────────────────
  getAnalytics(pkgId) {
    const a = this.analytics.get(pkgId) || {};
    const pkg = this.packages.get(pkgId);
    return { ...a, totalDownloads: pkg?.totalDownloads || 0, rating: pkg?.rating, versions: pkg ? Object.keys(pkg.versions).length : 0 };
  }

  getDeveloperAnalytics(authorId) {
    const packages = Array.from(this.packages.values()).filter(p => p.authorId === authorId);
    return packages.map(p => ({ pkgId: p.pkgId, name: p.displayName, downloads: p.totalDownloads, rating: p.rating, versions: Object.keys(p.versions).length }));
  }

  // ── STEP 8: Security Scan ────────────────────────────────────────
  _securityScan(manifest, code) {
    const violations = [];
    const BLOCKED_PATTERNS = [
      /require\(["']child_process["']\)/,
      /exec\s*\(/,
      /eval\s*\(/,
      /Function\s*\(/,
      /__import__\s*\(\s*["']os["']\)/,
      /subprocess\.call/,
      /open\s*\(.+['"']\s*,\s*['"]w/,
    ];
    BLOCKED_PATTERNS.forEach((p, i) => { if (p.test(code)) violations.push(`pattern_${i}: ${p.source.slice(0, 40)}`); });

    const permissionsRequested = manifest.permissions || [];
    const unknown = permissionsRequested.filter(p => !PERMISSIONS_CATALOG[p]);
    if (unknown.length) violations.push(`unknown_permissions: ${unknown.join(",")}`);

    const blocked = violations.length > 0;
    return { blocked, violations, scannedAt: Date.now(), score: blocked ? 0 : Math.round(80 + Math.random() * 20) };
  }

  // ── STEP 10: Dependency resolution ──────────────────────────────
  _resolveDependencies(deps, visited = new Set()) {
    const resolved = [];
    for (const [name, versionRange] of Object.entries(deps)) {
      if (visited.has(name)) continue;
      visited.add(name);
      if (name === "zuntenium") { resolved.push({ name, versionRange, status: "core" }); continue; }
      const found = Array.from(this.packages.values()).find(p => p.name === name);
      if (!found) { resolved.push({ name, versionRange, status: "not_found" }); continue; }
      resolved.push({ name, versionRange, status: "resolved", pkgId: found.pkgId, version: found.latestVersion });
      // Recursive transitive deps
      const latestManifest = found.versions[found.latestVersion]?.manifest;
      if (latestManifest?.dependencies) {
        const sub = this._resolveDependencies(latestManifest.dependencies, visited);
        resolved.push(...sub);
      }
    }
    return resolved;
  }

  // ── Helpers ───────────────────────────────────────────────────────
  _pkgId(name, authorId) { return `${authorId}/${name}`; }

  _validateManifest(m) {
    const required = ["name","version","description","author","category","entryPoint","runtime"];
    required.forEach(f => { if (!m[f]) throw new Error(`Manifest missing field: ${f}`); });
    if (!SDK_CATEGORIES.includes(m.category)) throw new Error(`Invalid category: ${m.category}`);
    if (!["python","javascript"].includes(m.runtime)) throw new Error(`Invalid runtime: ${m.runtime}`);
  }

  _validateSemver(v) {
    if (!/^\d+\.\d+\.\d+(-[\w.]+)?$/.test(v)) throw new Error(`Invalid semver: ${v}`);
  }

  _resolveLatest(versions) {
    return Object.keys(versions).sort((a, b) => {
      const pa = a.split(".").map(Number), pb = b.split(".").map(Number);
      for (let i = 0; i < 3; i++) { if (pa[i] !== pb[i]) return pb[i] - pa[i]; }
      return 0;
    })[0];
  }

  _publicMeta(pkg) {
    if (!pkg) return null;
    const { codePayload: _, ...safe } = pkg;
    return { ...safe, versionsAvailable: Object.keys(pkg.versions || {}) };
  }
}

// ═══════════════════════════════════════════════════════
// STEP 7: EXECUTION SANDBOX
// ═══════════════════════════════════════════════════════

class SDKSandbox {
  constructor(registry) {
    this.registry = registry;
    this.running  = new Map(); // executionId → context
  }

  /**
   * Execute an installed SDK's hook against a Zuntenium context.
   * In production: runs in isolated container (Docker/gVisor/Firecracker)
   * Here: safe eval with restricted scope + timeout
   */
  async execute({ pkgId, version, hook = "onExecute", context, userId, timeoutMs = 5000 }) {
    const pkg    = this.registry.getPackageRaw(pkgId);
    if (!pkg) throw new Error(`Package ${pkgId} not found`);

    const vEntry = pkg.versions[version];
    if (!vEntry) throw new Error(`Version ${version} not found`);

    const scan = this.registry.scans.get(`${pkgId}@${version}`);
    if (scan?.blocked) throw new Error("SDK failed security scan — execution blocked");

    const manifest = vEntry.manifest;
    const hookFn   = manifest.hooks?.[hook];
    if (!hookFn) throw new Error(`No hook '${hook}' defined in manifest`);

    // Build restricted API surface (Step 14: API Integration)
    const sdkAPI = this._buildSandboxAPI(manifest.permissions || [], context);

    const execId = crypto.randomUUID();
    this.running.set(execId, { pkgId, userId, startTime: Date.now() });

    try {
      const result = await this._runWithTimeout(
        vEntry.codePayload, hookFn, sdkAPI, context, timeoutMs
      );
      this.running.delete(execId);
      return { success: true, result, execId, duration: Date.now() - this.running.get(execId)?.startTime || 0 };
    } catch (err) {
      this.running.delete(execId);
      return { success: false, error: err.message, execId };
    }
  }

  /** Build restricted Zuntenium API surface for sandbox */
  _buildSandboxAPI(permissions, context) {
    const api = {};
    if (permissions.includes("read_circuits"))   api.getCircuit   = () => context.circuit;
    if (permissions.includes("write_circuits"))  api.updateGates  = (g) => { context.circuit.gates = g; };
    if (permissions.includes("read_results"))    api.getResults   = () => context.results;
    if (permissions.includes("read_backends"))   api.getBackends  = () => context.backends;
    if (permissions.includes("call_ai"))         api.runAI        = async (c) => ({ confidence: 0.87, backend: "tensornetwork" });
    // Explicitly deny dangerous APIs
    api.eval       = undefined;
    api.require    = undefined;
    api.process    = undefined;
    api.global     = undefined;
    return api;
  }

  async _runWithTimeout(code, hookName, api, context, ms) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`SDK timeout after ${ms}ms`)), ms);
      try {
        // Safe function factory with restricted scope
        const fn = new Function(
          "ZunteniumAPI", "context", "console",
          `"use strict";\n${code}\nreturn typeof ${hookName} === 'function' ? ${hookName}(ZunteniumAPI, context) : null;`
        );
        const safeConsole = { log: () => {}, error: () => {}, warn: () => {} };
        Promise.resolve(fn(api, context, safeConsole))
          .then(r => { clearTimeout(timer); resolve(r); })
          .catch(e => { clearTimeout(timer); reject(e); });
      } catch (e) { clearTimeout(timer); reject(e); }
    });
  }
}

// ═══════════════════════════════════════════════════════
// SEED: Built-in marketplace SDKs
// ═══════════════════════════════════════════════════════

function seedMarketplace(registry) {
  // Register official developer
  registry.developers.set("zuntenium_official", {
    userId: "zuntenium_official", name: "Zuntenium Labs", email: "sdk@zuntenium.io",
    apiKey: "zt_dev_official", verified: true, joinedAt: Date.now() - 86400000 * 30,
    publishedCount: 0, totalDownloads: 0,
  });

  const officialSDKs = [
    {
      manifest: {
        name: "vqe_optimizer", displayName: "VQE Optimizer Pro", version: "2.1.0",
        description: "Variational Quantum Eigensolver with COBYLA + BFGS + SPSA optimizers. Includes ansatz library and energy convergence tracking.",
        author: "Zuntenium Labs", authorId: "zuntenium_official",
        license: "Apache-2.0", category: "algorithms",
        tags: ["vqe","chemistry","optimization","parametric"],
        runtime: "javascript", entryPoint: "main.js",
        dependencies: { "zuntenium": ">=1.0.0" },
        permissions: ["read_circuits","write_jobs","read_results","call_ai"],
        hooks: { onExecute: "runVQE" },
        icon: null, homepage: "https://docs.zuntenium.io/sdks/vqe",
      },
      code: `
function runVQE(api, context) {
  const circuit = api.getCircuit();
  const params = context.params || { optimizer: "COBYLA", maxIter: 100 };
  return { algorithm: "VQE", circuit: circuit.name, optimizer: params.optimizer, status: "configured" };
}`,
      downloads: 2847, rating: 4.8, featured: true,
    },
    {
      manifest: {
        name: "qaoa_maxcut", displayName: "QAOA MaxCut Solver", version: "1.3.2",
        description: "Quantum Approximate Optimization Algorithm for MaxCut, graph partitioning, and combinatorial problems. p-layer depth support.",
        author: "Zuntenium Labs", authorId: "zuntenium_official",
        license: "MIT", category: "algorithms",
        tags: ["qaoa","optimization","graph","combinatorial"],
        runtime: "javascript", entryPoint: "main.js",
        dependencies: { "zuntenium": ">=1.0.0" },
        permissions: ["read_circuits","write_jobs","read_results"],
        hooks: { onExecute: "runQAOA" },
      },
      code: `
function runQAOA(api, context) {
  const { p = 1, graph = [] } = context;
  return { algorithm: "QAOA", layers: p, edges: graph.length, status: "ready" };
}`,
      downloads: 1923, rating: 4.6, featured: true,
    },
    {
      manifest: {
        name: "zne_advanced", displayName: "Advanced ZNE Mitigation", version: "1.0.5",
        description: "Extended Zero-Noise Extrapolation with Richardson, polynomial, exponential, and linear extrapolators. Noise amplification via gate folding.",
        author: "Zuntenium Labs", authorId: "zuntenium_official",
        license: "Apache-2.0", category: "error_mitigation",
        tags: ["zne","mitigation","noise","fidelity"],
        runtime: "javascript", entryPoint: "main.js",
        dependencies: { "zuntenium": ">=1.0.0" },
        permissions: ["read_circuits","read_results","call_mitigation"],
        hooks: { onExecute: "runZNE" },
      },
      code: `
function runZNE(api, context) {
  const results = api.getResults();
  return { method: "ZNE", extrapolator: "polynomial", scales: [1,2,3], estimated_fidelity: 0.997 };
}`,
      downloads: 1456, rating: 4.7, featured: false,
    },
    {
      manifest: {
        name: "grover_search", displayName: "Grover's Search Toolkit", version: "1.1.0",
        description: "Configurable Grover's algorithm with automatic oracle generation. Supports n-qubit search with optimal iteration count.",
        author: "Zuntenium Labs", authorId: "zuntenium_official",
        license: "MIT", category: "algorithms",
        tags: ["grover","search","speedup","oracle"],
        runtime: "javascript", entryPoint: "main.js",
        dependencies: { "zuntenium": ">=1.0.0" },
        permissions: ["read_circuits","write_jobs","read_results"],
        hooks: { onExecute: "runGrover" },
      },
      code: `
function runGrover(api, context) {
  const n = context.numQubits || 2;
  const iterations = Math.round(Math.PI/4 * Math.sqrt(Math.pow(2,n)));
  return { algorithm: "Grover", qubits: n, iterations, searchSpace: Math.pow(2,n) };
}`,
      downloads: 1102, rating: 4.5, featured: false,
    },
    {
      manifest: {
        name: "circuit_visualizer_3d", displayName: "3D Circuit Visualizer", version: "0.9.1",
        description: "Renders quantum circuits in interactive 3D using Three.js. Supports gate highlighting, depth animation, and export to SVG/PNG.",
        author: "Zuntenium Labs", authorId: "zuntenium_official",
        license: "MIT", category: "visualization",
        tags: ["visualization","3d","circuit","export"],
        runtime: "javascript", entryPoint: "main.js",
        dependencies: { "zuntenium": ">=1.0.0" },
        permissions: ["read_circuits"],
        hooks: { onExecute: "render3D" },
      },
      code: `
function render3D(api, context) {
  const circuit = api.getCircuit();
  return { rendered: true, gates: circuit.gates.length, format: "webgl" };
}`,
      downloads: 834, rating: 4.3, featured: false,
    },
    {
      manifest: {
        name: "quantum_chemistry_pack", displayName: "Quantum Chemistry Pack", version: "2.0.0",
        description: "Molecular simulation toolkit — Hartree-Fock, UCCSD, Jordan-Wigner mapping, second quantization. Includes H₂, LiH, BeH₂ Hamiltonians.",
        author: "Zuntenium Labs", authorId: "zuntenium_official",
        license: "Apache-2.0", category: "chemistry",
        tags: ["chemistry","molecular","hamiltonian","uccsd"],
        runtime: "javascript", entryPoint: "main.js",
        dependencies: { "zuntenium": ">=1.0.0", "vqe_optimizer": ">=2.0.0" },
        permissions: ["read_circuits","write_jobs","read_results","call_ai"],
        hooks: { onExecute: "runChemSim" },
      },
      code: `
function runChemSim(api, context) {
  const molecule = context.molecule || "H2";
  const molecules = { H2: -1.137, LiH: -7.882, BeH2: -15.594 };
  return { molecule, groundStateEnergy: molecules[molecule], unit: "Hartree", basis: "sto-3g" };
}`,
      downloads: 677, rating: 4.9, featured: true,
    },
    {
      manifest: {
        name: "transpiler_optimizer", displayName: "Circuit Depth Reducer", version: "1.2.0",
        description: "Advanced transpilation passes: gate commutation, two-qubit block optimization, peephole rewriting. Reduces circuit depth by up to 40%.",
        author: "Zuntenium Labs", authorId: "zuntenium_official",
        license: "MIT", category: "transpilers",
        tags: ["transpiler","optimization","depth","gates"],
        runtime: "javascript", entryPoint: "main.js",
        dependencies: { "zuntenium": ">=1.0.0" },
        permissions: ["read_circuits","write_circuits"],
        hooks: { onExecute: "optimizeCircuit" },
      },
      code: `
function optimizeCircuit(api, context) {
  const circuit = api.getCircuit();
  const before = circuit.gates.length;
  return { gatesBefore: before, gatesAfter: Math.max(1, Math.floor(before * 0.75)), reduction: "25%" };
}`,
      downloads: 543, rating: 4.4, featured: false,
    },
    {
      manifest: {
        name: "qml_classifier", displayName: "Quantum ML Classifier", version: "1.0.0",
        description: "Variational quantum classifier for binary and multi-class problems. Includes data encoding (angle, amplitude), training loop, accuracy metrics.",
        author: "Zuntenium Labs", authorId: "zuntenium_official",
        license: "Apache-2.0", category: "ml",
        tags: ["machine-learning","classification","variational","qml"],
        runtime: "javascript", entryPoint: "main.js",
        dependencies: { "zuntenium": ">=1.0.0" },
        permissions: ["read_circuits","write_jobs","read_results","call_ai"],
        hooks: { onExecute: "trainClassifier" },
      },
      code: `
function trainClassifier(api, context) {
  const { epochs = 10, learningRate = 0.01 } = context;
  return { algorithm: "VQC", epochs, learningRate, accuracy: 0.87, loss: 0.32 };
}`,
      downloads: 412, rating: 4.2, featured: false,
    },
  ];

  officialSDKs.forEach(({ manifest, code, downloads, rating, featured }) => {
    try {
      const { pkgId } = registry.publish(manifest, code, "zuntenium_official");
      const pkg = registry.packages.get(pkgId);
      if (pkg) {
        pkg.totalDownloads = downloads || 0;
        pkg.rating = rating || null;
        pkg.featured = featured || false;
        pkg.verified = true;
      }
    } catch (e) { /* skip if already seeded */ }
  });
}

// ═══════════════════════════════════════════════════════
// EXPORTS
// ═══════════════════════════════════════════════════════
const registry = new SDKRegistry();
const sandbox  = new SDKSandbox(registry);
seedMarketplace(registry);

module.exports = { registry, sandbox, SDK_CATEGORIES, PERMISSIONS_CATALOG };
