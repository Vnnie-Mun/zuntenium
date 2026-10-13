"use client";
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Activity, Braces, CircleHelp, Cloud, Copy, Download, FileCode2, FolderOpen, GitBranch, LogIn, LogOut, Minus, Plus, Play, Redo2, RotateCcw, Save, Share2, Square, Trash2, Undo2, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { GATE_LIBRARY, describeCondition, makeGate, noiseOptions, touched, validateCircuit, type Circuit, type Gate, type GateType, type NoiseModel, type RunResult } from '@/lib/quantum/model';
import { download, downloadResults } from '@/lib/quantum/export';
import { gatesToQasm, parseQasm } from '@/lib/quantum/qasm';
import { cancelJob, clearSession, fetchBackends, fetchJob, fetchJobResult, loginAccount, registerAccount, saveSession, storedAccessToken, submitCircuit, type CloudBackend } from '@/lib/qcanvas/zuntenium-api';

function ShareDialog({ open, onOpenChange }: { projectId: string; projectName: string; ownerEmail: string | null; open: boolean; onOpenChange: (v: boolean) => void }) {
  if (!open) return null;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader><DialogTitle>Share Project</DialogTitle><DialogDescription>Collaboration coming soon.</DialogDescription></DialogHeader>
      </DialogContent>
    </Dialog>
  );
}

type Tab = 'probabilities' | 'counts' | 'amplitudes' | 'bloch' | 'qsphere';
type History = { past: Circuit[]; present: Circuit; future: Circuit[] };
const initial: Circuit = { version: 1, numQubits: 3, gates: [] };
const groups = ['Single qubit', 'Rotations', 'Controlled', 'Measurement'];
const ROW = 62; const COL = 66; const START = 90;
const maxColumn = (c: Circuit) => Math.max(4, ...c.gates.map(g => g.column + 2));
const safeFilename = 'quflow-circuit.qasm';
type ProjectRole = 'owner' | 'viewer';
type ProjectRow = { id:string; name:string; updated_at:string; circuit:Circuit; owner_id:'local' };
const PROJECT_STORAGE_KEY = 'quflow.workspace.projects.v1';
const TOKEN_STORAGE_KEY = 'quflow.access-token';

function cloudResultToRunResult(raw: Awaited<ReturnType<typeof fetchJobResult>>, numQubits: number, requestedShots: number): RunResult {
  const statevector = (raw.statevector ?? []).map(([re, im], index) => {
    const probability = re * re + im * im;
    return {
      index,
      label: index.toString(2).padStart(numQubits, '0'),
      amplitude: { re, im },
      probability,
      phase: Math.atan2(im, re),
    };
  });
  const counts = raw.counts ?? {};
  const blochVectors = Array.from({ length: numQubits }, (_, qubit) => {
    const mask = 1 << qubit;
    let x = 0;
    let y = 0;
    let z = 0;
    for (let index = 0; index < statevector.length; index++) {
      if ((index & mask) !== 0) continue;
      const zero = statevector[index]?.amplitude;
      const one = statevector[index | mask]?.amplitude;
      if (!zero || !one) continue;
      x += 2 * (zero.re * one.re + zero.im * one.im);
      y += 2 * (zero.re * one.im - zero.im * one.re);
      z += zero.re ** 2 + zero.im ** 2 - one.re ** 2 - one.im ** 2;
    }
    return { qubit, x, y, z };
  });
  const countTotal = Object.values(counts).reduce((total, count) => total + count, 0);
  return {
    backend: 'zuntenium',
    backendName: raw.backend_used ?? 'unknown',
    numQubits,
    statevector,
    statevectorTruncated: false,
    totalNonzeroStates: statevector.filter(state => state.probability > 0).length,
    blochVectors,
    counts,
    shots: countTotal || requestedShots,
    noise: null,
    midCircuitMeasurement: false,
    hasClassicalControl: false,
    noiseShotsCapped: false,
    executionTimeMs: Math.round((raw.execution_time_s ?? 0) * 1000),
    seed: null,
  };
}

function wait(ms: number) {
  return new Promise<void>(resolve => window.setTimeout(resolve, ms));
}

export default function QCanvasWorkspace() {
  const [history, setHistory] = useState<History>({ past: [], present: initial, future: [] });
  const circuit = history.present;
  const [tool, setTool] = useState<GateType | null>(null);
  const [pending, setPending] = useState<number[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [conditional, setConditional] = useState(false);
  const [shots, setShots] = useState(1024);
  const [noise, setNoise] = useState<NoiseModel>('none');
  const [probability, setProbability] = useState(0.02);
  const [seed, setSeed] = useState('');
  const [result, setResult] = useState<RunResult | null>(null);
  const [resultCircuit, setResultCircuit] = useState<Circuit | null>(null);
  const [tab, setTab] = useState<Tab>('probabilities');
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState('');
  const [angleDialog, setAngleDialog] = useState<{ type: GateType; qubits: number[]; column: number } | null>(null);
  const [angle, setAngle] = useState('1.57079632679');
  const [conditionDialog, setConditionDialog] = useState<{ gate: Gate } | null>(null);
  const [, setConditionSource] = useState(0);
  const [conditionBits, setConditionBits] = useState<number[]>([]);
  const [conditionReg, setConditionReg] = useState('1');
  const [showImport, setShowImport] = useState(false);
  const [importText, setImportText] = useState('');
  const [showHelp, setShowHelp] = useState(false);
  const [showAccount, setShowAccount] = useState(false);
  const [showProjects, setShowProjects] = useState(false);
  const [showShare, setShowShare] = useState(false);
  const [userEmail, setUserEmail] = useState<string | null>(null);
  const [authMode, setAuthMode] = useState<'signin' | 'signup'>('signin');
  const [authEmail, setAuthEmail] = useState('');
  const [authUsername, setAuthUsername] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [projectName, setProjectName] = useState('Untitled circuit');
  const [projectId, setProjectId] = useState<string | null>(null);
  const [projects, setProjects] = useState<ProjectRow[]>([]);
  const [role, setRole] = useState<ProjectRole | null>(null);
  const [selectedBackend, setSelectedBackend] = useState('local');
  const [cloudBackends, setCloudBackends] = useState<CloudBackend[]>([]);
  const userId = userEmail ?? 'local';
  function googleSignIn() {
    setMessage('Google sign-in is not configured. Please use email or run locally.');
  }
  const worker = useRef<Worker | null>(null);
  const remoteJobId = useRef<string | null>(null);
  const [focusCell, setFocusCell] = useState<[number, number]>([0, 0]);
  function gridKey(e: React.KeyboardEvent, q: number, column: number) {
    const moves: Record<string, [number, number]> = { ArrowUp:[-1,0], ArrowDown:[1,0], ArrowLeft:[0,-1], ArrowRight:[0,1] };
    let next: [number, number] | null = null;
    if (moves[e.key]) { const [dq,dc] = moves[e.key]!; next = [Math.min(circuit.numQubits-1, Math.max(0,q+dq)), Math.min(columns-1, Math.max(0,column+dc))]; }
    else if (e.key === 'Home') next = [q,0]; else if (e.key === 'End') next = [q,columns-1];
    if (!next) return;
    e.preventDefault(); e.stopPropagation(); setFocusCell(next);
    document.querySelector<HTMLElement>(`[data-cell="${next[0]}-${next[1]}"]`)?.focus();
  }
  const requestId = useRef(0);
  const fileInput = useRef<HTMLInputElement>(null);
  const clipboard = useRef<Gate[]>([]);
  const latest = useRef<() => void>(() => {});
  const selectedRef = useRef(selected); selectedRef.current = selected;
  const columns = maxColumn(circuit);
  useEffect(() => { setFocusCell(([q,c]) => [Math.min(q, circuit.numQubits-1), Math.min(c, columns-1)]); }, [circuit.numQubits, columns]);
  const measured = useMemo(() => [...new Set(circuit.gates.filter(g => g.type === 'MEASURE').map(g => g.targets[0]))], [circuit.gates]);

  function commit(next: Circuit) {
    try { validateCircuit(next); setHistory(h => ({ past: [...h.past.slice(-99), h.present], present: next, future: [] })); setMessage(''); }
    catch (e) { setMessage(e instanceof Error ? e.message : 'Invalid circuit.'); }
  }
  function undo() { setHistory(h => h.past.length ? { past:h.past.slice(0,-1), present:h.past[h.past.length-1] ?? h.present, future:[h.present,...h.future] } : h); setSelected([]); }
  function redo() { setHistory(h => h.future.length ? { past:[...h.past,h.present], present:h.future[0] ?? h.present, future:h.future.slice(1) } : h); setSelected([]); }
  function addGate(type: GateType, qs: number[], column: number, theta?: number) {
    const gate = makeGate(type,qs,column,theta);
    if (conditional && type !== 'MEASURE') {
      const sources = circuit.gates.filter(g => g.type === 'MEASURE' && g.column < column).map(g => g.targets[0]);
      if (!sources.length) { setMessage('Place a measurement before a conditional gate.'); return; }
      setConditionSource(sources[0] ?? 0); setConditionBits([sources[0] ?? 0]); setConditionReg('1'); setConditionDialog({ gate }); return;
    }
    commit({ ...circuit, gates:[...circuit.gates,gate] });
  }
  function place(q: number, column: number) {
    if (!tool) return;
    if (pending.includes(q)) return;
    const count = GATE_LIBRARY.find(g => g.type === tool)?.qubits ?? 1;
    const qs = [...pending,q];
    if (qs.length < count) { setPending(qs); return; }
    setPending([]);
    const occupied = circuit.gates.some(g => g.column === column && [...qs].some(row => touched(g).includes(row)));
    if (occupied) { setMessage('That step is occupied. Choose another cell.'); return; }
    if (['RX','RY','RZ'].includes(tool)) { setAngleDialog({ type:tool, qubits:qs, column }); return; }
    addGate(tool,qs,column);
  }
  function removeSelected() {
    if (!selected.length) return;
    commit({ ...circuit, gates:circuit.gates.filter(g => !selected.includes(g.id)) }); setSelected([]);
  }
  function copySelected() { clipboard.current = circuit.gates.filter(g => selected.includes(g.id)); if (clipboard.current.length) setMessage(`${clipboard.current.length} gate(s) copied.`); }
  function paste() {
    if (!clipboard.current.length) return;
    const min = Math.min(...clipboard.current.map(g => g.column));
    const at = Math.max(0,...circuit.gates.map(g => g.column+1));
    const copies = clipboard.current.map(g => ({ ...g, id:crypto.randomUUID(), column:at + g.column - min }));
    const next = { ...circuit, gates:[...circuit.gates,...copies] };
    try { validateCircuit(next); commit(next); setSelected(copies.map(g => g.id)); } catch(e) { setMessage(e instanceof Error ? e.message : 'Cannot paste these gates.'); }
  }
  function moveSelected(delta: number) {
    if (!selected.length) return;
    const next = { ...circuit, gates:circuit.gates.map(g => selected.includes(g.id) ? { ...g, column:g.column+delta } : g) };
    commit(next);
  }
  async function run() {
    if (running) return;
    if (!circuit.gates.length) { setMessage('Add a gate to run this circuit.'); return; }
    const parsedSeed = seed.trim() ? Number(seed) : null;
    try {
      validateCircuit(circuit);
      if (parsedSeed !== null && (!Number.isSafeInteger(parsedSeed) || parsedSeed < 0 || parsedSeed > 4294967295)) throw new Error('Seed must be a whole number from 0 to 4,294,967,295.');
      if (!Number.isInteger(shots) || shots < 1 || shots > 20000) throw new Error('Shots must be between 1 and 20,000.');
      setMessage(''); setRunning(true);
      const id = ++requestId.current;
      if (selectedBackend !== 'local') {
        if (!accessToken) {
          setRunning(false);
          setShowAccount(true);
          setMessage('Sign in to connect to Zuntenium backends.');
          return;
        }
        const selectedCapability = cloudBackends.find(item => item.name === selectedBackend);
        if (!selectedCapability?.available) throw new Error(`${selectedBackend} is not currently available from the Zuntenium API.`);
        const hasDynamicCircuit = circuit.gates.some((gate, index) =>
          gate.classicalControl || (gate.type === 'MEASURE' && circuit.gates.slice(index + 1).some(next => next.type !== 'MEASURE')),
        );
        if (noise !== 'none' || parsedSeed !== null || hasDynamicCircuit) {
          throw new Error('This circuit uses noise, a seed, or mid-circuit feedback. Run it on the local simulator; the current cloud API does not accept those options.');
        }
        const remoteCircuit = {
          num_qubits: circuit.numQubits,
          name: projectName,
          gates: circuit.gates.filter(gate => gate.type !== 'MEASURE').map(gate => ({
            name: gate.type === 'CNOT' ? 'cx' : gate.type.toLowerCase(),
            qubits: [...gate.controls, ...gate.targets],
            params: gate.params.theta === undefined ? [] : [gate.params.theta],
          })),
        };
        try {
          const job = await submitCircuit(accessToken, remoteCircuit, {
            backend: selectedBackend,
            shots,
            mitigation: false,
          });
          remoteJobId.current = job.id;
          for (let attempt = 0; attempt < 225; attempt++) {
            if (requestId.current !== id) return;
            const status = await fetchJob(accessToken, job.id);
            if (status.status === 'failed') throw new Error(status.last_error || 'Zuntenium job failed.');
            if (status.status === 'cancelled') throw new Error('Zuntenium job was cancelled.');
            if (status.status === 'completed') {
              const remoteResult = await fetchJobResult(accessToken, job.id);
              if (requestId.current !== id) return;
              const normalized = cloudResultToRunResult(remoteResult, circuit.numQubits, shots);
              setResult(normalized);
              setResultCircuit(structuredClone(circuit));
              setTab(normalized.statevector.length ? 'probabilities' : 'counts');
              setMessage(`Completed on ${normalized.backendName}.`);
              return;
            }
            await wait(800);
          }
          throw new Error('Zuntenium job timed out while waiting for a result.');
        } catch (error) {
          if (requestId.current === id) setMessage(error instanceof Error ? error.message : 'Zuntenium job failed.');
        } finally {
          if (requestId.current === id) {
            remoteJobId.current = null;
            setRunning(false);
          }
        }
        return;
      }
      const instance = new Worker(new URL('../lib/quantum/simulator.worker.ts', import.meta.url), { type:'module' });
      worker.current = instance;
      instance.onmessage = (event: MessageEvent<{ result?: RunResult; error?: string }>) => {
        if (requestId.current !== id) return;
        setRunning(false); worker.current = null; instance.terminate();
        if (event.data.error) setMessage(event.data.error);
        else if (event.data.result) { setResult(event.data.result); setResultCircuit(structuredClone(circuit)); setTab('probabilities'); }
      };
      instance.onerror = () => { if (requestId.current === id) { setRunning(false); setMessage('Simulation could not finish. Try fewer qubits or shots.'); worker.current = null; } instance.terminate(); };
      instance.postMessage({ circuit, shots, model:noise, p:probability, seed:parsedSeed });
    } catch (e) { setRunning(false); setMessage(e instanceof Error ? e.message : 'Unable to run circuit.'); }
  }
  latest.current = run;
  function stop() {
    requestId.current++;
    worker.current?.terminate();
    worker.current = null;
    const jobId = remoteJobId.current;
    const token = accessToken;
    remoteJobId.current = null;
    if (jobId && token) void cancelJob(token, jobId).catch(() => {});
    setRunning(false);
    setMessage(jobId ? 'Cancellation requested from Zuntenium.' : 'Run cancelled.');
  }
  function importQasm() { try { const next = parseQasm(importText); stopIfRunning(); commit(next); setSelected([]); setShowImport(false); setMessage(`Imported ${next.gates.length} gates.`); } catch (e) { setMessage(e instanceof Error ? e.message : 'Invalid QASM.'); } }
  function stopIfRunning() { if (worker.current) { requestId.current++; worker.current.terminate(); worker.current=null; setRunning(false); } }
  async function authenticate() {
    setMessage('');
    const username = authMode === 'signup' ? authUsername.trim() : authEmail.trim();
    if (username.length < 3 || authPassword.length < 8) {
      setMessage(authMode === 'signup' ? 'Enter a username and password of at least 8 characters.' : 'Enter your username and password.');
      return;
    }
    try {
      if (authMode === 'signup') {
        const email = authEmail.trim().toLowerCase();
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(email)) throw new Error('Enter a valid email address.');
        await registerAccount(email, username, authPassword);
      }
      const session = await loginAccount(username, authPassword);
      localStorage.setItem(TOKEN_STORAGE_KEY, session.access_token);
          saveSession(session);
      setAccessToken(session.access_token);
      setUserEmail(username);
      setAuthPassword('');
      const response = await fetchBackends(session.access_token);
      setCloudBackends(response.backends);
      setShowAccount(false);
      setMessage('Connected to the Zuntenium API.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Zuntenium sign-in failed.');
    }
  }
  function signOut() {
    clearSession();
    setAccessToken(null);
    setUserEmail(null);
    setCloudBackends([]);
    setSelectedBackend('local');
    setShowAccount(false);
    setMessage('Signed out. Browser-local projects remain on this device.');
  }
  function saveProject() {
    try {
      validateCircuit(circuit);
      const id = projectId ?? `local-${Date.now().toString(36)}`;
      const project: ProjectRow = { id, name:projectName.trim().slice(0, 120) || 'Untitled circuit', updated_at:new Date().toISOString(), circuit:structuredClone(circuit), owner_id:'local' };
      const next = [project, ...projects.filter(item => item.id !== id)].slice(0, 100);
      localStorage.setItem(PROJECT_STORAGE_KEY, JSON.stringify(next));
      setProjects(next);
      setProjectId(id);
      setRole('owner');
      setProjectName(project.name);
      setMessage('Circuit saved in this browser.');
    } catch (error) {
      setMessage(error instanceof Error ? `Not saved: ${error.message}` : 'Circuit could not be saved.');
    }
  }
  function openProjects() {
    try {
      const stored = JSON.parse(localStorage.getItem(PROJECT_STORAGE_KEY) ?? '[]') as ProjectRow[];
      const valid = stored.filter(project => {
        try { validateCircuit(project.circuit); return true; } catch { return false; }
      });
      setProjects(valid);
      setShowProjects(true);
    } catch {
      setProjects([]);
      setMessage('Saved browser projects could not be read.');
    }
  }
  function loadProject(project: ProjectRow) {
    try {
      validateCircuit(project.circuit);
      stopIfRunning(); commit(project.circuit); setRole('owner'); setProjectId(project.id); setProjectName(project.name); setShowProjects(false); setSelected([]);
      setMessage(`Opened “${project.name}”.`);
    } catch { setMessage('That saved circuit is invalid and was not opened.'); }
  }
  function deleteProject(id: string, name: string) {
    if (!window.confirm(`Delete “${name}” from this browser?`)) return;
    const next = projects.filter(project => project.id !== id);
    localStorage.setItem(PROJECT_STORAGE_KEY, JSON.stringify(next));
    setProjects(next);
    if (projectId === id) { setProjectId(null); setRole(null); }
    setMessage(`Deleted “${name}”.`);
  }
  function duplicateProject(project: ProjectRow) {
    try {
      validateCircuit(project.circuit);
      const name = `${project.name} (copy)`.slice(0, 120);
      const duplicate: ProjectRow = { ...project, id:`local-${Date.now().toString(36)}`, name, updated_at:new Date().toISOString(), circuit:structuredClone(project.circuit) };
      const next = [duplicate, ...projects].slice(0, 100);
      localStorage.setItem(PROJECT_STORAGE_KEY, JSON.stringify(next));
      setProjects(next);
      setMessage(`Created “${name}” on this device.`);
    } catch { setMessage('That circuit could not be duplicated.'); }
  }
  function newCircuit() {
    if (circuit.gates.length && !window.confirm('Start a new circuit? Unsaved changes will be lost.')) return;
    stopIfRunning(); setHistory({ past:[], present:initial, future:[] }); setProjectId(null); setRole(null); setProjectName('Untitled circuit'); setSelected([]); setResult(null); setResultCircuit(null); setMessage('New circuit started.');
  }
  function duplicateSelected() { if (!selected.length) return; copySelected(); paste(); }
  useEffect(() => { return () => worker.current?.terminate(); }, []);
  useEffect(() => {
    const savedProjects = localStorage.getItem(PROJECT_STORAGE_KEY);
    if (savedProjects) {
      try { setProjects(JSON.parse(savedProjects) as ProjectRow[]); } catch { localStorage.removeItem(PROJECT_STORAGE_KEY); }
    }
    const token = storedAccessToken();
    if (!token) return;
    setAccessToken(token);
    void fetchBackends(token).then(response => setCloudBackends(response.backends)).catch(() => {
      clearSession();
      setAccessToken(null);
    });
  }, []);
  useEffect(() => {
    if (!projectId || !role) return;
    const timer = window.setTimeout(() => {
      try {
        validateCircuit(circuit);
        setProjects(current => {
          const next = current.map(project => project.id === projectId
            ? { ...project, name:projectName.trim() || 'Untitled circuit', circuit:structuredClone(circuit), updated_at:new Date().toISOString() }
            : project);
          localStorage.setItem(PROJECT_STORAGE_KEY, JSON.stringify(next));
          return next;
        });
      } catch { /* keep invalid edits unsaved until corrected */ }
    }, 700);
    return () => window.clearTimeout(timer);
  }, [circuit, projectId, projectName, role]);
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.closest('input,textarea,[contenteditable="true"]')) return;
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo(); }
      else if (mod && e.key === 'Enter') { e.preventDefault(); latest.current(); }
      else if (mod && e.key.toLowerCase() === 'c') { e.preventDefault(); copySelected(); }
      else if (mod && e.key.toLowerCase() === 'v') { e.preventDefault(); paste(); }
      else if (mod && e.key.toLowerCase() === 'a') { e.preventDefault(); setSelected(circuit.gates.map(g => g.id)); }
      else if (mod && e.key.toLowerCase() === 'd') { e.preventDefault(); duplicateSelected(); }
      else if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); removeSelected(); }
      else if (e.key === 'Escape') { setTool(null); setPending([]); setSelected([]); setAngleDialog(null); setConditionDialog(null); }
      else if (e.key === 'ArrowLeft' && selectedRef.current.length) { e.preventDefault(); moveSelected(-1); }
      else if (e.key === 'ArrowRight' && selectedRef.current.length) { e.preventDefault(); moveSelected(1); }
    };
    window.addEventListener('keydown',handler); return () => window.removeEventListener('keydown',handler);
  });

  return <div className="qc-app">
    <header className="qc-topbar">
      <div className="qc-brand"><span className="qc-mark"><Activity size={19}/></span><span>Q<span className="qc-brand-light">canvas</span></span><span className="qc-brand-divider"/><span className="qc-workspace-name">{projectName}</span></div>
      <div className="qc-top-actions"><span className="qc-local-status" role="status"><span/> {!projectId ? 'Local workspace' : 'Saved on this device'}</span><Button variant="ghost" size="icon" aria-label="Help" title="Help" onClick={() => setShowHelp(true)}><CircleHelp size={17}/></Button><Button variant="outline" size="sm" onClick={() => setShowAccount(true)}>{accessToken ? <Cloud/> : <LogIn/>}{accessToken ? userEmail : 'Connect'}</Button></div>
    </header>
    <div className="qc-toolbar"><div className="qc-toolbar-left"><span className="qc-eyebrow">CIRCUIT EDITOR</span><span className="qc-toolbar-divider"/><Button variant="ghost" size="icon" aria-label="Undo" title="Undo" disabled={!history.past.length} onClick={undo}><Undo2/></Button><Button variant="ghost" size="icon" aria-label="Redo" title="Redo" disabled={!history.future.length} onClick={redo}><Redo2/></Button><span className="qc-toolbar-divider"/><Button variant="ghost" size="icon" aria-label="Copy selected gates" title="Copy selected gates" disabled={!selected.length} onClick={copySelected}><Copy/></Button><Button variant="ghost" size="icon" aria-label="Delete selected gates" title="Delete selected gates" disabled={!selected.length} onClick={removeSelected}><Trash2/></Button><Button variant="ghost" size="icon" aria-label="Duplicate selected gates" title="Duplicate selected gates (Ctrl+D)" disabled={!selected.length} onClick={duplicateSelected}><Plus/></Button><Button variant="ghost" size="icon" aria-label="Clear circuit" title="Clear circuit" disabled={!circuit.gates.length} onClick={() => { commit({ ...circuit, gates:[] }); setSelected([]); }}><RotateCcw/></Button></div>
      <div className="qc-toolbar-right"><Button variant="outline" size="sm" onClick={newCircuit}><Plus/> New</Button><Button variant="outline" size="sm" onClick={openProjects}><FolderOpen/> Projects</Button><Button variant="outline" size="sm" onClick={saveProject}><Save/> {role === 'viewer' ? 'Save copy' : 'Save'}</Button>{projectId && role === 'owner' && <Button variant="outline" size="sm" onClick={() => setShowShare(true)}><Share2/> Share</Button>}<Button variant="outline" size="sm" onClick={() => setShowImport(true)}><FolderOpen/> Import QASM</Button><Button variant="outline" size="sm" disabled={!circuit.gates.length} onClick={() => download(safeFilename,gatesToQasm(circuit.numQubits,circuit.gates))}><FileCode2/> Export QASM</Button><Button size="sm" onClick={running ? stop : run}>{running ? <><Square/> Stop</> : <><Play fill="currentColor"/> Run circuit</>}</Button></div>
    </div>
    <main className="qc-workarea">
      <aside className="qc-palette"><div className="qc-section-head"><span>GATE LIBRARY</span><span>{GATE_LIBRARY.length}</span></div><div className="qc-palette-scroll">{groups.map(group => <section key={group} className="qc-gate-group"><h3>{group}</h3><div className="qc-gate-list">{GATE_LIBRARY.filter(g => g.group === group).map(g => <Button key={g.type} variant="ghost" className={`qc-gate-item ${tool === g.type ? 'is-active' : ''}`} aria-pressed={tool === g.type} title={`${g.label} (${g.qubits} qubit${g.qubits > 1 ? 's' : ''})`} onClick={() => { setTool(tool === g.type ? null : g.type); setPending([]); setSelected([]); }}><span className={`qc-gate-symbol qc-gate-${g.group.replace(' ','-').toLowerCase()}`}>{g.symbol}</span><span className="qc-gate-label">{g.label}</span><span className="qc-gate-meta">{g.qubits}q</span></Button>)}</div></section>)}</div><div className="qc-palette-foot"><div className="qc-section-head">EXECUTION BACKEND</div><div className="qc-backend-current"><span className="qc-backend-icon"><Activity size={15}/></span><div><strong>Local simulator</strong><small>Statevector engine</small></div><span className="qc-active-dot"/></div><p className="qc-unavailable">Zuntenium unavailable — API documentation and configuration required.</p></div></aside>
      <div className="qc-center"><section className="qc-circuit-section"><div className="qc-panel-head"><div><span className="qc-eyebrow">WORKSPACE / {!projectId ? 'LOCAL CIRCUIT' : role === 'owner' ? 'SAVED PROJECT' : 'SHARED PROJECT'}</span><input className="qc-project-title" aria-label="Circuit name" maxLength={120} value={projectName} onChange={e => setProjectName(e.target.value)}/></div><div className="qc-circuit-summary"><span>{circuit.numQubits} qubits</span><span>{circuit.gates.length} gates</span><span>Depth {circuit.gates.length ? Math.max(...circuit.gates.map(g => g.column))+1 : 0}</span></div></div>
      <div className="qc-config"><div className="qc-config-cell"><label>QUBITS</label><div className="qc-stepper"><Button variant="ghost" size="icon" aria-label="Remove qubit" disabled={circuit.numQubits <= 1 || circuit.gates.some(g => touched(g).includes(circuit.numQubits-1))} onClick={() => commit({ ...circuit, numQubits:circuit.numQubits-1 })}><Minus/></Button><strong>{circuit.numQubits}</strong><Button variant="ghost" size="icon" aria-label="Add qubit" disabled={circuit.numQubits >= 12} onClick={() => commit({ ...circuit, numQubits:circuit.numQubits+1 })}><Plus/></Button></div></div><div className="qc-config-cell"><label htmlFor="qc-shots">SHOTS</label><Input id="qc-shots" type="number" min={1} max={20000} value={shots} onChange={e => setShots(Number(e.target.value))}/></div><div className="qc-config-cell"><label>NOISE MODEL</label><Select value={noise} onValueChange={v => setNoise(v as NoiseModel)}><SelectTrigger aria-label="Noise model"><SelectValue/></SelectTrigger><SelectContent>{noiseOptions.map(n => <SelectItem value={n.id} key={n.id}>{n.label}</SelectItem>)}</SelectContent></Select></div>{noise !== 'none' && <div className="qc-config-cell"><label htmlFor="qc-noise">PROBABILITY</label><Input id="qc-noise" type="number" min={0} max={1} step={0.01} value={probability} onChange={e => setProbability(Number(e.target.value))}/></div>}<div className="qc-config-cell"><label htmlFor="qc-seed">SEED <span>OPTIONAL</span></label><Input id="qc-seed" type="number" min={0} max={4294967295} value={seed} placeholder="Random" onChange={e => setSeed(e.target.value)}/></div><div className="qc-config-cell qc-condition-cell"><label>CLASSICAL CONTROL</label><Button variant={conditional ? 'secondary' : 'outline'} size="sm" disabled={!measured.length} onClick={() => setConditional(v => !v)} title={measured.length ? 'Condition the next gate on an earlier measurement' : 'Place a measurement first'}><GitBranch/> {conditional ? 'On' : 'Off'}</Button></div></div>
      <div className="qc-canvas-scroller" role="group" aria-label={`Circuit grid, ${circuit.numQubits} qubits by ${columns} steps. Use arrow keys to move, Enter to place.`}><div className="qc-grid" style={{ width:START + columns*COL + 24, height:58+circuit.numQubits*ROW }}><div className="qc-ruler" style={{gridTemplateColumns:`${START}px repeat(${columns}, ${COL}px)`}}><span/> {Array.from({length:columns},(_,i) => <span key={i}>{String(i+1).padStart(2,'0')}</span>)}</div>{Array.from({length:circuit.numQubits},(_,q) => <div className="qc-wire-row" key={q} style={{top:44+q*ROW,width:START+columns*COL}}><span className="qc-wire-label">q<sub>{q}</sub><small>|0⟩</small></span><div className="qc-wire"/>{Array.from({length:columns},(_,column) => <Button variant="ghost" key={column} className={`qc-cell ${tool ? 'can-place' : ''} ${pending.includes(q) ? 'is-pending' : ''}`} title={tool ? `Place ${tool} on q${q}, step ${column+1}` : `q${q}, step ${column+1}`} aria-label={`q${q}, step ${column+1}${tool ? `, place ${tool}` : ''}`} tabIndex={focusCell[0]===q && focusCell[1]===column ? 0 : -1} data-cell={`${q}-${column}`} onFocus={() => setFocusCell([q,column])} onKeyDown={e => gridKey(e,q,column)} style={{left:START+column*COL}} onClick={() => place(q,column)} />)}</div>)}{circuit.gates.map(g => { const qs=touched(g); const first=Math.min(...qs); const last=Math.max(...qs); const x=START+g.column*COL+COL/2; return <div key={g.id}>{first !== last && <div className="qc-connector" style={{left:x,top:44+first*ROW+ROW/2,height:(last-first)*ROW}}/>}{g.controls.map(q => <span key={q} className="qc-control-dot" style={{left:x,top:44+q*ROW+ROW/2}} />)}{g.targets.map(q => <Button key={q} variant="ghost" className={`qc-placed-gate qc-placed-${g.type.toLowerCase()} ${selected.includes(g.id) ? 'is-selected' : ''} ${g.classicalControl ? 'is-conditional' : ''}`} style={{left:x,top:44+q*ROW+ROW/2}} title={`${g.type}${g.params.theta !== undefined ? ` θ=${g.params.theta.toFixed(3)}` : ''}${g.classicalControl ? ` if ${describeCondition(g.classicalControl)}` : ''} — click to select, double-click to remove`} aria-pressed={selected.includes(g.id)} aria-label={`${GATE_LIBRARY.find(d => d.type === g.type)?.label ?? g.type} on q${q} at step ${g.column+1}${g.params.theta !== undefined ? `, angle ${g.params.theta.toFixed(3)}` : ''}${g.controls.length ? `, controlled by ${g.controls.map(c => `q${c}`).join(' and ')}` : ''}${g.classicalControl ? `, only if ${describeCondition(g.classicalControl)}` : ''}`} onClick={e => { e.stopPropagation(); setTool(null); setPending([]); setSelected(previous => e.shiftKey ? previous.includes(g.id) ? previous.filter(id => id !== g.id) : [...previous,g.id] : [g.id]); }} onDoubleClick={() => { commit({ ...circuit,gates:circuit.gates.filter(item => item.id !== g.id) }); setSelected([]); }}>{g.type === 'MEASURE' ? 'M' : g.type === 'SWAP' ? '×' : ['CNOT','CCX'].includes(g.type) ? '⊕' : g.type === 'CZ' ? 'Z' : g.type.startsWith('R') ? `R${g.type[1]?.toLowerCase()}` : g.type}</Button>)}</div>; })}</div></div>
      <div className="qc-canvas-footer"><span>{tool ? `Placing ${tool}${pending.length ? ` · ${pending.length} of ${GATE_LIBRARY.find(g => g.type === tool)?.qubits} qubits selected` : ''}` : selected.length ? `${selected.length} gate(s) selected · use ← → to move` : 'Select a gate from the library to place it on a wire'}</span><span>q₀ = least significant bit</span></div>{message && <div className="qc-alert" role="status">{message}<Button variant="ghost" size="icon" aria-label="Dismiss message" onClick={() => setMessage('')}><X/></Button></div>}
      </section><Results result={result} circuit={resultCircuit} running={running} tab={tab} setTab={setTab}/></div>
    </main>
    <Dialog open={Boolean(angleDialog)} onOpenChange={open => { if (!open) setAngleDialog(null); }}><DialogContent><DialogHeader><DialogTitle>Rotation angle</DialogTitle><DialogDescription>Enter θ in radians for {angleDialog?.type}.</DialogDescription></DialogHeader><Input aria-label="Angle in radians" type="number" step="any" value={angle} onChange={e => setAngle(e.target.value)}/><Button onClick={() => { if (!angleDialog) return; const value=Number(angle); if (!Number.isFinite(value)) { setMessage('Enter a finite rotation angle.'); return; } addGate(angleDialog.type,angleDialog.qubits,angleDialog.column,value); setAngleDialog(null); }}>Place gate</Button></DialogContent></Dialog>
    <Dialog open={Boolean(conditionDialog)} onOpenChange={open => { if (!open) setConditionDialog(null); }}><DialogContent><DialogHeader><DialogTitle>Classical condition</DialogTitle><DialogDescription>Apply this gate only when an earlier measurement has the selected value.</DialogDescription></DialogHeader>{(() => { const avail = [...new Set(circuit.gates.filter(g => g.type === 'MEASURE' && g.column < (conditionDialog?.gate.column ?? 0)).map(g => g.targets[0] as number))].sort((a,b) => a-b); const bits = conditionBits.filter(q => avail.includes(q)); const max = 2 ** Math.max(bits.length,1) - 1; return <><fieldset className="qc-dialog-row"><legend className="text-sm">Measured bits (first ticked = least significant)</legend>{avail.map(q => <label key={q} className="flex items-center gap-1"><input type="checkbox" checked={bits.includes(q)} onChange={e => setConditionBits(b => e.target.checked ? [...b.filter(x => avail.includes(x)),q] : b.filter(x => x !== q))}/> c{q}</label>)}</fieldset><div className="qc-dialog-row"><label htmlFor="qc-cond-value">Register equals (0–{max})</label><Input id="qc-cond-value" type="number" min={0} max={max} value={conditionReg} onChange={e => setConditionReg(e.target.value)}/></div>{bits.length > 0 && <p className="text-sm text-muted-foreground">Gate fires when {describeCondition({ qubits:bits, equals:Number(conditionReg) || 0 })}{bits.length > 1 ? ` (bit order ${[...bits].reverse().map(q => `c${q}`).join(' ')})` : ''}.</p>}<Button disabled={!bits.length} onClick={() => { if (!conditionDialog) return; const v = Number(conditionReg); if (!Number.isInteger(v) || v < 0 || v > max) { setMessage(`Condition value must be a whole number from 0 to ${max}.`); return; } const classicalControl = bits.length === 1 ? { qubit:bits[0] as number, equals:v } : { qubits:bits, equals:v }; commit({ ...circuit, gates:[...circuit.gates,{...conditionDialog.gate,classicalControl}] }); setConditionDialog(null); }}>Place conditional gate</Button></>; })()}</DialogContent></Dialog>
    <Dialog open={showImport} onOpenChange={setShowImport}><DialogContent><DialogHeader><DialogTitle>Import OpenQASM 2.0</DialogTitle><DialogDescription>Importing replaces the current circuit. You can undo the change afterward.</DialogDescription></DialogHeader><input ref={fileInput} type="file" accept=".qasm,.txt" className="sr-only" onChange={async e => { const file=e.target.files?.[0]; if(file) setImportText(await file.text()); }}/><Button variant="outline" onClick={() => fileInput.current?.click()}><FolderOpen/> Choose a file</Button><textarea className="qc-import-text" aria-label="OpenQASM source" value={importText} onChange={e => setImportText(e.target.value)} placeholder={'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];'}/><Button onClick={importQasm} disabled={!importText.trim()}>Import circuit</Button></DialogContent></Dialog>
    <Dialog open={showHelp} onOpenChange={setShowHelp}><DialogContent><DialogHeader><DialogTitle>QuFlow reference</DialogTitle><DialogDescription>Circuit editing and simulation controls.</DialogDescription></DialogHeader><div className="qc-help"><p>Choose a gate, then click a wire at the step where it should run. For multi-qubit gates, select the control wire first. Measurement records its result for later conditional operations.</p><p>Shift-click gates for multi-selection. Use arrow keys to move selected gates, Delete to remove, ⌘/Ctrl+C and V to copy and paste, ⌘/Ctrl+Z to undo, and ⌘/Ctrl+Enter to run. Escape clears the active tool or selection.</p><p>Amplitudes, Bloch vectors, and the Q-sphere show the ideal coherent state. For noise and classical feedback, shot counts reflect the actual sampled trajectories.</p></div></DialogContent></Dialog>
    <Dialog open={showAccount} onOpenChange={setShowAccount}><DialogContent><DialogHeader><DialogTitle>{userEmail ? 'Your QuFlow account' : authMode === 'signin' ? 'Sign in to QuFlow' : 'Create your QuFlow account'}</DialogTitle><DialogDescription>{userEmail ? 'Your saved circuits are private to this account.' : 'Build locally without an account, or sign in to save circuits privately.'}</DialogDescription></DialogHeader>{userEmail ? <><div className="qc-account-email">Signed in as <strong>{userEmail}</strong></div><Button variant="outline" onClick={signOut}><LogOut/> Sign out</Button></> : <><Button variant="outline" onClick={googleSignIn}>Continue with Google</Button><div className="qc-auth-divider"><span>or use email</span></div>{authMode === 'signup' && <Input aria-label="Username" type="text" autoComplete="username" value={authUsername} placeholder="Username" onChange={e => setAuthUsername(e.target.value)}/>}<Input aria-label="Email" type="email" autoComplete="email" value={authEmail} placeholder="researcher@example.com" onChange={e => setAuthEmail(e.target.value)}/><Input aria-label="Password" type="password" autoComplete={authMode === 'signin' ? 'current-password' : 'new-password'} value={authPassword} placeholder="At least 8 characters" onChange={e => setAuthPassword(e.target.value)}/><Button onClick={authenticate}>{authMode === 'signin' ? 'Sign in' : 'Create account'}</Button><Button variant="ghost" onClick={() => setAuthMode(mode => mode === 'signin' ? 'signup' : 'signin')}>{authMode === 'signin' ? 'Need an account? Create one' : 'Already have an account? Sign in'}</Button></>}</DialogContent></Dialog>
    {projectId && role === 'owner' && <ShareDialog projectId={projectId} projectName={projectName} ownerEmail={userEmail} open={showShare} onOpenChange={setShowShare}/>}
    <Dialog open={showProjects} onOpenChange={setShowProjects}><DialogContent><DialogHeader><DialogTitle>Saved circuits</DialogTitle><DialogDescription>Your circuits and circuits others have shared with you.</DialogDescription></DialogHeader><div className="qc-project-list">{projects.length ? projects.map(project => <div key={project.id} className="flex items-center gap-1"><Button variant="ghost" className="qc-project-row flex-1" onClick={() => loadProject(project)}><span>{project.name}</span><small>{project.owner_id !== userId ? 'Shared with you · ' : ''}{new Date(project.updated_at).toLocaleString()}</small></Button><Button variant="ghost" size="icon" aria-label={`Duplicate ${project.name}`} title="Duplicate" onClick={() => duplicateProject(project)}><Copy/></Button>{project.owner_id === userId && <Button variant="ghost" size="icon" aria-label={`Delete ${project.name}`} title="Delete" onClick={() => deleteProject(project.id,project.name)}><Trash2/></Button>}</div>) : <p>No saved circuits yet.</p>}</div></DialogContent></Dialog>
  </div>;
}

function Results({result,circuit,running,tab,setTab}:{result:RunResult|null;circuit:Circuit|null;running:boolean;tab:Tab;setTab:(value:Tab)=>void}) {
  const tabs: { id:Tab; label:string }[] = [{id:'probabilities',label:'Probabilities'},{id:'counts',label:'Shot counts'},{id:'amplitudes',label:'Amplitudes'},{id:'bloch',label:'Bloch vectors'},{id:'qsphere',label:'Q-sphere'}];
  const entries = result?.statevector.filter((state) => state.probability > 0) ?? [];
  return <section className="qc-results"><div className="qc-results-head"><div><span className="qc-eyebrow">ANALYSIS</span><h2>Results <span>{result ? `· ${result.shots.toLocaleString()} shots` : ''}</span></h2></div>{result && circuit && <div className="qc-result-actions"><Button variant="outline" size="sm" title="Download CSV" onClick={() => downloadResults(result,circuit,tab === 'counts' ? 'counts' : 'amplitudes')}><Download/> CSV</Button><Button variant="outline" size="sm" title="Download full result as JSON" onClick={() => downloadResults(result,circuit,'json')}><Braces/> JSON</Button></div>}</div>
    {!result ? <div className="qc-empty-results"><div className="qc-empty-graphic"><Activity size={30}/></div><strong>{running ? 'Simulation in progress…' : 'No results yet'}</strong><span>{running ? 'Computing your circuit locally.' : 'Run your circuit to inspect its quantum state.'}</span></div> : <><div className="qc-result-meta"><span><i className="qc-active-dot"/> Local simulator</span><span>{result.numQubits} qubits</span><span>{result.executionTimeMs} ms</span>{result.noise && <span>{result.noise.model.replaceAll('_',' ')} · p={result.noise.p}</span>}{result.seed !== null && <span>Seed {result.seed}</span>}{result.noiseShotsCapped && <span>Trajectory limit: {result.shots.toLocaleString()} shots</span>}{running && <span>Updating…</span>}</div><div className="qc-tabs" role="tablist" aria-label="Result views">{tabs.map(item => <Button variant="ghost" role="tab" aria-selected={tab === item.id} key={item.id} className={tab === item.id ? 'is-current' : ''} onClick={() => setTab(item.id)}>{item.label}</Button>)}</div>{(result.hasClassicalControl || result.midCircuitMeasurement || result.noise) && tab !== 'counts' && <p className="qc-result-note">Ideal coherent state shown here; shot counts include measurement collapse, conditions and noise.</p>}
      {tab === 'probabilities' && <div className="qc-bars">{entries.length ? entries.map(s => <div className="qc-bar-row" key={s.label}><span className="qc-basis">|{s.label}⟩</span><div className="qc-bar-track"><span style={{width:`${Math.max(0,s.probability*100)}%`}}/></div><span className="qc-bar-value">{(s.probability*100).toFixed(2)}%</span></div>) : <p>No nonzero states.</p>}</div>}
      {tab === 'counts' && <div className="qc-bars">{Object.entries(result.counts).sort((a,b) => b[1]-a[1]).map(([state,count]) => <div className="qc-bar-row" key={state}><span className="qc-basis">|{state}⟩</span><div className="qc-bar-track qc-count-track"><span style={{width:`${count/result.shots*100}%`}}/></div><span className="qc-bar-value">{count.toLocaleString()}</span></div>)}</div>}
      {tab === 'amplitudes' && <div className="qc-table-wrap"><table><thead><tr><th>STATE</th><th>REAL</th><th>IMAGINARY</th><th>PROBABILITY</th><th>PHASE (RAD)</th></tr></thead><tbody>{result.statevector.map(s => <tr key={s.label}><td className="qc-basis">|{s.label}⟩</td><td>{s.amplitude.re.toFixed(6)}</td><td>{s.amplitude.im.toFixed(6)}</td><td>{(s.probability*100).toFixed(3)}%</td><td>{s.phase.toFixed(4)}</td></tr>)}</tbody></table></div>}
      {tab === 'bloch' && <div className="qc-bloch-list">{result.blochVectors.map(v => {const cx=60, cy=60, x=cx+43*(v.x*.95-v.y*.35), y=cy-43*(v.z*.95+v.y*.22); return <div className="qc-bloch-item" key={v.qubit}><svg viewBox="0 0 120 120" role="img" aria-label={`Bloch vector for qubit ${v.qubit}: x ${v.x}, y ${v.y}, z ${v.z}`}><circle cx="60" cy="60" r="43" className="qc-bloch-circle"/><ellipse cx="60" cy="60" rx="43" ry="14" className="qc-bloch-circle"/><line x1="60" y1="17" x2="60" y2="103" className="qc-bloch-axis"/><line x1="60" y1="60" x2={x} y2={y} className="qc-bloch-vector"/><circle cx={x} cy={y} r="4" className="qc-bloch-point"/></svg><div><strong>q{v.qubit}</strong><span>x {v.x.toFixed(3)} · y {v.y.toFixed(3)} · z {v.z.toFixed(3)}</span></div></div>; })}</div>}
      {tab === 'qsphere' && <div className="qc-qsphere"><svg viewBox="0 0 300 300" role="img" aria-label="Q-sphere state representation"><circle cx="150" cy="150" r="110" className="qc-sphere-outline"/><ellipse cx="150" cy="150" rx="110" ry="34" className="qc-sphere-outline"/>{entries.slice(0,48).map((s) => {const w=s.label.split('1').length-1; const theta=Math.PI*w/result.numQubits; const ring=entries.filter(e => e.label.split('1').length-1 === w); const position=ring.findIndex(e => e.index === s.index); const phi=2*Math.PI*position/Math.max(1,ring.length);const x=150+110*Math.sin(theta)*(Math.cos(phi)*.95-Math.sin(phi)*.35);const y=150-110*(Math.cos(theta)*.95+Math.sin(theta)*Math.sin(phi)*.22);return <g key={s.label}><line x1="150" y1="150" x2={x} y2={y} className="qc-sphere-spoke"/><circle cx={x} cy={y} r={3+17*Math.sqrt(s.probability)} className="qc-sphere-point" opacity={.55+.45*Math.cos(s.phase)}><title>{`|${s.label}⟩ · ${(s.probability*100).toFixed(2)}% · phase ${s.phase.toFixed(3)} rad`}</title></circle></g>;})}</svg><div className="qc-sphere-key">Marker size represents probability · opacity reflects phase · {Math.min(entries.length,48)} states shown</div></div>}{result.statevectorTruncated && <p className="qc-result-note">Showing the top {result.statevector.length} of {result.totalNonzeroStates} nonzero amplitudes.</p>}</>}
  </section>;
}
