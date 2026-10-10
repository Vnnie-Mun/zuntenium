import type { Circuit, RunResult } from './model';
export function download(filename: string, content: string, type = 'text/plain') {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement('a');
  link.href = url; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function csv(rows: (string | number)[][]) { return rows.map(row => row.map(value => `"${String(value).replaceAll('"','""')}"`).join(',')).join('\r\n') + '\r\n'; }
export function downloadResults(result: RunResult, circuit: Circuit, kind: 'amplitudes' | 'counts' | 'json') {
  if (kind === 'json') return download('qcanvas-results.json', JSON.stringify({ circuit, result, exportedAt: new Date().toISOString() }, null, 2), 'application/json');
  if (kind === 'counts') return download('qcanvas-counts.csv', csv([['state', 'count', 'probability'], ...Object.entries(result.counts).sort().map(([state, count]) => [state, count, count / result.shots])] ), 'text/csv');
  return download('qcanvas-amplitudes.csv', csv([['state', 'real', 'imaginary', 'magnitude', 'probability', 'phase'], ...result.statevector.map(s => [s.label, s.amplitude.re, s.amplitude.im, Math.hypot(s.amplitude.re, s.amplitude.im), s.probability, s.phase])]), 'text/csv');
}
