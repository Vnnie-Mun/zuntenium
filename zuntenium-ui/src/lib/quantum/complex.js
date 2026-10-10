// Minimal complex-number helpers. Represented as { re, im } objects —
// clear and easy to verify against textbook gate definitions, which matters
// more here than raw speed for the qubit counts a browser tool will see.

export const C = (re, im = 0) => ({ re, im });

export const cAdd = (a, b) => C(a.re + b.re, a.im + b.im);
export const cSub = (a, b) => C(a.re - b.re, a.im - b.im);
export const cMul = (a, b) => C(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
export const cScale = (a, k) => C(a.re * k, a.im * k);
export const cAbs2 = (a) => a.re * a.re + a.im * a.im;

// e^{i*theta}
export const cExpI = (theta) => C(Math.cos(theta), Math.sin(theta));
