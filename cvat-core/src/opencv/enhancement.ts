// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

/* eslint-disable no-param-reassign */ // the methods edit the pixel buffer in place: frames are ~100 MB each

import { BaseImageFilter, SerializedImageFilter } from './image-processing';

// Ports of RedScrap's image enhancement (src/redscrap/image_enhancement/enhancement/assets), same names as the
// enhanced/<method>/ folders of a bag. RedScrap's scripts/check_cvat_enhancement.sh compares them with the
// production code on a real frame.
export type EnhancementMethod = 'clahe_luminance' | 'clahe_channels' | 'he_luminance' | 'he_channels' |
'rek' | 'riesz' | 'riesz_luminance_only' | 'super_color' | 'super_luminance';

export interface EnhancementOptions {
    method: EnhancementMethod;
    params: Record<string, number>;
}

type Method = (cv: any, px: Uint8ClampedArray, width: number, height: number, params: Record<string, number>) => void;

// Runs fn on an 8-bit Mat of the RGB pixels and writes its RGB result back into px (alpha untouched).
function onRgb(
    cv: any, px: Uint8ClampedArray, width: number, height: number,
    fn: (rgb: any, keep: (m: any) => any) => any,
): void {
    const mats: any[] = [];
    const keep = (m: any): any => { mats.push(m); return m; };
    try {
        const rgba = keep(cv.matFromImageData({ data: px, width, height }));
        const rgb = keep(new cv.Mat());
        cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB, 0);
        const out = fn(rgb, keep).data;
        for (let i = 0, j = 0; i < px.length; i += 4, j += 3) {
            px[i] = out[j]; px[i + 1] = out[j + 1]; px[i + 2] = out[j + 2];
        }
    } finally {
        mats.forEach((m) => m.delete());
    }
}

// op on the L channel of Lab (histeq_capi.cpp HeLuminance/ClaheLuminance) or on each channel (HeChannels/ClaheChannels)
function luminance(cv: any, op: (src: any, dst: any) => void): Method {
    return (_cv, px, width, height) => onRgb(cv, px, width, height, (rgb, keep) => {
        const lab = keep(new cv.Mat());
        const planes = keep(new cv.MatVector());
        cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab, 0);
        cv.split(lab, planes);
        const L = keep(planes.get(0));
        const equalized = keep(new cv.Mat());
        op(L, equalized);
        planes.set(0, equalized);
        cv.merge(planes, lab);
        const out = keep(new cv.Mat());
        cv.cvtColor(lab, out, cv.COLOR_Lab2RGB, 0);
        return out;
    });
}

function channels(cv: any, op: (src: any, dst: any) => void): Method {
    return (_cv, px, width, height) => onRgb(cv, px, width, height, (rgb, keep) => {
        const planes = keep(new cv.MatVector());
        cv.split(rgb, planes);
        for (let c = 0; c < 3; c++) {
            const plane = keep(planes.get(c));
            const equalized = keep(new cv.Mat());
            op(plane, equalized);
            planes.set(c, equalized);
        }
        const out = keep(new cv.Mat());
        cv.merge(planes, out);
        return out;
    });
}

function clahe(cv: any, clipLimit: number): (src: any, dst: any) => void {
    return (src, dst) => {
        const c = new cv.CLAHE(clipLimit, new cv.Size(8, 8)); // createCLAHE() defaults: 8x8 tiles
        try { c.apply(src, dst); } finally { c.delete(); }
    };
}

// rek_capi.cpp: backlight compensation, relit image blended in where the luminance is dark
const rek: Method = (_cv, px, width, height, { p, alpha: alphaParam }) => {
    const n = width * height;
    const lum = new Uint8Array(n);
    let mm = 255;
    let MM = 0;
    for (let i = 0, k = 0; i < n; i++, k += 4) {
        lum[i] = Math.trunc((px[k] + px[k + 1] + px[k + 2]) / 3);
        mm = Math.min(mm, lum[i]);
        MM = Math.max(MM, lum[i]);
    }
    if (MM === mm) return; // flat image: the C++ divides by zero here

    let alpha = alphaParam;
    // auto: from the mean and spread of the light and dark regions (ThresholdImage2 + MeanOnDLRegionsPixels)
    if (alpha < 0) {
        const thresh = (MM - mm) * 0.5 + mm;
        const light = new Uint8Array(n);
        let pl = 0; let pd = 0; let LL = 0; let DD = 0;
        for (let i = 0, k = 0; i < n; i++, k += 4) {
            light[i] = (px[k] + px[k + 1] + px[k + 2]) / 3 > thresh ? 1 : 0;
            if (light[i]) { LL += lum[i] / 255; pl++; } else { DD += lum[i] / 255; pd++; }
        }
        LL *= 255 / pl;
        DD *= 255 / pd;
        let dl = 0;
        for (let i = 0; i < n; i++) if (light[i]) dl += (lum[i] - LL) * (lum[i] - LL);
        alpha = (LL - Math.sqrt(dl / pl)) / DD;
    }

    const weight = new Float64Array(256); // GetWeights depends on the 8-bit luminance only
    for (let v = 0; v < 256; v++) weight[v] = (1 - (v - mm) / (MM - mm)) ** p;
    for (let i = 0, k = 0; i < n; i++, k += 4) {
        const w = weight[lum[i]];
        for (let c = k; c < k + 3; c++) {
            let relit = px[c] * alpha;
            if (relit > 256) relit = 255;
            px[c] = Math.trunc((1 - w) * px[c] + w * Math.trunc(relit));
        }
    }
};

// riesz_lib.py riesz_fractional_homomorphic_enhancement: log, FFT, radial gain gainLow..gainHigh ~ r^alpha, inverse,
// exp. The filter is built directly in unshifted FFT order, which equals fftshift -> filter -> ifftshift.
function riesz(
    cv: any, y: Float64Array, width: number, height: number, { alpha, gainHigh, gainLow }: Record<string, number>,
): Uint8Array {
    const logY = new cv.Mat(height, width, cv.CV_32F);
    const spectrum = new cv.Mat();
    const back = new cv.Mat();
    try {
        const l = logY.data32F;
        for (let i = 0; i < y.length; i++) l[i] = Math.log(Math.fround(y[i] / 255) + 1e-5);
        cv.dft(logY, spectrum, cv.DFT_COMPLEX_OUTPUT);
        const crow = Math.floor(height / 2);
        const ccol = Math.floor(width / 2);
        const maxRadius = Math.sqrt(crow * crow + ccol * ccol) || 1;
        const s = spectrum.data32F;
        for (let r = 0, k = 0; r < height; r++) {
            const u = r < height - crow ? r : r - height;
            for (let c = 0; c < width; c++, k += 2) {
                const v = c < width - ccol ? c : c - width;
                const gain = gainLow + (gainHigh - gainLow) * (Math.sqrt(u * u + v * v) / maxRadius) ** alpha;
                s[k] *= gain;
                s[k + 1] *= gain;
            }
        }
        // eslint-disable-next-line no-bitwise
        cv.dft(spectrum, back, cv.DFT_INVERSE | cv.DFT_SCALE | cv.DFT_REAL_OUTPUT);
        const b = back.data32F;
        const out = new Uint8Array(y.length);
        for (let i = 0; i < y.length; i++) out[i] = Math.trunc(Math.min(1, Math.max(0, Math.exp(b[i]) - 1e-5)) * 255);
        return out;
    } finally {
        logY.delete(); spectrum.delete(); back.delete();
    }
}

// riesz_lib.py apply_to_color: Riesz on the channel mean, chroma ratios kept
const rieszColor: Method = (cv, px, width, height, params) => {
    const n = width * height;
    const lum = new Float64Array(n);
    for (let i = 0, k = 0; i < n; i++, k += 4) lum[i] = (px[k] + px[k + 1] + px[k + 2]) / 3;
    const enhanced = riesz(cv, lum, width, height, params);
    for (let i = 0, k = 0; i < n; i++, k += 4) {
        if (lum[i] > 0) {
            for (let c = k; c < k + 3; c++) px[c] = Math.trunc(Math.min(255, (px[c] / lum[i]) * enhanced[i]));
        }
    }
};

// compute_enhancements.py riesz_luminance_only: Riesz on the OpenCV grayscale, shown as gray
const rieszGray: Method = (cv, px, width, height, params) => {
    const rgba = cv.matFromImageData({ data: px, width, height });
    const gray = new cv.Mat();
    try {
        cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY, 0);
        const enhanced = riesz(cv, Float64Array.from(gray.data), width, height, params);
        for (let i = 0, k = 0; i < enhanced.length; i++, k += 4) {
            px[k] = enhanced[i]; px[k + 1] = enhanced[i]; px[k + 2] = enhanced[i];
        }
    } finally {
        rgba.delete(); gray.delete();
    }
};

// SuPeR (Milano Retinex), super_capi.cpp / super_luminance_capi.cpp, bit-identical to production. The C++ loops, for
// every pixel, over all brighter levels or grid cells (~20 s per 24 MP frame in JS). The summed weight is linear in
// the pixel position, so running sums over the levels / the sorted cells give the same sums in O(1) per pixel, up to
// float rounding (~1e-12 here). That only changes the truncated output when the value is within rounding of an
// integer: those pixels (a few %) are recomputed with the C++ loop, same operations in the same order.
const NEAR_INTEGER = 1e-6;
const nearInteger = (value: number): boolean => Math.abs(value - Math.round(value)) < NEAR_INTEGER;

// super_capi.cpp Super::compute: per channel, grid of block maxima; out = v * sum(w_k / k) / sum(w_k) over the levels
// k > v, w_k = sum over the blocks with maximum k of (K - squared distance to the block centre),
// K = 1.000000001 * diag^2
const superColor: Method = (_cv, px, width, height, { grid }) => {
    const M = Math.round(grid);
    const dR = Math.ceil(height / M);
    const dC = Math.ceil(width / M);
    const K = (1.0 + 0.000000001) * (height * height + width * width);
    for (let ch = 0; ch < 3; ch++) {
        const bins = new Uint8Array(M * M);
        for (let r = 0; r < height; r++) {
            const R = Math.floor(r / dR) * M;
            for (let c = 0, k = r * width * 4 + ch; c < width; c++, k += 4) {
                const b = R + Math.floor(c / dC);
                if (px[k] > bins[b]) bins[b] = px[k];
            }
        }
        // fillAux: per level, count, sum x, sum x^2, sum y, sum y^2 of the block centres
        const aux = Array.from({ length: 5 }, () => new Float64Array(256));
        for (let i = 0; i < M; i++) {
            for (let j = 0; j < M; j++) {
                const v = bins[i * M + j];
                const x = dR * (i + 0.5);
                const y = dC * (j + 0.5);
                aux[0][v]++; aux[1][v] += x; aux[2][v] += x * x; aux[3][v] += y; aux[4][v] += y * y;
            }
        }
        // the same per level divided by the level, then both summed from the top: index v holds the totals over the
        // levels above v
        const S = aux.map((a) => Float64Array.from({ length: 257 }, (_, v) => (v < 256 ? a[v] : 0)));
        const T = aux.map((a) => Float64Array.from({ length: 257 }, (_, v) => (v > 0 && v < 256 ? a[v] / v : 0)));
        for (let a = 0; a < 5; a++) {
            for (let v = 255; v >= 0; v--) { S[a][v] += S[a][v + 1]; T[a][v] += T[a][v + 1]; }
        }
        for (let r = 0; r < height; r++) {
            for (let c = 0, k = r * width * 4 + ch; c < width; c++, k += 4) {
                const p = px[k];
                const v = p + 1;
                if (!S[0][v]) { px[k] = 255; continue; } // no level above: den = 0, res = 1
                const A = K - r * r - c * c;
                const den = A * S[0][v] + 2 * r * S[1][v] - S[2][v] + 2 * c * S[3][v] - S[4][v];
                const num = A * T[0][v] + 2 * r * T[1][v] - T[2][v] + 2 * c * T[3][v] - T[4][v];
                let out = 255 * ((p * num) / den);
                if (nearInteger(out)) {
                    let n = 0.0;
                    let d = 0.0;
                    for (let l = v; l < 256; l++) {
                        if (aux[0][l] > 0) {
                            let dist = aux[0][l] * r * r - 2 * r * aux[1][l] + aux[2][l];
                            dist += aux[0][l] * c * c - 2 * c * aux[3][l] + aux[4][l];
                            const Sdelta = aux[0][l] * K - dist;
                            n += Sdelta / l;
                            d += Sdelta;
                        }
                    }
                    out = 255 * (d > 0.0 ? (p * n) / d : 1.0);
                }
                px[k] = Math.trunc(out);
            }
        }
    }
};

// super_luminance_capi.cpp: SuPeR on the channel mean with gridSize x gridSize overlapping windows, sorted by maximum;
// each pixel sums over the windows whose maximum is >= its value, weight 1 - squared distance / diag^2
interface Cell { row: number, col: number, intensity: number }

// the C++ sorts the cells with a std::priority_queue: cells with the same maximum come out in the order of libstdc++'s
// heap, which sets the order of the sums. These are its push_heap / pop_heap.
function sortLikePriorityQueue(cells: Cell[]): Cell[] {
    const heap: Cell[] = [];
    const less = (a: Cell, b: Cell): boolean => a.intensity < b.intensity;
    const pushHeap = (start: number, value: Cell): void => {
        let hole = start;
        let parent = Math.trunc((hole - 1) / 2);
        while (hole > 0 && less(heap[parent], value)) {
            heap[hole] = heap[parent];
            hole = parent;
            parent = Math.trunc((hole - 1) / 2);
        }
        heap[hole] = value;
    };
    const adjustHeap = (len: number, value: Cell): void => {
        let hole = 0;
        let child = 0;
        while (child < Math.trunc((len - 1) / 2)) {
            child = 2 * (child + 1);
            if (less(heap[child], heap[child - 1])) child--;
            heap[hole] = heap[child];
            hole = child;
        }
        if ((len & 1) === 0 && child === Math.trunc((len - 2) / 2)) {
            child = 2 * (child + 1);
            heap[hole] = heap[child - 1];
            hole = child - 1;
        }
        pushHeap(hole, value);
    };
    for (const cell of cells) {
        heap.push(cell);
        pushHeap(heap.length - 1, cell);
    }
    const sorted: Cell[] = [];
    while (heap.length) {
        sorted.push(heap[0]);
        const last = heap.length - 1;
        if (last > 0) {
            const value = heap[last];
            heap[last] = heap[0];
            adjustHeap(last, value);
        }
        heap.pop();
    }
    return sorted;
}

const superLuminance: Method = (_cv, px, width, height, { grid }) => {
    const n = width * height;
    const sum3 = new Uint16Array(n); // GetLuminance: channel sum (exact), / 3
    for (let i = 0, k = 0; i < n; i++, k += 4) sum3[i] = px[k] + px[k + 1] + px[k + 2];

    // GetSquareGridMax
    const steps = Math.round(grid);
    const sr = height / steps;
    const sc = width / steps;
    const cells: Cell[] = [];
    for (let i = 0; i < steps; i++) {
        const r0 = Math.trunc(i * sr);
        const r1 = Math.min(r0 + Math.trunc(sr), height - 1);
        for (let j = 0; j < steps; j++) {
            const c0 = Math.trunc(j * sc);
            const c1 = Math.min(c0 + Math.trunc(sc), width - 1);
            let max3 = 0;
            for (let r = r0; r <= r1; r++) {
                for (let c = c0; c <= c1; c++) max3 = Math.max(max3, sum3[r * width + c]);
            }
            let br = 0; // integer sums, exact in any order
            let bc = 0;
            for (let r = r0; r <= r1; r++) br += r * (c1 - c0 + 1);
            for (let c = c0; c <= c1; c++) bc += c * (r1 - r0 + 1);
            const p = (r1 - r0 + 1) * (c1 - c0 + 1);
            cells.push({ row: br / p, col: bc / p, intensity: max3 / 3 });
        }
    }
    const sorted = sortLikePriorityQueue(cells);
    let nc = 0; // cells with maximum 0 never count, and sit at the end
    while (nc < sorted.length && sorted[nc].intensity > 0) nc++;

    // prefix sums over the sorted cells of 1, x, x^2 + y^2, y and the same divided by the maximum
    const P = Array.from({ length: 4 }, () => new Float64Array(nc + 1));
    const Q = Array.from({ length: 4 }, () => new Float64Array(nc + 1));
    for (let g = 0; g < nc; g++) {
        const { row: x, col: y, intensity: m } = sorted[g];
        [1, x, x * x + y * y, y].forEach((term, a) => {
            P[a][g + 1] = P[a][g] + term;
            Q[a][g + 1] = Q[a][g] + term / m;
        });
    }
    // how many cells have a maximum >= the value, for every channel sum (a 0 pixel counts as 1e-10)
    const upTo = new Int32Array(766);
    for (let t = 0, L = nc; t < 766; t++) {
        const val = Math.max(t / 3, 0.0000000001);
        while (L > 0 && sorted[L - 1].intensity < val) L--;
        upTo[t] = L;
    }

    // SuPeR, then the recombination into color
    const D = height * height + width * width;
    const exact = (r: number, c: number, val: number): number => {
        let w = 0;
        let ret = 0;
        for (let g = 0; g < nc && val <= sorted[g].intensity; g++) {
            const x = r - sorted[g].row;
            const y = c - sorted[g].col;
            let d = x * x + y * y;
            d = 1 - d / D;
            ret += (d * val) / sorted[g].intensity;
            w += d;
        }
        return w > 0 ? ret / w : 0;
    };
    for (let r = 0, i = 0, k = 0; r < height; r++) {
        for (let c = 0; c < width; c++, i++, k += 4) {
            const z = sum3[i] / 3;
            const L = upTo[sum3[i]];
            if (z === 0 || L === 0) { px[k] = 0; px[k + 1] = 0; px[k + 2] = 0; continue; }
            const val = z;
            const q = r * r + c * c;
            const w = P[0][L] - (q * P[0][L] - 2 * r * P[1][L] + P[2][L] - 2 * c * P[3][L]) / D;
            const ret = val * (Q[0][L] - (q * Q[0][L] - 2 * r * Q[1][L] + Q[2][L] - 2 * c * Q[3][L]) / D);
            let u = (255 * (ret / w)) / z;
            if (nearInteger(px[k] * u) || nearInteger(px[k + 1] * u) || nearInteger(px[k + 2] * u)) {
                u = (255 * exact(r, c, val)) / z;
            }
            for (let ch = k; ch < k + 3; ch++) px[ch] = Math.trunc(Math.min(255, px[ch] * u));
        }
    }
};

export default class EnhancementImplementation extends BaseImageFilter {
    private cv: any;
    public method: EnhancementMethod;
    public params: Record<string, number>;

    constructor(cv: any, options: EnhancementOptions) {
        super();
        this.cv = cv;
        this.configure(options);
    }

    public configure(options: object): void {
        ({ method: this.method, params: this.params } = options as EnhancementOptions);
    }

    public toJSON(): SerializedImageFilter {
        return { alias: 'opencv.enhancement', params: { method: this.method, params: this.params } };
    }

    public processImage(src: ImageData, frameNumber: number): ImageData {
        const { cv, params } = this;
        const methods: Record<EnhancementMethod, Method> = {
            clahe_luminance: luminance(cv, clahe(cv, params.clipLimit)),
            clahe_channels: channels(cv, clahe(cv, params.clipLimit)),
            he_luminance: luminance(cv, (s, d) => cv.equalizeHist(s, d)),
            he_channels: channels(cv, (s, d) => cv.equalizeHist(s, d)),
            rek,
            riesz: rieszColor,
            riesz_luminance_only: rieszGray,
            super_color: superColor,
            super_luminance: superLuminance,
        };
        const px = new Uint8ClampedArray(src.data);
        methods[this.method](cv, px, src.width, src.height, params);
        this.currentProcessedImage = frameNumber;
        return new ImageData(px, src.width, src.height);
    }
}
