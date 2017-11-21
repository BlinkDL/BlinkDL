// BlinkDL: a minimalist deep learning library (proof-of-concept and buggy at the moment)
// License: Apache-2.0
// By Bo Peng @ https://github.com/BlinkDL/BlinkDL

"use strict";

var gParam; // net parameters
const gHeap = new ArrayBuffer(8 * 1024 * 1024); // FIXME: needs a memory manager
const gBackend = Backend({
    Float32Array: Float32Array,
    Math: Math
}, null, gHeap);
const gBatchNormEPS = 0.001;
const gFloatSize = 4; // float = 4 bytes

function Backend(stdlib, foreign, heap) {
    'use asm';

    const imul = stdlib.Math.imul,
        fround = stdlib.Math.fround,
        buf = new stdlib.Float32Array(heap);

    function sgemm_trans(m, n, k, a, b, c) { // B is transposed (for speedup)
        m = m | 0;
        n = n | 0;
        k = k | 0;
        a = a | 0;
        b = b | 0;
        c = c | 0;

        var i = 0,
            j = 0,
            x = 0,
            xfinal = 0,
            y = 0,
            value = fround(0.0);

        for (;
            (i | 0) < (m | 0); i = i + 1 | 0) {
            for (j = 0;
                (j | 0) < (n | 0); j = j + 1 | 0) {
                x = a + ((imul(i, k) | 0) << 2) | 0;
                xfinal = x + (k << 2) | 0;
                y = b + ((imul(j, k) | 0) << 2) | 0;

                value = fround(0.0);

                for (;
                    (x | 0) < (xfinal - 12 | 0); x = x + 16 | 0, y = y + 16 | 0) {
                    value = fround(value + fround(buf[x >> 2] * buf[y >> 2]));
                    value = fround(value + fround(buf[x + 4 >> 2] * buf[y + 4 >> 2]));
                    value = fround(value + fround(buf[x + 8 >> 2] * buf[y + 8 >> 2]));
                    value = fround(value + fround(buf[x + 12 >> 2] * buf[y + 12 >> 2]));
                }

                for (;
                    (x | 0) < (xfinal | 0); x = x + 4 | 0, y = y + 4 | 0) {
                    value = fround(value + fround(buf[x >> 2] * buf[y >> 2]));
                }

                buf[c >> 2] = value;
                c = c + 4 | 0;
            }
        }
    }

    return {
        sgemm_trans: sgemm_trans,
    };
}

function im2col3x3pad1(from, to, channel, sz, transpose = false) {

    var desc = 0;
    var src = 0;
    const sz2 = sz * sz;

    if (!transpose) {
        for (var i = 0; i < channel; i++) {
            for (var j = sz + 1; j < sz2; j++) {
                if (j % sz != 0)
                    to[desc + j] = from[src + j - sz - 1]
            }
            desc += sz2;
            for (var j = sz; j < sz2; j++) {
                to[desc + j] = from[src + j - sz]
            }
            desc += sz2;
            for (var j = sz; j < sz2 - 1; j++) {
                if (j % sz != (sz - 1))
                    to[desc + j] = from[src + j - sz + 1]
            }
            desc += sz2;
            for (var j = 1; j < sz2; j++) {
                if (j % sz != 0)
                    to[desc + j] = from[src + j - 1]
            }
            desc += sz2;
            for (var j = 0; j < sz2; j++) {
                to[desc + j] = from[src + j]
            }
            desc += sz2;
            for (var j = 0; j < sz2 - 1; j++) {
                if (j % sz != (sz - 1))
                    to[desc + j] = from[src + j + 1]
            }
            desc += sz2;
            for (var j = 1; j < sz2 - sz; j++) {
                if (j % sz != 0)
                    to[desc + j] = from[src + j + sz - 1]
            }
            desc += sz2;
            for (var j = 0; j < sz2 - sz; j++) {
                to[desc + j] = from[src + j + sz]
            }
            desc += sz2;
            for (var j = 0; j < sz2 - sz - 1; j++) {
                if (j % sz != (sz - 1))
                    to[desc + j] = from[src + j + sz + 1]
            }
            desc += sz2;
            src += sz2;
        }
    } else {
        for (var i = 0; i < channel; i++) {
            for (var j = sz + 1; j < sz2; j++) {
                if (j % sz != 0)
                    to[desc + j * channel * 9] = from[src + j - sz - 1]
            }
            desc++;
            for (var j = sz; j < sz2; j++) {
                to[desc + j * channel * 9] = from[src + j - sz]
            }
            desc++;
            for (var j = sz; j < sz2 - 1; j++) {
                if (j % sz != (sz - 1))
                    to[desc + j * channel * 9] = from[src + j - sz + 1]
            }
            desc++;
            for (var j = 1; j < sz2; j++) {
                if (j % sz != 0)
                    to[desc + j * channel * 9] = from[src + j - 1]
            }
            desc++;
            for (var j = 0; j < sz2; j++) {
                to[desc + j * channel * 9] = from[src + j]
            }
            desc++;
            for (var j = 0; j < sz2 - 1; j++) {
                if (j % sz != (sz - 1))
                    to[desc + j * channel * 9] = from[src + j + 1]
            }
            desc++;
            for (var j = 1; j < sz2 - sz; j++) {
                if (j % sz != 0)
                    to[desc + j * channel * 9] = from[src + j + sz - 1]
            }
            desc++;
            for (var j = 0; j < sz2 - sz; j++) {
                to[desc + j * channel * 9] = from[src + j + sz]
            }
            desc++;
            for (var j = 0; j < sz2 - sz - 1; j++) {
                if (j % sz != (sz - 1))
                    to[desc + j * channel * 9] = from[src + j + sz + 1]
            }
            desc++;
            src += sz2;
        }
    }
}

function BlinkArray() {}

BlinkArray.prototype.Init = function (backend) {
    this.nChannel = 0;
    this.data = null;
    this.backend = backend;

    this.paramIndex = 0;
    (new Float32Array(gHeap)).fill(0); // clear global heap
}

BlinkArray.prototype.Convolution = function (outChannel, kerSize) {

    const m = outChannel;
    const n = this.data.length / this.nChannel;
    const k = kerSize * kerSize * this.nChannel;
    const a = new Float32Array(gHeap, 0, m * k)
    const b = new Float32Array(gHeap, gFloatSize * (m * k), k * n)

    // FIXME: BUGGY WHEN MEMORIES OVERLAP
    // FIXME: can only do 3x3 pad 1 or 1x1
    if (kerSize == 3) {
        b.fill(0);
        im2col3x3pad1(this.data, b, this.nChannel, Math.sqrt(n), this.backend != 'weblas');
    } else if (kerSize == 1) {
        if (this.backend == 'weblas') {
            for (var i = 0; i < k * n; i++) {
                b[i] = this.data[i];
            }
        } else { // transpose
            for (var i = 0; i < n; i++) {
                for (var j = 0; j < k; j++) {
                    b[i * k + j] = this.data[j * n + i];
                }
            }
        }
    }

    var out = new Float32Array(gHeap, 6 * 1024 * 1024, m * n); // HACK TO AVOID MEMORY OVERLAP

    // get weights
    for (var i = 0; i < m * k; i++) {
        a[i] = gParam[this.paramIndex + i];
    }
    this.paramIndex += m * k;

    // convolution
    if (this.backend == 'asm.js')
        gBackend.sgemm_trans(m, n, k, a.byteOffset, b.byteOffset, out.byteOffset);
    else if (this.backend == 'weblas')
        out = weblas.sgemm(m, n, k, 1.0, a, b, 0.0, null);

    // bias
    for (var i = 0; i < m; i++) {
        const bias = gParam[this.paramIndex + i];
        const target = i * n;
        for (var j = 0; j < n; j++) {
            out[target + j] += bias;
        }
    }

    this.paramIndex += m; // FIXME: should check parameter range
    this.data = out;
    this.nChannel = m;

    return this;
}

BlinkArray.prototype.BatchNorm = function () {
    const NN = this.data.length / this.nChannel;
    for (var i = 0; i < this.nChannel; i++) {
        const mean = gParam[this.paramIndex + this.nChannel * 0 + i];
        const variance = gParam[this.paramIndex + this.nChannel * 1 + i];
        const beta = gParam[this.paramIndex + this.nChannel * 2 + i];
        const gamma = gParam[this.paramIndex + this.nChannel * 3 + i];
        const mul = gamma / Math.sqrt(variance + gBatchNormEPS);
        const add = -gamma * mean / Math.sqrt(variance + gBatchNormEPS) + beta;

        const target = i * NN;
        for (var j = 0; j < NN; j++) {
            this.data[target + j] = mul * this.data[target + j] + add;
        }
    }
    this.paramIndex += 4 * this.nChannel;
    return this;
}

BlinkArray.prototype.ReLU = function () {
    for (var i = 0; i < this.data.length; i++) {
        if (this.data[i] < 0) {
            this.data[i] = 0;
        }
    }
    return this;
}

BlinkArray.prototype.Softmax = function () {
    var sum = 0.0;
    for (var i = 0; i < this.data.length; i++) {
        this.data[i] = Math.exp(this.data[i]);
        sum += this.data[i];
    }
    sum = 1.0 / sum;
    for (var i = 0; i < this.data.length; i++) {
        this.data[i] *= sum;
    }
    return this;
}

BlinkArray.prototype.Add = function (data) {
    for (var i = 0; i < this.data.length; i++) {
        this.data[i] += data[i];
    }
    return this;
}

BlinkArray.prototype.CopyTo = function (data) {
    for (var i = 0; i < this.data.length; i++) {
        data[i] = this.data[i];
    }
}
