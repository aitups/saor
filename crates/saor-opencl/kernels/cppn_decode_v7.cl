// Decodificador CPPN-v7 (Vía B-v7: la CPPN GENERA los pesos con geometría
// aprendida por canal) en OpenCL C 1.2. Un work-item por CONEXIÓN.
//
// El genoma v7 (7250 f32, hidden=48) es:
//   w0[48*6] | b0[48] | wm0[48*48] | bm0[48] | wm1[48*48] | bm1[48]
//     | w2[2*48] | b2[2] | coord_h[576] | coord_i[1536]
// (espejo de `python/saor_orchestrator/reference/v7cppn.py::from_flatten`).
//
// Por conexión conn = i*d_out + j (i = entrada, j = salida; i-mayor, igual que
// el bit-tensor `ffn_dag_adjacency`): evalúa la CPPN sobre v = [ci, cj, cj-ci,
// z, 0, 0] donde ci/cj son las coordenadas APRENDIDAS de los canales (modo
// "hi": entrada=coord_h, salida=coord_i; modo "ih": al revés), aplica la
// máscara sigmoid(link) > tau y escribe el peso denso enmascarado + el bit de
// adyacencia. Semántica exacta del contrato v7: adyacencia = (l > tau),
// pesos = salida de pesos de la CPPN.

#define V7_INPUT 6
#define V7_HIDDEN 48
#define V7_COORD_H_N 576
#define V7_COORD_I_N 1536
#define V7_OFF_W0 0
#define V7_OFF_B0 (V7_HIDDEN * V7_INPUT)                  // 288
#define V7_OFF_WM0 (V7_OFF_B0 + V7_HIDDEN)                // 336
#define V7_OFF_BM0 (V7_OFF_WM0 + V7_HIDDEN * V7_HIDDEN)   // 2640
#define V7_OFF_WM1 (V7_OFF_BM0 + V7_HIDDEN)               // 2688
#define V7_OFF_BM1 (V7_OFF_WM1 + V7_HIDDEN * V7_HIDDEN)   // 4992
#define V7_OFF_W2 (V7_OFF_BM1 + V7_HIDDEN)                // 5040
#define V7_OFF_B2 (V7_OFF_W2 + 2 * V7_HIDDEN)             // 5136
#define V7_OFF_COORD_H (V7_OFF_B2 + 2)                    // 5138
#define V7_OFF_COORD_I (V7_OFF_COORD_H + V7_COORD_H_N)    // 5714
#define V7_GENOME_LEN (V7_OFF_COORD_I + V7_COORD_I_N)     // 7250
#define V7_PI 3.14159265358979323846f

// Evalúa la CPPN-v7 para un vector de entrada [xi, xj, z]: la arquitectura es
//   h = tanh(v·w0 + b0)
//   2 capas medias: h <- sin(PI·h); h <- tanh(h·wm + bm); h <- exp(-½(1.2h)²)
//   salida: w = b2_0 + Σ h·w2_0; l_raw = b2_1 + Σ h·w2_1; l = sigmoid(l_raw)
// Secuencia f32 escalar (referencia canónica de paridad). Devuelve el peso y
// deja el sigmoide del link en *out_l.
float v7_eval_cppn(__global const float* g, float xi, float xj, float z,
                   float* out_l) {
    float v[V7_INPUT];
    v[0] = xi;
    v[1] = xj;
    v[2] = xj - xi;
    v[3] = z;
    v[4] = 0.0f;
    v[5] = 0.0f;

    // h0 = tanh(v·w0^T + b0)
    float a[V7_HIDDEN];
    for (int o = 0; o < V7_HIDDEN; o++) {
        float acc = g[V7_OFF_B0 + o];
        for (int k = 0; k < V7_INPUT; k++) {
            acc += g[o * V7_INPUT + k] * v[k];
        }
        a[o] = tanh(acc);
    }
    float b[V7_HIDDEN];
    for (int it = 0; it < 2; it++) {
        const int off_w = (it == 0) ? V7_OFF_WM0 : V7_OFF_WM1;
        const int off_b = (it == 0) ? V7_OFF_BM0 : V7_OFF_BM1;
        // b = sin(PI·a)
        for (int o = 0; o < V7_HIDDEN; o++) {
            b[o] = sin(V7_PI * a[o]);
        }
        // a = exp(-½(1.2·tanh(b·wm^T + bm))²)
        for (int o = 0; o < V7_HIDDEN; o++) {
            float acc = g[off_b + o];
            for (int m = 0; m < V7_HIDDEN; m++) {
                acc += g[off_w + o * V7_HIDDEN + m] * b[m];
            }
            acc = tanh(acc);
            float q = 1.2f * acc;
            a[o] = exp(-0.5f * q * q);
        }
    }
    // Salida lineal final sobre a.
    float acc_w = g[V7_OFF_B2 + 0];
    float acc_l = g[V7_OFF_B2 + 1];
    for (int o = 0; o < V7_HIDDEN; o++) {
        acc_w += g[V7_OFF_W2 + 0 * V7_HIDDEN + o] * a[o];
        acc_l += g[V7_OFF_W2 + 1 * V7_HIDDEN + o] * a[o];
    }
    *out_l = 1.0f / (1.0f + exp(-acc_l));
    return acc_w;
}

// Decodifica una conexión del genoma-v7: escribe w_out[conn] (denso
// enmascarado, ceros donde l <= tau) y acumula la adyacencia en palabras u32.
__kernel void cppn_decode_v7(
    __global const float* genome, // [V7_GENOME_LEN]
    const int d_in,
    const int d_out,
    const float tau,
    const int layer,
    const int n_layers,
    const int mode,               // 0 = "hi" (ci=coord_h, cj=coord_i); 1 = "ih"
    __global float* w_out,        // d_in * d_out (i-mayor), enmascarada
    __global uint* adj_words,     // ceil(total/32)
    __global uint* active_out)    // [1]
{
    const int conn = get_global_id(0);
    const int total = d_in * d_out;
    if (conn >= total) return;
    const int i = conn / d_out;
    const int j = conn % d_out;
    float ci = (mode == 0) ? genome[V7_OFF_COORD_H + i] : genome[V7_OFF_COORD_I + i];
    float cj = (mode == 0) ? genome[V7_OFF_COORD_I + j] : genome[V7_OFF_COORD_H + j];
    if (ci < -5.0f) ci = -5.0f; else if (ci > 5.0f) ci = 5.0f;
    if (cj < -5.0f) cj = -5.0f; else if (cj > 5.0f) cj = 5.0f;
    float zc = (n_layers > 1) ? -1.0f + 2.0f * ((float)layer + 0.5f) / (float)n_layers : 0.0f;
    if (zc < -2.0f) zc = -2.0f; else if (zc > 2.0f) zc = 2.0f;
    float l;
    float w = v7_eval_cppn(genome, ci, cj, zc, &l);
    if (l > tau) {
        w_out[conn] = w;
        atomic_or(&adj_words[conn >> 5], 1u << (conn & 31));
        atomic_inc(active_out);
    } else {
        w_out[conn] = 0.0f;
    }
}

// Convierte las palabras u32 de adyacencia al bit-tensor u8 (LSB-first).
__kernel void v7_pack_adjacency(
    __global const uint* adj_words, // ceil(total/32)
    __global uchar* adj_out)        // ceil(total/8)
{
    const int w = get_global_id(0);
    const uint word = adj_words[w];
    adj_out[4 * w + 0] = (uchar)(word & 0xFFu);
    adj_out[4 * w + 1] = (uchar)((word >> 8) & 0xFFu);
    adj_out[4 * w + 2] = (uchar)((word >> 16) & 0xFFu);
    adj_out[4 * w + 3] = (uchar)((word >> 24) & 0xFFu);
}
