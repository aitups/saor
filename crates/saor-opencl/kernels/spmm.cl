// SpMM para el DAG irregular (OpenCL C 1.2): Y[b][j] = sum_i X[b][i] * W[j][i].
// Variante densa-enmascarada (referencia) y variante CSR (producción).
// Un work-item por par (b, j); el bucle interno recorre d_in o los no-cero.

__kernel void spmm_dense(
    __global const float* x, // B * d_in
    __global const float* w, // d_out * d_in (enmascarada con ceros)
    const int d_in,
    const int d_out,
    __global float* y) // B * d_out
{
    const int gid = get_global_id(0);
    const int b = gid / d_out;
    const int j = gid % d_out;
    float acc = 0.0f;
    for (int i = 0; i < d_in; i++) {
        acc += x[b * d_in + i] * w[j * d_in + i];
    }
    y[b * d_out + j] = acc;
}

__kernel void spmm_csr(
    __global const float* x,     // B * d_in
    __global const int* row_ptr, // d_out + 1
    __global const int* col_idx, // nnz
    __global const float* vals,  // nnz
    const int d_in,
    const int d_out,
    __global float* y) // B * d_out
{
    const int gid = get_global_id(0);
    const int b = gid / d_out;
    const int j = gid % d_out;
    float acc = 0.0f;
    for (int k = row_ptr[j]; k < row_ptr[j + 1]; k++) {
        acc += x[b * d_in + col_idx[k]] * vals[k];
    }
    y[b * d_out + j] = acc;
}

// Cuenta conexiones activas por fila de salida `j` (D17: CSR en GPU para el
// warm-start teacher-copy, evitando el loop CPU O(N) por candidato).
__kernel void count_rows(
    __global const uchar* adj, // bit-tensor, conn = i*d_out+j (LSB-first)
    const int d_in,
    const int d_out,
    __global int* counts) // [d_out]
{
    const int j = get_global_id(0);
    int c = 0;
    for (int i = 0; i < d_in; i++) {
        const int conn = i * d_out + j;
        c += (adj[conn >> 3] >> (conn & 7)) & 1;
    }
    counts[j] = c;
}

// Gather de los pesos del profesor en las posiciones activas, en orden
// j-mayor (CSR: filas = salidas j, columnas = entradas i).
__kernel void gather_csr_teacher(
    __global const uchar* adj,
    __global const float* w0,       // [d_out, d_in] fila-mayor
    __global const int* row_ptr,    // [d_out+1]
    const int d_in,
    const int d_out,
    __global int* col_idx,          // nnz
    __global float* vals)           // nnz
{
    const int j = get_global_id(0);
    int w = 0;
    for (int i = 0; i < d_in; i++) {
        const int conn = i * d_out + j;
        if ((adj[conn >> 3] >> (conn & 7)) & 1) {
            const int off = row_ptr[j] + w;
            col_idx[off] = i;
            vals[off] = w0[j * d_in + i];
            w++;
        }
    }
}
