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
