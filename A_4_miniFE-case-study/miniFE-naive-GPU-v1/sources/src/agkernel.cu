#include <iostream>
#include <Box.hpp>
#include <box_utils.hpp>
#include <agkernel.hpp>
#include <SparseMatrix_functions.hpp>
#include <aggpucontainers.hpp>
#include <ag_simple_mesh_description.hpp>
#include <ElemData.hpp>
#include <Hex8_ElemData.hpp>
#include <agcudacomon.hpp>

__global__ void
impose_dirichlet_cuda_kernel(miniFE::CMatrixType::GlobalOrdinalType ROW_COUNT, gpu_matrix A, MINIFE_SCALAR *b, gpu_set<miniFE::CMatrixType::GlobalOrdinalType> bc_rows, miniFE::CMatrixType::ScalarType prescribed_value, MINIFE_GLOBAL_ORDINAL *Apc)
{
    typedef miniFE::CMatrixType::GlobalOrdinalType GlobalOrdinal;
    typedef miniFE::CMatrixType::LocalOrdinalType LocalOrdinal;
    typedef miniFE::CMatrixType::ScalarType Scalar;

    MINIFE_GLOBAL_ORDINAL i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= ROW_COUNT)
    {
        return;
    }

    GlobalOrdinal row = A.rows[i];

    if (bc_rows.find(row) != bc_rows.end())
        return;

    size_t row_length = 0;
    GlobalOrdinal *cols = NULL;
    Scalar *coefs = NULL;
    A.get_row_pointers(row, row_length, cols, coefs);

    Scalar sum = 0;
    for (size_t j = 0; j < row_length; ++j)
    {

        if (bc_rows.find(cols[j]) != bc_rows.end())
        {
            sum += coefs[j];

            coefs[j] = 0;
        }
    }

    // #pragma omp atomic
    atomicAdd(b + i, -sum * prescribed_value);
}

void miniFE::impose_dirichlet_cuda(CMatrixType::ScalarType prescribed_value,
                                   CMatrixType &A,
                                   CVectorType &b,
                                   int global_nx,
                                   int global_ny,
                                   int global_nz,
                                   const std::set<CMatrixType::GlobalOrdinalType> &bc_rows)
{
    typedef CMatrixType::GlobalOrdinalType GlobalOrdinal;
    typedef CMatrixType::LocalOrdinalType LocalOrdinal;
    typedef CMatrixType::ScalarType Scalar;
    std::cout << "CUDA...";

    GlobalOrdinal first_local_row = A.rows.size() > 0 ? A.rows[0] : 0;
    GlobalOrdinal last_local_row = A.rows.size() > 0 ? A.rows[A.rows.size() - 1] : -1;

    typename std::set<GlobalOrdinal>::const_iterator
        bc_iter = bc_rows.begin(),
        bc_end = bc_rows.end();
    for (; bc_iter != bc_end; ++bc_iter)
    {
        GlobalOrdinal row = *bc_iter;
        if (row >= first_local_row && row <= last_local_row)
        {
            size_t local_row = row - first_local_row;
            b.coefs[local_row] = prescribed_value;
            zero_row_and_put_1_on_diagonal(A, row);
        }
    }

    const int ROW_COUNT = A.rows.size();

    // #pragma omp parallel for
    //     for (MINIFE_GLOBAL_ORDINAL i = 0; i < ROW_COUNT; ++i)
    //     {
    //     }
    int minGridSize;
    int blockSize;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void *)impose_dirichlet_cuda_kernel,
        0,
        ROW_COUNT);

    auto gridSize = (ROW_COUNT + blockSize - 1) / blockSize;

    gpu_matrix a_gpu{
        .has_local_indices = A.has_local_indices,
        .rows = A.rows.data(),
        .rows_size = A.rows.size(),
        .row_offsets = A.row_offsets.data(),
        .row_offsets_external = A.row_offsets_external.data(),
        .packed_cols = A.packed_cols.data(),
        .packed_coefs = A.packed_coefs.data(),
        .num_cols = A.num_cols};

    gpu_set bc_rows_gpu(bc_rows);

    impose_dirichlet_cuda_kernel<<<gridSize, blockSize>>>(ROW_COUNT, a_gpu, b.coefs, bc_rows_gpu, prescribed_value, A.packed_cols.data());

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

__global__ void perform_element_loop_cuda_kernel(const miniFE::simple_mesh_description_gpu<MINIFE_GLOBAL_ORDINAL> mesh,
                                                 MINIFE_GLOBAL_ORDINAL *elemIDs, MINIFE_GLOBAL_ORDINAL elemID_size, gpu_matrix A, gpu_vector b)
{
    typedef MINIFE_SCALAR Scalar;
    typedef MINIFE_GLOBAL_ORDINAL GlobalOrdinal;
    using namespace miniFE;

    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= elemID_size)
    {
        return;
    }
    ElemData<GlobalOrdinal, Scalar> elem_data;
    compute_gradient_values(elem_data.grad_vals);

    auto elemId = elemIDs[i];

    get_elem_nodes_and_coords_gpu(mesh, elemId, elem_data.elem_node_ids, elem_data.elem_node_coords);
    compute_element_matrix_and_vector(elem_data);
    sum_into_global_linear_system(elem_data, A, b);
}

void miniFE::perform_element_loop_cuda(const simple_mesh_description<MINIFE_GLOBAL_ORDINAL> &mesh,
                                       std::vector<MINIFE_GLOBAL_ORDINAL, UvmAllocator<MINIFE_GLOBAL_ORDINAL>> &elemIDs,
                                       CMatrixType &A, CVectorType &b)
{
    typedef CMatrixType::ScalarType Scalar;
    typedef MINIFE_GLOBAL_ORDINAL GlobalOrdinal;

    std::cout << "CUDA...";

    const MINIFE_GLOBAL_ORDINAL elemID_size = elemIDs.size();

    ElemData<GlobalOrdinal, Scalar> elem_data;

    simple_mesh_description_gpu<MINIFE_GLOBAL_ORDINAL> gpu_mesh(mesh);

    gpu_matrix a_gpu{
        .has_local_indices = A.has_local_indices,
        .rows = A.rows.data(),
        .rows_size = A.rows.size(),
        .row_offsets = A.row_offsets.data(),
        .row_offsets_external = A.row_offsets_external.data(),
        .packed_cols = A.packed_cols.data(),
        .packed_coefs = A.packed_coefs.data(),
        .num_cols = A.num_cols};

    int minGridSize;
    int blockSize;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void *)perform_element_loop_cuda_kernel,
        0,
        elemID_size);

    auto gridSize = (elemID_size + blockSize - 1) / blockSize;

    perform_element_loop_cuda_kernel<<<gridSize, blockSize>>>(gpu_mesh, elemIDs.data(), elemID_size, a_gpu, b);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // #pragma omp parallel for shared(elemIDs)
    //     for (MINIFE_GLOBAL_ORDINAL i = 0; i < elemID_size; ++i)
    //     {
    //         ElemData<GlobalOrdinal, Scalar> elem_data;
    //         compute_gradient_values(elem_data.grad_vals);

    //         get_elem_nodes_and_coords(mesh, elemIDs[i], elem_data);
    //         compute_element_matrix_and_vector(elem_data);
    //         sum_into_global_linear_system(elem_data, A, b);
    //     }

    // std::cout << std::endl<<"get-nodes: " << t_gn << std::endl;
    // std::cout << "compute-elems: " << t_ce << std::endl;
    // std::cout << "sum-in: " << t_si << std::endl;
}

__global__ void matvec_std_operator_gpu_kernel(const MINIFE_GLOBAL_ORDINAL rows_size, const MINIFE_LOCAL_ORDINAL *const Arowoffsets, const MINIFE_GLOBAL_ORDINAL *const Acols, const MINIFE_SCALAR *const Acoefs, const MINIFE_SCALAR *const xcoefs, MINIFE_SCALAR *ycoefs)
{
    MINIFE_GLOBAL_ORDINAL row = threadIdx.x + blockIdx.x * blockDim.x;

    if (row >= rows_size)
    {
        return;
    }

    const MINIFE_GLOBAL_ORDINAL row_start = Arowoffsets[row];
    const MINIFE_GLOBAL_ORDINAL row_end = Arowoffsets[row + 1];

    MINIFE_SCALAR sum = 0;

    for (MINIFE_GLOBAL_ORDINAL i = row_start; i < row_end; ++i)
    {
        sum += Acoefs[i] * xcoefs[Acols[i]];
    }

    ycoefs[row] = sum;
}

void miniFE::matvec_std_operator_gpu(CMatrixType &A,
                                     CVectorType &x,
                                     CVectorType &y)
{
    exchange_externals(A, x);

    typedef typename CMatrixType::ScalarType ScalarType;
    typedef typename CMatrixType::GlobalOrdinalType GlobalOrdinalType;
    typedef typename CMatrixType::LocalOrdinalType LocalOrdinalType;

    const MINIFE_GLOBAL_ORDINAL rows_size = A.rows.size();
    const LocalOrdinalType *const Arowoffsets = &A.row_offsets[0];
    const GlobalOrdinalType *const Acols = &A.packed_cols[0];
    const ScalarType *const Acoefs = &A.packed_coefs[0];
    const ScalarType *const xcoefs = &x.coefs[0];
    ScalarType *ycoefs = &y.coefs[0];

    int minGridSize;
    int blockSize;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void *)matvec_std_operator_gpu_kernel,
        0,
        rows_size);

    auto gridSize = (rows_size + blockSize - 1) / blockSize;
    matvec_std_operator_gpu_kernel<<<gridSize, blockSize>>>(rows_size, Arowoffsets, Acols, Acoefs, xcoefs, ycoefs);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

__global__ void dxpy_kernel(const size_t n, const MINIFE_SCALAR *xcoefs, MINIFE_SCALAR *ycoefs)
{
    MINIFE_GLOBAL_ORDINAL i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n)
    {
        return;
    }
    ycoefs[i] += xcoefs[i];
}
__global__ void daxpy_kernel(const size_t n, const MINIFE_SCALAR *xcoefs, MINIFE_SCALAR *ycoefs, const MINIFE_SCALAR alpha)
{
    MINIFE_GLOBAL_ORDINAL i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n)
    {
        return;
    }
    ycoefs[i] += alpha * xcoefs[i];
}
__global__ void dxbpy_kernel(const size_t n, const MINIFE_SCALAR *xcoefs, MINIFE_SCALAR *ycoefs, const MINIFE_SCALAR beta)
{
    MINIFE_GLOBAL_ORDINAL i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n)
    {
        return;
    }
    ycoefs[i] = xcoefs[i] + beta * ycoefs[i];
}
__global__ void dax_kernel(const size_t n, const MINIFE_SCALAR *xcoefs, MINIFE_SCALAR *ycoefs, const MINIFE_SCALAR alpha)
{
    MINIFE_GLOBAL_ORDINAL i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n)
    {
        return;
    }
    ycoefs[i] = alpha * xcoefs[i];
}
__global__ void daxpby_kernel(const size_t n, const MINIFE_SCALAR *xcoefs, MINIFE_SCALAR *ycoefs, const MINIFE_SCALAR alpha, const MINIFE_SCALAR beta)
{
    MINIFE_GLOBAL_ORDINAL i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n)
    {
        return;
    }
    ycoefs[i] = alpha * xcoefs[i] + beta * ycoefs[i];
}

void miniFE::daxpby_gpu(const MINIFE_SCALAR alpha,
                        const CVectorType &x,
                        const MINIFE_SCALAR beta,
                        CVectorType &y)
{
    const MINIFE_LOCAL_ORDINAL n = MINIFE_MIN(x.local_size, y.local_size);
    const MINIFE_SCALAR *xcoefs = &x.coefs[0];
    MINIFE_SCALAR *ycoefs = &y.coefs[0];

    int minGridSize;
    int blockSize;

    if (alpha == 1.0 && beta == 1.0)
    {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)dxpy_kernel, 0, n);
        auto gridSize = (n + blockSize - 1) / blockSize;
        dxpy_kernel<<<gridSize, blockSize>>>(n, xcoefs, ycoefs);
    }
    else if (beta == 1.0)
    {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)daxpy_kernel, 0, n);
        auto gridSize = (n + blockSize - 1) / blockSize;
        daxpy_kernel<<<gridSize, blockSize>>>(n, xcoefs, ycoefs, alpha);
    }
    else if (alpha == 1.0)
    {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)dxbpy_kernel, 0, n);
        auto gridSize = (n + blockSize - 1) / blockSize;
        dxbpy_kernel<<<gridSize, blockSize>>>(n, xcoefs, ycoefs, beta);
    }
    else if (beta == 0.0)
    {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)dax_kernel, 0, n);
        auto gridSize = (n + blockSize - 1) / blockSize;
        dax_kernel<<<gridSize, blockSize>>>(n, xcoefs, ycoefs, alpha);
    }
    else
    {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)daxpby_kernel, 0, n);
        auto gridSize = (n + blockSize - 1) / blockSize;
        daxpby_kernel<<<gridSize, blockSize>>>(n, xcoefs, ycoefs, alpha, beta);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

__global__ void generate_matrix_structure_gpu_kernel(const miniFE::simple_mesh_description_gpu<MINIFE_GLOBAL_ORDINAL> mesh,
                                                     const Box box, MINIFE_GLOBAL_ORDINAL *const row_ptr,
                                                     MINIFE_LOCAL_ORDINAL *const row_offset_ptr,
                                                     MINIFE_LOCAL_ORDINAL *const row_coords_ptr,
                                                     const int global_nodes_x, const int global_nodes_y,
                                                     const int global_nodes_z, const MINIFE_GLOBAL_ORDINAL global_nrows)
{

    MINIFE_GLOBAL_ORDINAL r = threadIdx.x + blockIdx.x * blockDim.x;
    const MINIFE_GLOBAL_ORDINAL r_n = (box[2][1] - box[2][0]) *
                                      (box[1][1] - box[1][0]) *
                                      (box[0][1] - box[0][0]);
    if (r >= r_n)
    {
        return;
    }
    const MINIFE_GLOBAL_ORDINAL z_width = box[2][1] - box[2][0];
    const MINIFE_GLOBAL_ORDINAL y_width = box[1][1] - box[1][0];
    const MINIFE_GLOBAL_ORDINAL x_width = box[0][1] - box[0][0];
    const MINIFE_GLOBAL_ORDINAL xy_width = x_width * y_width;

    int iz = r / (xy_width) + box[2][0];
    int iy = (r / x_width) % y_width + box[1][0];
    int ix = r % x_width + box[0][0];

    MINIFE_GLOBAL_ORDINAL row_id =
        miniFE::get_id<MINIFE_GLOBAL_ORDINAL>(global_nodes_x, global_nodes_y, global_nodes_z,
                                              ix, iy, iz);
    row_ptr[r] = mesh.map_id_to_row(row_id);
    row_coords_ptr[r * 3] = ix;
    row_coords_ptr[r * 3 + 1] = iy;
    row_coords_ptr[r * 3 + 2] = iz;

    MINIFE_LOCAL_ORDINAL nnz = 0;
    for (int sz = -1; sz <= 1; ++sz)
    {
        for (int sy = -1; sy <= 1; ++sy)
        {
            for (int sx = -1; sx <= 1; ++sx)
            {
                MINIFE_GLOBAL_ORDINAL col_id =
                    miniFE::get_id<MINIFE_GLOBAL_ORDINAL>(global_nodes_x, global_nodes_y, global_nodes_z,
                                                          ix + sx, iy + sy, iz + sz);

                if (col_id >= 0 && col_id < global_nrows)
                {
                    ++nnz;
                }
            }
        }
    }
    row_offset_ptr[r + 1] = nnz;
}

void miniFE::generate_matrix_structure_gpu(const simple_mesh_description<MINIFE_GLOBAL_ORDINAL> &mesh,
                                           Box &box, MINIFE_GLOBAL_ORDINAL *const row_ptr,
                                           MINIFE_LOCAL_ORDINAL *const row_offset_ptr,
                                           MINIFE_LOCAL_ORDINAL *const row_coords_ptr,
                                           const int global_nodes_x, const int global_nodes_y,
                                           const int global_nodes_z, const MINIFE_GLOBAL_ORDINAL global_nrows)
{
    const MINIFE_GLOBAL_ORDINAL r_n = (box[2][1] - box[2][0]) *
                                      (box[1][1] - box[1][0]) *
                                      (box[0][1] - box[0][0]);

    int minGridSize;
    int blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)generate_matrix_structure_gpu_kernel, 0, r_n);
    auto gridSize = (r_n + blockSize - 1) / blockSize;
    generate_matrix_structure_gpu_kernel<<<gridSize, blockSize>>>(mesh, box, row_ptr, row_offset_ptr, row_coords_ptr, global_nodes_x, global_nodes_y, global_nodes_z, global_nrows);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}