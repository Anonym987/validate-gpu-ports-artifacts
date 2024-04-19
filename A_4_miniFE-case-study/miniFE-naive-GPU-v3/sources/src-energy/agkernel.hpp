#pragma once

#include <cstddef>
#include <vector>
#include <set>
#include <algorithm>
#include <sstream>
#include <fstream>

#include <Vector.hpp>
#include <Vector_functions.hpp>
#include <ElemData.hpp>
#include <MatrixInitOp.hpp>
#include <MatrixCopyOp.hpp>
#include <exchange_externals.hpp>
#include <mytimer.hpp>
#include <agcudacomon.hpp>
#include <cusparse.h>

#ifdef MINIFE_HAVE_TBB
#include <LockingMatrix.hpp>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#define CHECK_CUSP_ERROR(val) check_cusp((val), #val, __FILE__, __LINE__)
template <typename T>
void check_cusp(T err, const char *const func, const char *const file,
                const int line)
{
    if (err != CUSPARSE_STATUS_SUCCESS)
    {
        std::cerr << "CuSparse Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cusparseGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline cusparseHandle_t cusparseHandle;

namespace miniFE
{
    typedef CSRMatrix<MINIFE_SCALAR, MINIFE_LOCAL_ORDINAL, MINIFE_GLOBAL_ORDINAL> CMatrixType;
    typedef Vector<MINIFE_SCALAR, MINIFE_LOCAL_ORDINAL, MINIFE_GLOBAL_ORDINAL> CVectorType;
    void impose_dirichlet_cuda(CMatrixType::ScalarType prescribed_value,
                               CMatrixType &A,
                               CVectorType &b,
                               int global_nx,
                               int global_ny,
                               int global_nz,
                               const std::set<CMatrixType::GlobalOrdinalType> &bc_rows);
    void perform_element_loop_cuda(const simple_mesh_description<MINIFE_GLOBAL_ORDINAL> &mesh,
                                   std::vector<MINIFE_GLOBAL_ORDINAL, UvmAllocator<MINIFE_GLOBAL_ORDINAL>> &elemIDs,
                                   CMatrixType &A, CVectorType &b);
    void matvec_std_operator_gpu(CMatrixType &A,
                                 CVectorType &x,
                                 CVectorType &y);
    void daxpby_gpu(const MINIFE_SCALAR alpha,
                    const CVectorType &x,
                    const MINIFE_SCALAR beta,
                    CVectorType &y);
    void generate_matrix_structure_gpu(const simple_mesh_description<MINIFE_GLOBAL_ORDINAL> &mesh,
                                       Box &box, MINIFE_GLOBAL_ORDINAL *const row_ptr,
                                       MINIFE_LOCAL_ORDINAL *const row_offset_ptr,
                                       MINIFE_LOCAL_ORDINAL *const row_coords_ptr,
                                       const int global_nodes_x, const int global_nodes_y,
                                       const int global_nodes_z, const MINIFE_GLOBAL_ORDINAL global_nrows);
    void init_matrix_gpu(CMatrixType &M,
                         const std::vector<MINIFE_GLOBAL_ORDINAL, UvmAllocator<MINIFE_GLOBAL_ORDINAL>> &rows,
                         const std::vector<MINIFE_LOCAL_ORDINAL, UvmAllocator<MINIFE_LOCAL_ORDINAL>> &row_offsets,
                         const std::vector<int, UvmAllocator<int>> &row_coords,
                         int global_nodes_x,
                         int global_nodes_y,
                         int global_nodes_z,
                         MINIFE_GLOBAL_ORDINAL global_nrows,
                         const simple_mesh_description<MINIFE_GLOBAL_ORDINAL> &mesh);

    MINIFE_SCALAR dot_r2_gpu(const CVectorType &x);
    MINIFE_SCALAR dot_gpu(const CVectorType &x, const CVectorType &y);
}

template <typename VectorType, typename ValueType>
void vector_compare(VectorType a, VectorType b, ValueType maxDiff = 0, std::string name = "A")
{
    for (size_t i = 0; i < a.size(); i++)
    {
        if (abs(b.at(i) - a.at(i)) > maxDiff)
        {
            std::cerr << name << " different at " << i << " A: " << a.at(i) << " A_gpu: " << b.at(i) << std::endl;
            std::exit(5);
        }
    }
}

template <typename V>
void memPrefetchVectorToCPU(V &vector)
{
    // size_t size = vector.size() * sizeof(typename V::value_type);
    // if (size > 0)
    // {
    //     CHECK_CUDA_ERROR(cudaMemPrefetchAsync(vector.data(), size, cudaCpuDeviceId));
    // }
}

__device__ void sort_if_needed_gpu(MINIFE_GLOBAL_ORDINAL *list,
                                   MINIFE_GLOBAL_ORDINAL list_len);

template <typename MatrixType>
struct MatrixInitOpGPU
{
};

template <>
struct MatrixInitOpGPU<miniFE::CSRMatrix<MINIFE_SCALAR, MINIFE_LOCAL_ORDINAL, MINIFE_GLOBAL_ORDINAL>>
{
    MatrixInitOpGPU(const std::vector<MINIFE_GLOBAL_ORDINAL, UvmAllocator<MINIFE_GLOBAL_ORDINAL>> &rows_vec,
                    const std::vector<MINIFE_LOCAL_ORDINAL, UvmAllocator<MINIFE_LOCAL_ORDINAL>> &row_offsets_vec,
                    const std::vector<int, UvmAllocator<int>> &row_coords_vec,
                    int global_nx, int global_ny, int global_nz,
                    MINIFE_GLOBAL_ORDINAL global_n_rows,
                    const miniFE::simple_mesh_description<MINIFE_GLOBAL_ORDINAL> &input_mesh,
                    miniFE::CSRMatrix<MINIFE_SCALAR, MINIFE_LOCAL_ORDINAL, MINIFE_GLOBAL_ORDINAL> &matrix)
        : rows(&rows_vec[0]),
          row_offsets(&row_offsets_vec[0]),
          row_coords(&row_coords_vec[0]),
          global_nodes_x(global_nx),
          global_nodes_y(global_ny),
          global_nodes_z(global_nz),
          global_nrows(global_n_rows),
          mesh(input_mesh),
          dest_rows(&matrix.rows[0]),
          dest_rowoffsets(&matrix.row_offsets[0]),
          dest_cols(&matrix.packed_cols[0]),
          dest_coefs(&matrix.packed_coefs[0]),
          n(matrix.rows.size())
    {
        if (matrix.packed_cols.capacity() != matrix.packed_coefs.capacity())
        {
            std::cout << "Warning, packed_cols.capacity (" << matrix.packed_cols.capacity() << ") != "
                      << "packed_coefs.capacity (" << matrix.packed_coefs.capacity() << ")" << std::endl;
        }

        size_t nnz = row_offsets_vec[n];
        if (matrix.packed_cols.capacity() < nnz)
        {
            std::cout << "Warning, packed_cols.capacity (" << matrix.packed_cols.capacity() << ") < "
                                                                                               " nnz ("
                      << nnz << ")" << std::endl;
        }

        matrix.packed_cols.resize(nnz);
        matrix.packed_coefs.resize(nnz);
        dest_rowoffsets[n] = nnz;
    }

    typedef MINIFE_GLOBAL_ORDINAL GlobalOrdinalType;
    typedef MINIFE_LOCAL_ORDINAL LocalOrdinalType;
    typedef MINIFE_SCALAR ScalarType;

    const GlobalOrdinalType *rows;
    const LocalOrdinalType *row_offsets;
    const int *row_coords;

    int global_nodes_x;
    int global_nodes_y;
    int global_nodes_z;

    GlobalOrdinalType global_nrows;

    GlobalOrdinalType *dest_rows;
    LocalOrdinalType *dest_rowoffsets;
    GlobalOrdinalType *dest_cols;
    ScalarType *dest_coefs;
    int n;

    const miniFE::simple_mesh_description_gpu<GlobalOrdinalType> mesh;

    __device__ inline void operator()(int i)
    {
        dest_rows[i] = rows[i];
        int offset = row_offsets[i];
        dest_rowoffsets[i] = offset;
        int ix = row_coords[i * 3];
        int iy = row_coords[i * 3 + 1];
        int iz = row_coords[i * 3 + 2];
        GlobalOrdinalType nnz = 0;
        for (int sz = -1; sz <= 1; ++sz)
        {
            for (int sy = -1; sy <= 1; ++sy)
            {
                for (int sx = -1; sx <= 1; ++sx)
                {
                    GlobalOrdinalType col_id =
                        miniFE::get_id<GlobalOrdinalType>(global_nodes_x, global_nodes_y, global_nodes_z,
                                                          ix + sx, iy + sy, iz + sz);
                    if (col_id >= 0 && col_id < global_nrows)
                    {
                        GlobalOrdinalType col = mesh.map_id_to_row(col_id);
                        dest_cols[offset + nnz] = col;
                        dest_coefs[offset + nnz] = 0;
                        ++nnz;
                    }
                }
            }
        }

        sort_if_needed_gpu(&dest_cols[offset], nnz);
    }
};