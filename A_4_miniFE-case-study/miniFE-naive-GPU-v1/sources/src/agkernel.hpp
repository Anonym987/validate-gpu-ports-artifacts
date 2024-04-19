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

#ifdef MINIFE_HAVE_TBB
#include <LockingMatrix.hpp>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

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