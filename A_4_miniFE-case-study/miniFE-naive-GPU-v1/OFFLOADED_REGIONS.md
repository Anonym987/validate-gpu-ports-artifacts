This file lists the offloaded regions, that were added to the CPU-only version to create the naive GPU port.
Based on p=32 n=64000000: 
Five biggest parallel regions.

* 13.26	int generate_matrix_structure(const simple_mesh_description&, MatrixType&) -> 7750.11 !$omp parallel @generate_matrix_structure.hpp:114
* 123.69	void perform_element_loop(const simple_mesh_description&, const Box&, MatrixType&, VectorType&, Parameters&) -> 5.39e4 !$omp parallel @perform_element_loop.hpp:80
* 0.50	void cg_solve()  -> 2.39e4 !$omp parallel @SparseMatrix_functions.hpp:517
* 3663.97	void daxpby(double, const VectorType&, double, VectorType&) -> 2458.16 !$omp parallel @Vector_functions.hpp:215
* 1.89	void impose_dirichlet(typename ScalarType, MatrixType&, VectorType&, int, int, int, const set&) -> 1935.14 !$omp parallel @SparseMatrix_functions.hpp:463


