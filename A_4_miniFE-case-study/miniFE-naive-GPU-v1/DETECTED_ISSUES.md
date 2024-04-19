This file lists the issues found in this version:
* ISSUE in make_local_matrix():
	* NOTHING CHANGED
	* Potentially caused by virtual memory
* ISSUE in impose_dirichlet()
	* CPU implementation is faster then GPU implementation
* ISSUE in init_matrix()
	* NOTHING CHANGED
	* Potentially caused by virtual memory
* ISSUE in cg_solve():
	* Closer inspection reveals !$omp for @SparseMatrix_functions.hpp:517 is smaller than GPU matvec_std_operator_gpu_kernel(…)
	* And daxpy is smaller than daxpy_gpu
	* Likely caused by ineficcient access into datastructure
* ISSUE in generate_matrix_structure():
	* CPU implementation is faster then GPU implementation
