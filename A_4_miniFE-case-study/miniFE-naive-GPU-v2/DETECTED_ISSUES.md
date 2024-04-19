* ISSUE in main->miniFE::driver->miniFE::generate_matrix_structure->miniFE::init_matrix->miniFE::find_row_for_id:
	* Too many calls. Not on the GPU, expensive copy operations for data
* ISSUE in main->miniFE::driver->miniFE::generate_matrix_structure->miniFE::init_matrix
	* Not offloaded to the GPU, expensive copy operations for data
* ISSUE main->miniFE::driver->miniFE::cg_solve:
	* Potentially to many copy operations, due to access from both CPU and GPU.
* ISSUE main->miniFE::driver->miniFE::generate_matrix_structure->miniFE::init_matrix->miniFE::get_id
	* Too many calls. Not on the GPU, expensive copy operations for data
* ISSUE main->miniFE::driver->miniFE::generate_matrix_structure
	* MPI_Allreduce is a bottleneck, cannot be sped up using the GPU. So hardware-adjusted runtime cannot be met. => Shortcoming of expectation generation for hardware efficiency.
