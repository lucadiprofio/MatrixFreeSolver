#include "MatrixFree.hpp"

int main(int argc, char *argv[]) {

  // dim and fe_degree are compile-time constants on purpose. The whole point of
  // the matrix-free approach is that the polynomial degree is known to the
  // compiler: this lets deal.II generate fully-unrolled, vectorized
  // sum-factorization kernels (FEEvaluation) instead of generic loops. Passing
  // the degree as a template parameter is what makes the on-the-fly operator
  // evaluation competitive with (and more memory-efficient than) a stored
  // sparse matrix.

  static constexpr unsigned int dim = 3;
  static constexpr unsigned int fe_degree = 2;

  try {
    using namespace MtxFree;

    //It initializes the threading backend (TBB) used internally by deal.II.

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
    SolverClass<dim, fe_degree> solver;
    solver.run();
  } catch (std::exception &exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  } catch (...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
