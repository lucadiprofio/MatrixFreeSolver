#include "MatrixFree.hpp"

int main(int argc, char *argv[]) {
  static constexpr unsigned int dim = 3;
  static constexpr unsigned int fe_degree = 2;

  try {
    using namespace MtxFree;
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
