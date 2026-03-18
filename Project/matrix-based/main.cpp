#include "MatrixBased.hpp"

int main(int argc, char *argv[]) {
  constexpr unsigned int dim = 3;
  constexpr unsigned int fe_degree = 2;

  try {
    using namespace MatrixBased;
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