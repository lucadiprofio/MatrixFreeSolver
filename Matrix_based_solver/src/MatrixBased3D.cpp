#include "ADRMatrixBased3D.hpp"
#include "../../Coefficients.hpp"

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  constexpr unsigned int dim = ADRMatrixBased3D::dim;

  const unsigned int degree        = 1;

  ADRMatrixBased3D problem(degree);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}