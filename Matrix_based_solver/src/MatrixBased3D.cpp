#include "ADRMatrixBased3D.hpp"

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  constexpr unsigned int dim = ADRMatrixBased3D::dim;

  const std::string  mesh_filename = "../mesh/mesh-cube-40.msh";
  const unsigned int degree        = 1;

  const auto mu = [](const Point<dim> &/*p*/) { return 1.0; };
  const auto f     = [](const Point<dim>     &/*p*/) { return 1.0; };
  const auto sigma = [](const Point<dim> & /*p*/) { return 0.0; };
  const auto beta =  [](const Point<dim>&/*p*/){ 
            Tensor<1,dim> b; 
            b[0] = 1.0; b[1] = 0.0; b[2] = 0.0; 
            return b; 
        };

  ADRMatrixBased3D problem(mesh_filename, degree, mu, beta, sigma, f);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}