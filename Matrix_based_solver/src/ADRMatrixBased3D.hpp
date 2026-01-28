#ifndef ADR_MATRIX_BASED_3D_HPP
#define ADR_MATRIX_BASED_3D_HPP

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <filesystem>
#include <fstream>
#include <iostream>

#include "../../Coefficients.hpp"

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class ADRMatrixBased3D
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  ADRMatrixBased3D(const unsigned int r_):
    r(r_)
    , dof_handler(triangulation)
{}

  // Initialization.
  void
  setup();

  // System assembly.
  void
  assemble();

  // System solution.
  void
  solve();

  // Output.
  void
  output() const;

  // Compute the error against a given exact solution.
  double
  compute_error(const VectorTools::NormType &norm_type,
                const Function<dim>         &exact_solution) const;

  void set_boundary_ids();

protected:

  // Polynomial degree.
  const unsigned int r;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  dealii::Triangulation<dim> triangulation;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // Quadrature formula for boundary integrals.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // Sparsity pattern.
  SparsityPattern sparsity_pattern;

  // System matrix.
  SparseMatrix<double> system_matrix;

  // System right-hand side.
  Vector<double> system_rhs;

  // System solution.
  Vector<double> solution;

  std::set<types::boundary_id> dirichlet_ids = {1};

  std::set<types::boundary_id> neumann_ids = {};

  bool use_advection = false; 

  const Project::Mu<dim> mu;

  const Project::Beta<dim> beta;

  const Project::Gamma<dim> gamma;

  const Project::WellKnownTerm<dim> f;

  const Project::DirichletBC<dim> dirichlet_bc;

  const Project::NeumannBC<dim> neumann_bc;

};

#endif