#ifndef MATRIX_BASED_HPP
#define MATRIX_BASED_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include "../test/manufactured.hpp"
#include "../test/test1.hpp"
#include "../test/test2.hpp"


namespace MatrixBased {
using namespace dealii;
using namespace test1;

template <unsigned int dim, unsigned int degree> class SolverClass {
public:
  SolverClass()
      : mpi_communicator(MPI_COMM_WORLD),
        mpi_size(Utilities::MPI::n_mpi_processes(mpi_communicator)),
        mpi_rank(Utilities::MPI::this_mpi_process(mpi_communicator)),

        triangulation(mpi_communicator), mapping(), fe(degree),
        dof_handler(triangulation),

        pcout(std::cout, mpi_rank == 0),
        computing_timer(mpi_communicator, pcout, TimerOutput::summary, TimerOutput::wall_times),

        mu_function(), b_function(), sigma_function(), rhs_function(),
        dirichlet_function_inlet(), dirichlet_function_walls(),
        neumann_function(), exact_solution() {};

  void run();

private:
  using MatrixType = TrilinosWrappers::SparseMatrix;
  using VectorType = TrilinosWrappers::MPI::Vector;

  using SolverType = TrilinosWrappers::SolverCG;
  using PrecType = TrilinosWrappers::PreconditionAMG;

  void init_mesh();
  void setup_system();
  void assemble_system();
  void solve();
  void refine_grid(const VectorType &);
  void output_results(const unsigned int, const VectorType &);
  void estimate_error(const unsigned int, const VectorType &);
  void print_tables();
  void track_memory() const;

  std::set<types::boundary_id> neumann_boundary_ids{};

  MPI_Comm mpi_communicator;

  const unsigned int mpi_size;
  const unsigned int mpi_rank;

  parallel::distributed::Triangulation<dim> triangulation;
  const MappingQ1<dim> mapping;
  const FE_Q<dim> fe;

  DoFHandler<dim> dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;
  AffineConstraints<double> constraints;

  MatrixType system_matrix;
  VectorType solution;
  VectorType system_rhs;

  ConditionalOStream pcout;
  ConvergenceTable convergence_table;
  TimerOutput computing_timer;

  // problem data
  DiffusionCoefficient<dim> mu_function;
  AdvectionCoefficient<dim> b_function;
  ReactionCoefficient<dim> sigma_function;

  RightHandSide<dim> rhs_function;

  DirichletBoundaryInlet<dim> dirichlet_function_inlet;
  DirichletBoundaryWalls<dim> dirichlet_function_walls;
  NeumannBoundaryValues<dim> neumann_function;

  ExactSolution<dim> exact_solution;
};

} // namespace MatrixBased

#endif
