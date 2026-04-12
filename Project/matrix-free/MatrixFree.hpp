#ifndef MATRIX_FREE_HPP
#define MATRIX_FREE_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_gmres.h>

#include "ADROperator.hpp"

namespace MtxFree {
using namespace dealii;

template <unsigned int dim, unsigned int degree> class SolverClass {
public:
  SolverClass()
      : mpi_communicator(MPI_COMM_WORLD),
        mpi_size(Utilities::MPI::n_mpi_processes(mpi_communicator)),
        mpi_rank(Utilities::MPI::this_mpi_process(mpi_communicator)),

        triangulation(mpi_communicator,
                      Triangulation<dim>::limit_level_difference_at_vertices,
                      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
        mapping(), fe(degree), dof_handler(triangulation),

        pcout(std::cout, mpi_rank == 0),
        computing_timer(mpi_communicator, pcout, TimerOutput::summary,
                        TimerOutput::wall_times),

        mu_function(), b_function(), sigma_function(), rhs_function(),
        dirichlet_function_inlet(), dirichlet_function_walls(),
        neumann_function(), exact_solution() {};

  void run();

private:
  using MatrixFreeActiveMatrix = ADR_Operator<dim, degree, double>;
  using MatrixFreeLevelMatrix = ADR_Operator<dim, degree, float>;
  using MatrixFreeActiveVector = LinearAlgebra::distributed::Vector<double>;
  using MatrixFreeLevelVector = LinearAlgebra::distributed::Vector<float>;

  template <typename VecType> using SolverType = SolverGMRES<VecType>;

  void init_mesh();
  void setup_system();
  void setup_multigrid();
  void assemble_rhs();
  void solve();
  void refine_grid();
  void output_results(const unsigned int);
  void estimate_error(const unsigned int);
  void print_tables();
  void track_memory() const;

  
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

  MatrixFreeActiveMatrix system_matrix;
  MatrixFreeActiveVector locally_relevant_solution;
  MatrixFreeActiveVector system_rhs;

  MGConstrainedDoFs mg_constrained_dofs;
  MGLevelObject<MatrixFreeLevelMatrix> mg_matrices;

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

  std::set<types::boundary_id> neumann_boundary_ids{};

};

} // namespace MtxFree

#endif