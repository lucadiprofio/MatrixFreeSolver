#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include "ADROperator.hpp"
#include "../../Coefficients.hpp"

namespace Project
{
  using namespace dealii;

  constexpr unsigned int degree_finite_element = 2;
  constexpr unsigned int dim       = 3;

  class ADRProblem
  {
  public:
    ADRProblem();
    void run();

  private:
    void set_boundary_ids();
    void setup_system();
    void RHS();
    void solve();
    void output_results(unsigned int cycle) const;


#ifdef DEAL_II_WITH_P4EST
    parallel::distributed::Triangulation<dim> triangulation;
#else
    Triangulation<dim> triangulation;
#endif

    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;
    MappingQ1<dim>  mapping;

    AffineConstraints<double> constraints;

    using SystemMatrix = ADR_Operator<dim, degree_finite_element, double>;
    SystemMatrix system_matrix;

    // MG precond (spesso conviene usare operatore diffusione+reazione separato,
    // ma per iniziare puoi anche riusare quello e “disabilitare beta” nei livelli)
    MGConstrainedDoFs mg_constrained_dofs;
    using LevelMatrixType = ADR_Operator<dim, degree_finite_element, float>;
    MGLevelObject<LevelMatrixType> mg_matrices;

    LinearAlgebra::distributed::Vector<double> solution, system_rhs;
    double setup_time;
    dealii::ConditionalOStream time_details;
    dealii::ConditionalOStream pcout;
    std::set<dealii::types::boundary_id> dirichlet_ids ={1};
    std::set<dealii::types::boundary_id> neumann_ids = {};
    bool use_advection = false; 

  };
}
