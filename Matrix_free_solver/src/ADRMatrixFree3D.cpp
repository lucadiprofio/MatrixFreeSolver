#include "ADRMatrixFree3D.hpp"

#include <deal.II/base/timer.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

namespace Project
{
  using namespace dealii;

  ADRProblem::ADRProblem()
#ifdef DEAL_II_WITH_P4EST
    : triangulation(MPI_COMM_WORLD,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
#else
    : triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
#endif
    , fe(degree_finite_element)
    , dof_handler(triangulation)
    , setup_time(0.)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    , time_details(std::cout, false &&
                 dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
  {}

  void ADRProblem::setup_system()
  {
    Timer time;
    setup_time = 0;
    Mu<dim>    mu_fun;
    Beta<dim>  beta_fun;
    Gamma<dim> gamma_fun;
    DirichletBC<dim> g;

    {
      system_matrix.clear();
      mg_matrices.clear_elements();
 
      dof_handler.distribute_dofs(fe);
      dof_handler.distribute_mg_dofs();
 
      pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
 
      constraints.clear();
      constraints.reinit(DoFTools::extract_locally_relevant_dofs(dof_handler));
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);

      for (const auto id : dirichlet_ids)
      dealii::VectorTools::interpolate_boundary_values(mapping, dof_handler, id, g, constraints);
      
      constraints.close();
    }
    setup_time += time.wall_time();
    time_details << "Distribute DoFs & B.C.     (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << 's' << std::endl;
    time.restart();
    {
      {
        typename MatrixFree<dim, double>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme =
          MatrixFree<dim, double>::AdditionalData::none;
        additional_data.mapping_update_flags =
          (update_gradients | update_JxW_values | update_quadrature_points);
        std::shared_ptr<MatrixFree<dim, double>> system_mf_storage(
          new MatrixFree<dim, double>());
        system_mf_storage->reinit(mapping,
                                  dof_handler,
                                  constraints,
                                  QGauss<1>(fe.degree + 1),
                                  additional_data);
        system_matrix.initialize(system_mf_storage);
      }
 
      system_matrix.evaluate_coefficients(mu_fun, beta_fun, gamma_fun);
 
      system_matrix.initialize_dof_vector(solution);
      system_matrix.initialize_dof_vector(system_rhs);
    }
    setup_time += time.wall_time();
    time_details << "Setup matrix-free system   (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << 's' << std::endl;
    time.restart();
 
    {
      const unsigned int nlevels = triangulation.n_global_levels();
      mg_matrices.resize(0, nlevels - 1);
 
      const std::set<types::boundary_id> dirichlet_boundary_ids = dirichlet_ids;
      mg_constrained_dofs.initialize(dof_handler);
      mg_constrained_dofs.make_zero_boundary_constraints(
        dof_handler, dirichlet_boundary_ids);
 
      for (unsigned int level = 0; level < nlevels; ++level)
        {
          AffineConstraints<double> level_constraints;
            level_constraints.reinit( DoFTools::extract_locally_relevant_level_dofs(dof_handler, level));
          for (const types::global_dof_index dof_index :
               mg_constrained_dofs.get_boundary_indices(level))
            level_constraints.add_line(dof_index);
          level_constraints.close();
 
          typename MatrixFree<dim, float>::AdditionalData additional_data;
          additional_data.tasks_parallel_scheme =
            MatrixFree<dim, float>::AdditionalData::none;
          additional_data.mapping_update_flags =
            (update_gradients | update_JxW_values | update_quadrature_points);
          additional_data.mg_level = level;
          std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level =
            std::make_shared<MatrixFree<dim, float>>();
          mg_mf_storage_level->reinit(mapping,
                                      dof_handler,
                                      level_constraints,
                                      QGauss<1>(fe.degree + 1),
                                      additional_data);
 
          mg_matrices[level].initialize(mg_mf_storage_level,
                                        mg_constrained_dofs,
                                        level);
          mg_matrices[level].evaluate_coefficients(mu_fun, beta_fun, gamma_fun);
        }
    }
    setup_time += time.wall_time();
    time_details << "Setup matrix-free levels   (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << 's' << std::endl;
  }
 

  void ADRProblem::RHS()
  {
    Timer time;
 
    system_rhs = 0;

    WellKnownTerm<dim> f; 

    FEEvaluation<dim, degree_finite_element> phi(
      *system_matrix.get_matrix_free());

    for (unsigned int cell = 0;
         cell < system_matrix.get_matrix_free()->n_cell_batches();
         ++cell)
      {
        phi.reinit(cell);
        for (const unsigned int q : phi.quadrature_point_indices())
          {
            const auto x_q = phi.quadrature_point(q);
            phi.submit_value(f.template value_t<dealii::VectorizedArray<double>>(x_q),q);
          }
        phi.integrate(EvaluationFlags::values);
        phi.distribute_local_to_global(system_rhs);
      }

    NeumannBC<dim> h;

   FEFaceEvaluation<dim, degree_finite_element> phi_face(
    *system_matrix.get_matrix_free(), true /* boundary */);

    for (unsigned int face = 0;
       face < system_matrix.get_matrix_free()->n_boundary_face_batches();
       ++face)
    {
      phi_face.reinit(face);

      const auto boundary_id = phi_face.boundary_id();
      if (neumann_ids.find(boundary_id) == neumann_ids.end())
        continue;

      for (const unsigned int q : phi_face.quadrature_point_indices())
        {
          const auto x_q = phi_face.quadrature_point(q);
          phi_face.submit_value(h.template value_t<dealii::VectorizedArray<double>>(x_q), q);
        }

      phi_face.integrate(EvaluationFlags::values);
      phi_face.distribute_local_to_global(system_rhs);
    }

    system_rhs.compress(VectorOperation::add);
 
    setup_time += time.wall_time();
    time_details << "Assemble right hand side   (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << 's' << std::endl;
  }
 
 
  void ADRProblem::solve()
  {
    Timer                            time;
    MGTransferMatrixFree<dim, float> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);
    setup_time += time.wall_time();
    time_details << "MG build transfer time     (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << "s\n";
    time.restart();
 
    using SmootherType =
      PreconditionChebyshev<LevelMatrixType,
                            LinearAlgebra::distributed::Vector<float>>;
    mg::SmootherRelaxation<SmootherType,
                           LinearAlgebra::distributed::Vector<float>>
                                                         mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels();
         ++level)
      {
        smoother_data[level].smoothing_range     = 15.;
        smoother_data[level].degree              = 5;
        smoother_data[level].eig_cg_n_iterations = 10;
        mg_matrices[level].compute_diagonal();
        smoother_data[level].preconditioner =
          mg_matrices[level].get_matrix_diagonal_inverse();
      }
    mg_smoother.initialize(mg_matrices, smoother_data);
 
    MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>>
      mg_coarse;
    mg_coarse.initialize(mg_smoother);
 
    mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(
      mg_matrices);
 
    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>
      mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels();
         ++level)
      mg_interface_matrices[level].initialize(mg_matrices[level]);
    mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_interface(
      mg_interface_matrices);
 
    Multigrid<LinearAlgebra::distributed::Vector<float>> mg(
      mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);
 
    PreconditionMG<dim,
                   LinearAlgebra::distributed::Vector<float>,
                   MGTransferMatrixFree<dim, float>>
      preconditioner(dof_handler, mg, mg_transfer);

    time.reset();
    time.start();
    constraints.set_zero(solution);
    setup_time += time.wall_time();

    time_details << "MG build smoother time     (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << "s\n";
    pcout << "Total setup time               (wall) " << setup_time << "s\n";
 
    SolverControl solver_control(1000, 1e-10 * system_rhs.l2_norm());
    if (!use_advection){
      SolverCG<LinearAlgebra::distributed::Vector<double>> solver(solver_control);
      solver.solve(system_matrix, solution, system_rhs, preconditioner);
    }
    else{
      SolverGMRES<LinearAlgebra::distributed::Vector<double>>::AdditionalData gmres_data;
      gmres_data.max_n_tmp_vectors = 30; // restart

      SolverGMRES<LinearAlgebra::distributed::Vector<double>> solver(solver_control, gmres_data);
      solver.solve(system_matrix, solution, system_rhs, preconditioner);
    }
 
    constraints.distribute(solution);
 
    pcout << "Time solve (" << solver_control.last_step() << " iterations)"
          << (solver_control.last_step() < 10 ? "  " : " ") << "(CPU/wall) "
          << time.cpu_time() << "s/" << time.wall_time() << "s\n";
  }
 

  void ADRProblem::output_results(const unsigned int cycle) const
  {
    Timer time;
    if (triangulation.n_global_active_cells() > 1000000)
      return;
 
    DataOut<dim> data_out;
 
    solution.update_ghost_values();
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches(mapping);
 
    DataOutBase::VtkFlags flags;
    flags.compression_level = DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(flags);
    data_out.write_vtu_with_pvtu_record(
      "./", "solution", cycle, MPI_COMM_WORLD, 3);
 
    time_details << "Time write output          (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << "s\n";
  }
 

  void ADRProblem::run()
  {
    {
      const unsigned int n_vect_doubles = VectorizedArray<double>::size();
      const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;
 
      pcout << "Vectorization over " << n_vect_doubles
            << " doubles = " << n_vect_bits << " bits ("
            << Utilities::System::get_current_vectorization_level() << ')'
            << std::endl;
    }
 
    for (unsigned int cycle = 0; cycle < 9 - dim; ++cycle)
      {
        pcout << "Cycle " << cycle << std::endl;
 
        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation, 0., 1.);
            triangulation.refine_global(3 - dim);
          }
        triangulation.refine_global(1);
        set_boundary_ids();
        setup_system();
        RHS();
        solve();
        output_results(cycle);
        pcout << std::endl;
      };
  }

  void ADRProblem::set_boundary_ids()
  {
  for (const auto &cell : triangulation.active_cell_iterators())
    for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
        {
           cell->face(f)->set_boundary_id(1);
        }
  }

}