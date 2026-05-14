#include "MatrixFree.hpp"

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/lac/precondition.h>

#include <deal.II/matrix_free/evaluation_flags.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>


namespace MtxFree {
using namespace dealii;

template <unsigned int dim, unsigned int degree>
void SolverClass<dim, degree>::init_mesh() {
  TimerOutput::Scope t(computing_timer, "Generate mesh");

  Point<dim> p1(0, 0, 0);
  Point<dim> p2(1, 1, 1);
  std::vector<unsigned int> subdivisions(dim, 4);
  GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2, true);

  // Original boundary IDs assigned by GridGenerator:
  // X-min (Inlet) = 0
  // X-max (Outlet) = 1
  // Y-min (Wall) = 2
  // Y-max (Wall) = 3
  // Z-min (Wall) = 4 (3D)
  // Z-max (Wall) = 5 (3D)
  const unsigned int boundary_id_walls = 2;
  const unsigned int boundary_id_inlet = 4;
  const unsigned int boundary_id_outlet = 5;

  // Remap all wall faces to a single ID for convenience:
  for (const auto &cell : triangulation.active_cell_iterators()) {
    if (cell->is_locally_owned())
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        if (cell->face(f)->at_boundary()) {
          const unsigned int id = cell->face(f)->boundary_id();
          if (id != boundary_id_inlet && id != boundary_id_outlet)
            cell->face(f)->set_boundary_id(boundary_id_walls);
        }
  }

  triangulation.refine_global(2);
}


template <unsigned int dim, unsigned int degree>
void SolverClass<dim, degree>::setup_system() {
  TimerOutput::Scope t(computing_timer, "Setup");

  system_matrix.clear();
  mg_matrices.clear_elements();

  dof_handler.distribute_dofs(fe);

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  // Dirichlet boundary condition handling via AffineConstraints
  std::map<types::boundary_id, const Function<dim> *> dirichlet_map;
  dirichlet_map[2] = &dirichlet_function_walls;
  dirichlet_map[4] = &dirichlet_function_inlet;
  dirichlet_map[5] = &dirichlet_function_walls;

  VectorTools::interpolate_boundary_values(mapping, dof_handler, dirichlet_map, constraints);
  constraints.close();

  // init MatrixFree
  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
  additional_data.mapping_update_flags =
      (update_values | update_gradients | update_JxW_values |
       update_quadrature_points); 

  additional_data.mapping_update_flags_boundary_faces = 
    (update_values | update_JxW_values | update_quadrature_points);

  std::shared_ptr<MatrixFree<dim, double>> system_mf_storage = std::make_shared<MatrixFree<dim, double>>();
  system_mf_storage->reinit(mapping, dof_handler, constraints, QGauss<1>(fe.degree + 1), additional_data);
  system_matrix.initialize(system_mf_storage);

  system_matrix.setup_coefficients(mu_function, b_function, sigma_function);

  // Must use initialize_dof_vector so that locally_relevant_solution gets the
  // MatrixFree-internal partitioner (owned + the exact ghost DOFs required for
  // cell-based read operations). Using a custom reinit with
  // locally_relevant_dofs gives a *different* partitioner and causes a segfault
  // in parallel because read_dof_values_plain() uses MatrixFree's cached ghost
  // offsets to index into the vector's memory — those offsets are wrong if the
  // partitioner doesn't match.
  system_matrix.initialize_dof_vector(solution);
  system_matrix.initialize_dof_vector(system_rhs);
}


template <unsigned int dim, unsigned int degree>
void SolverClass<dim, degree>::setup_multigrid() {
  TimerOutput::Scope timing(computing_timer, "Setup multigrid");

  dof_handler.distribute_mg_dofs();

  mg_constrained_dofs.clear();
  mg_constrained_dofs.initialize(dof_handler);

  // only Dirichlet
  const std::set<types::boundary_id> boundary_ids = {2, 4, 5};
  mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, boundary_ids);

  const unsigned int nlevels = triangulation.n_global_levels();
  mg_matrices.resize(0, nlevels - 1);

  for (unsigned int level = 0; level < nlevels; ++level) {
    IndexSet locally_relevant_level_dofs;
    DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, locally_relevant_level_dofs);

    AffineConstraints<double> level_constraints;
    level_constraints.clear();
    level_constraints.reinit(locally_relevant_level_dofs);
    for (const types::global_dof_index dof_index : mg_constrained_dofs.get_boundary_indices(level))
      level_constraints.add_line(dof_index); // Adds a constraint in AffCon: x_i = 0
    level_constraints.close();

    typename MatrixFree<dim, float>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = MatrixFree<dim, float>::AdditionalData::none;
    additional_data.mapping_update_flags =
        (update_values | update_gradients | update_JxW_values |
         update_quadrature_points);
    additional_data.mg_level = level;

    std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level = std::make_shared<MatrixFree<dim, float>>();
    mg_mf_storage_level->reinit(mapping, dof_handler, level_constraints, QGauss<1>(fe.degree + 1), additional_data);

    mg_matrices[level].initialize(mg_mf_storage_level, mg_constrained_dofs, level);
    mg_matrices[level].setup_coefficients(mu_function, b_function, sigma_function);

    mg_matrices[level].compute_diagonal();
  }
}


template <unsigned int dim, unsigned int degree>
void SolverClass<dim, degree>::assemble_rhs() {
  TimerOutput::Scope timing(computing_timer, "Assemble right-hand side");

  solution = 0.;
  constraints.distribute(solution);
  solution.update_ghost_values();

  system_rhs = 0.;
  FEEvaluation<dim, degree, degree + 1, 1, double> phi(*system_matrix.get_matrix_free());

  const unsigned int n_batches = system_matrix.get_matrix_free()->n_cell_batches();
  for (unsigned int cell = 0; cell < n_batches; ++cell) {
    phi.reinit(cell);
    phi.read_dof_values_plain(solution); // plain to allow inhomogeneous Dirichlet boundary conditions

    phi.evaluate(EvaluationFlags::gradients | EvaluationFlags::values);

    for (unsigned int q = 0; q < phi.n_q_points; ++q) {
      const auto x_q = phi.quadrature_point(q);
      const auto u = phi.get_value(q);
      const auto grad_u = phi.get_gradient(q);

      const auto mu_val = mu_function.value(x_q);
      const auto beta_val = b_function.value(x_q);
      const auto gamma_val = sigma_function.value(x_q);

      const auto f_val = rhs_function.value(x_q);

      // SUPG parameter
      const auto inv_jac = phi.inverse_jacobian(q);
      VectorizedArray<double> max_inv_jac = 0.0;
      for (unsigned int i = 0; i < dim; ++i)
        max_inv_jac = std::max(max_inv_jac, inv_jac[i].norm());
      const auto h = 1.0 / max_inv_jac;
      const auto vel_norm = beta_val.norm();

      const auto tau_inv_sq =
        (2.0 * vel_norm / h) * (2.0 * vel_norm / h) +
        9.0 * (4.0 * mu_val / (h * h)) * (4.0 * mu_val / (h * h)) +
        (gamma_val) * (gamma_val);

      // const auto tau = 1.0 / std::sqrt(tau_inv_sq); // SUPG on
      (void) tau_inv_sq;
      const auto tau = 0. / std::sqrt(tau_inv_sq); // SUPG off

      const auto strong_residual = beta_val * grad_u + gamma_val * u;

      phi.submit_gradient(
        -mu_val * grad_u
        - (tau * strong_residual) * beta_val
        + (tau * f_val) * beta_val, //  SUPG term
      q);
      phi.submit_value(-strong_residual + f_val, q);
    }

    phi.integrate(EvaluationFlags::gradients | EvaluationFlags::values);
    phi.distribute_local_to_global(system_rhs);
  }

  if (!neumann_boundary_ids.empty()) {
    FEFaceEvaluation<dim, degree, degree + 1, 1, double> phi_face(*system_matrix.get_matrix_free(), true);
    const unsigned int n_face_batches = system_matrix.get_matrix_free()->n_boundary_face_batches();
    for (unsigned int face = 0; face < n_face_batches; ++face) {
      const types::boundary_id id = system_matrix.get_matrix_free()->get_boundary_id(face);

      if (!neumann_boundary_ids.count(id))
        continue;

      phi_face.reinit(face);
      for (unsigned int q = 0; q < phi_face.n_q_points; ++q) {
        const auto x_q = phi_face.quadrature_point(q);
        phi_face.submit_value(neumann_function.value(x_q), q);
      }
      phi_face.integrate(EvaluationFlags::values);
      phi_face.distribute_local_to_global(system_rhs);
    }
  }

  system_rhs.compress(VectorOperation::add);
}


template <unsigned int dim, unsigned int degree>
void SolverClass<dim, degree>::solve() {
  TimerOutput::Scope t(computing_timer, "Solve");

  computing_timer.enter_subsection("Solve: Preconditioner setup");

  MGTransferMatrixFree<dim, float> mg_transfer(mg_constrained_dofs);
  mg_transfer.build(dof_handler);

  using SmootherType = PreconditionChebyshev<MatrixFreeLevelMatrix, MatrixFreeLevelVector>;
  using MGSmootherType = MGSmootherPrecondition<MatrixFreeLevelMatrix, SmootherType, MatrixFreeLevelVector>;

  // Additional Chebyshev data
  const unsigned int n_levels = triangulation.n_global_levels();
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
  smoother_data.resize(0, n_levels - 1);
  for (unsigned int level = 0; level < n_levels; ++level) {

    if (level > 0) {
      smoother_data[level].smoothing_range = 20.0; // Chebyshev smoothing range (alpha)
      smoother_data[level].degree = 2; // Number of Chebyshev iterations per sweep
  
      // Number of CG iterations used to estimate the largest eigenvalue,
      // which is needed to set the Chebyshev polynomial coefficients.
      smoother_data[level].eig_cg_n_iterations = 20;
    }
    else {
      smoother_data[0].smoothing_range = 2e-2;
      smoother_data[0].degree = numbers::invalid_unsigned_int;
      smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
    }

    smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
  }

  // Setup smoother MG
  MGSmootherType mg_smoother;
  mg_smoother.initialize(mg_matrices, smoother_data);

  // Coarse grid solver
  MGCoarseGridApplySmoother<MatrixFreeLevelVector> mg_coarse;
  mg_coarse.initialize(mg_smoother);

  mg::Matrix<MatrixFreeLevelVector> mg_matrix(mg_matrices);

  // Interface operators handle the coupling between coarse and fine levels
  MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<MatrixFreeLevelMatrix>> mg_interface_matrices;
  mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
  for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    mg_interface_matrices[level].initialize(mg_matrices[level]);
  mg::Matrix<MatrixFreeLevelVector> mg_interface(mg_interface_matrices);

  Multigrid<MatrixFreeLevelVector> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
  mg.set_edge_matrices(mg_interface, mg_interface);

  PreconditionMG<dim, MatrixFreeLevelVector, MGTransferMatrixFree<dim, float>> preconditioner(dof_handler, mg, mg_transfer);

  computing_timer.leave_subsection("Solve: Preconditioner setup");

  track_memory();

  // Measure 1 V-Cycle
  {
    TimerOutput::Scope timing(computing_timer, "Solve: 1 multigrid V-cycle");
    preconditioner.vmult(solution, system_rhs);
  }

  solution = 0.;
  constraints.distribute(solution);

  SolverControl solver_control(1000, 1e-12 + 1e-8 * system_rhs.l2_norm());
  {
    TimerOutput::Scope timing(computing_timer, "Solve: GMRES");

    SolverType<MatrixFreeActiveVector>::AdditionalData gmres_data(false, 50);
    SolverType<MatrixFreeActiveVector> solver(solver_control, gmres_data);
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
  }

  // this call sets the constrained DOFs to their prescribed Dirichlet values,
  // giving u = u_0 + u_D.
  constraints.distribute(solution);

  pcout << "\tNumber of GMRES iterations: " << solver_control.last_step() << std::endl;
}


template <unsigned int dim, unsigned int degree>
void SolverClass<dim, degree>::refine_grid(const MatrixFreeActiveVector &locally_relevant_solution) {
  TimerOutput::Scope t(computing_timer, "Refine");

  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

  // Neumann map
  std::map<types::boundary_id, const Function<dim> *> neumann_map;
  // neumann_map[5] = &neumann_function;

  KellyErrorEstimator<dim>::estimate(
      dof_handler, QGauss<dim - 1>(fe.degree + 1), neumann_map,
      locally_relevant_solution, estimated_error_per_cell);

  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(triangulation, estimated_error_per_cell, 0.3, 0.03);
  triangulation.execute_coarsening_and_refinement();
}


template <unsigned int dim, unsigned int degree>
void SolverClass<dim, degree>::output_results(const unsigned int cycle, const MatrixFreeActiveVector &locally_relevant_solution) {
  TimerOutput::Scope timing(computing_timer, "Output results");

  if (triangulation.n_global_active_cells() > 20000000)
    pcout << "WARNING: Over 20 million cells, output might be slow." << std::endl;

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(locally_relevant_solution, "solution");

  data_out.build_patches(mapping);

  DataOutBase::VtkFlags flags;
  flags.compression_level = DataOutBase::CompressionLevel::best_speed;
  data_out.set_flags(flags);

  data_out.write_vtu_with_pvtu_record("../out/", "solution", cycle, mpi_communicator, 2, 1);
}


template <unsigned int dim, unsigned int degree>
void SolverClass<dim, degree>::estimate_error(const unsigned int cycle, const MatrixFreeActiveVector &locally_relevant_solution) {
  Vector<float> difference_per_cell(triangulation.n_active_cells());

  // Fill difference_per_cell and compute L2 error
  VectorTools::integrate_difference(
    dof_handler,
    locally_relevant_solution,
    exact_solution,
    difference_per_cell,
    QGauss<dim>(fe.degree + 1),
    VectorTools::L2_norm
  );
  const double L2_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);

  // Fill difference_per_cell and compute H1 seminorm error
  VectorTools::integrate_difference(
    dof_handler,
    locally_relevant_solution,
    exact_solution,
    difference_per_cell,
    QGauss<dim>(fe.degree + 1),
    VectorTools::H1_seminorm
  );
  const double H1_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::H1_seminorm);

  const unsigned int n_active_cells = triangulation.n_global_active_cells();
  const unsigned int n_dofs = dof_handler.n_dofs();

  convergence_table.add_value("cycles", cycle);
  convergence_table.add_value("cells", n_active_cells);
  convergence_table.add_value("dofs", n_dofs);
  convergence_table.add_value("L2", L2_error);
  convergence_table.add_value("H1", H1_error);
}


template <unsigned int dim, unsigned int degree>
void SolverClass<dim, degree>::print_tables() {
  convergence_table.set_precision("L2", 3);
  convergence_table.set_precision("H1", 3);

  convergence_table.set_scientific("L2", true);
  convergence_table.set_scientific("H1", true);

  convergence_table.set_tex_caption("cells", "\\# cells");
  convergence_table.set_tex_caption("dofs", "\\# dofs");
  convergence_table.set_tex_caption("L2", "$L^2$-error");
  convergence_table.set_tex_caption("H1", "$H^1$-error");

  convergence_table.set_tex_format("cells", "r");
  convergence_table.set_tex_format("dofs", "r");

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
    convergence_table.add_column_to_supercolumn("cells", "n cells");

    std::vector<std::string> new_order;
    new_order.emplace_back("n cells");
    new_order.emplace_back("dofs");
    new_order.emplace_back("H1");
    new_order.emplace_back("L2");
    convergence_table.set_column_order(new_order);

    // Compute convergence rates
    convergence_table.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate_log2);

    // print table
    std::cout << std::endl;
    std::cout << "Convergence analysis:" << std::endl;
    convergence_table.write_text(std::cout);
    std::cout << std::endl;
  }
}


template <unsigned int dim, unsigned int degree>
void SolverClass<dim, degree>::track_memory() const {
  Utilities::System::MemoryStats stats;
  Utilities::System::get_memory_stats(stats);

  // Compute min, max, sum, and avg RSS memory across all MPI processes
  const double local_mem_gb = stats.VmRSS / (1024.0 * 1024.0);
  Utilities::MPI::MinMaxAvg mem_mpi =
      Utilities::MPI::min_max_avg(local_mem_gb, mpi_communicator);

  pcout << "\tMemory (RSS) [GB] -> "
        << "Sum: " << mem_mpi.sum << " | "
        << "Avg: " << mem_mpi.avg << " | "
        << "Max: " << mem_mpi.max << std::endl;
}


template <unsigned int dim, unsigned int degree>
void SolverClass<dim, degree>::run() {
  const unsigned int n_cycles = 4;

  {
    const unsigned int n_mpi_procs = Utilities::MPI::n_mpi_processes(mpi_communicator);
    const unsigned int n_vect_doubles = VectorizedArray<double>::size();
    const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;

    pcout << "===========================================" << std::endl;
    pcout << "Polynomial degree " << degree << std::endl;
    pcout << "Running with " << n_mpi_procs << " MPI process(es)" << std::endl;
    pcout << "Vectorization over " << n_vect_doubles
          << " doubles = " << n_vect_bits << " bits ("
          << Utilities::System::get_current_vectorization_level() << ')'
          << std::endl;
    pcout << "===========================================" << std::endl;
  }

  init_mesh();

  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle) {
    pcout << "Cycle " << cycle << ':' << std::endl;

    setup_system();
    setup_multigrid();

    pcout << "\tNumber of active cells: " << triangulation.n_global_active_cells() << std::endl;
    pcout << "\tNumber of dofs: " << dof_handler.n_dofs() << std::endl;

    assemble_rhs();
    solve();

    {
      // Create relevant dof vector for post-processing
      MatrixFreeActiveVector relevant_solution;
      relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
      relevant_solution = solution;
      relevant_solution.update_ghost_values();

      // output_results(cycle, relevant_solution);
      estimate_error(cycle, relevant_solution);

      if (cycle < n_cycles - 1)
        triangulation.refine_global(1);
        // refine_grid(relevant_solution);
    }

    computing_timer.print_summary();
    computing_timer.reset();

    pcout << std::endl;
  }

  print_tables();
}




// add others explicit instantiations if needed...
template class SolverClass<3, 1>;
template class SolverClass<3, 2>;
template class SolverClass<3, 3>;
template class SolverClass<3, 4>;
template class SolverClass<3, 5>;
template class SolverClass<3, 6>;
template class SolverClass<3, 7>;

} // namespace MtxFree
