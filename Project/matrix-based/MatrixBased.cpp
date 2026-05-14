#include "MatrixBased.hpp"

namespace MatrixBased {

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

  dof_handler.reinit(triangulation);
  dof_handler.distribute_dofs(fe);

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);

  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  // Dirichlet map
  std::map<types::boundary_id, const Function<dim> *> dirichlet_map;
  dirichlet_map[2] = &dirichlet_function_walls;
  dirichlet_map[4] = &dirichlet_function_inlet;
  dirichlet_map[5] = &dirichlet_function_walls;

  VectorTools::interpolate_boundary_values(mapping, dof_handler, dirichlet_map, constraints);
  constraints.close();

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp, locally_owned_dofs, mpi_communicator, locally_relevant_dofs);
  system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
}


template <unsigned int dim, unsigned int degree>
void SolverClass<dim, degree>::assemble_system() {
  TimerOutput::Scope t(computing_timer, "Assembly");

  const QGauss<dim> quadrature_formula(fe.degree + 1);
  const QGauss<dim - 1> quadrature_face_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(fe, quadrature_face_formula,
                                   update_quadrature_points | update_values |
                                       update_JxW_values |
                                       update_normal_vectors);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points = quadrature_formula.size();
  const unsigned int n_face_q_points = quadrature_face_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> mu_values(n_q_points);
  std::vector<double> sigma_values(n_q_points);
  std::vector<double> rhs_values(n_q_points);
  std::vector<Tensor<1, dim>> b_values(n_q_points, Tensor<1, dim>());
  std::vector<double> neumann_values(n_face_q_points);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);

    cell_matrix = 0.;
    cell_rhs = 0.;

    // Evaluate coefficients
    mu_function.value_list(fe_values.get_quadrature_points(), mu_values);
    sigma_function.value_list(fe_values.get_quadrature_points(), sigma_values);
    rhs_function.value_list(fe_values.get_quadrature_points(), rhs_values);
    b_function.tensor_value_list(fe_values.get_quadrature_points(), b_values);

    for (unsigned int q = 0; q < n_q_points; ++q) {

      // SUPG parameter
      const double h = cell->diameter();
      const double vel_norm = b_values[q].norm();

      double tau_inv_sq = (2.0 * vel_norm / h) * (2.0 * vel_norm / h) +
                          9.0 * (4.0 * mu_values[q] / (h * h)) *
                              (4.0 * mu_values[q] / (h * h)) +
                          (sigma_values[q]) * (sigma_values[q]);

      // double tau = 1.0 / std::sqrt(tau_inv_sq); // SUPG on
      double tau = 0.0 / std::sqrt(tau_inv_sq); // SUPG off
      if (tau_inv_sq < 1e-12)
        tau = 0.0;

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        const double phi_i = fe_values.shape_value(i, q);
        const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q);

        const double supg_test_i = tau * (b_values[q] * grad_phi_i);

        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          const Tensor<1, dim> grad_phi_j = fe_values.shape_grad(j, q);
          const double phi_j = fe_values.shape_value(j, q);

          cell_matrix(i, j) += (
            mu_values[q] * grad_phi_j * grad_phi_i
            + b_values[q] * grad_phi_j * phi_i
            + sigma_values[q] * phi_j * phi_i
            + ((b_values[q] * grad_phi_j) + sigma_values[q] * phi_j) * supg_test_i // SUPG
          ) * fe_values.JxW(q);
        }

        double rhs_galerkin = (rhs_values[q]) * phi_i;
        double rhs_supg = (rhs_values[q]) * supg_test_i;
        cell_rhs(i) += (rhs_galerkin + rhs_supg) * fe_values.JxW(q);
      }
    }

    if (cell->at_boundary()) {
      // Iterates on all cell faces
      for (const auto &face : cell->face_iterators()) // Iterate over boundary faces
        if (face->at_boundary()) {
          const types::boundary_id id = face->boundary_id();

          if (neumann_boundary_ids.count(id)) {
            fe_face_values.reinit(cell, face); // Initialize FE values on this face

            neumann_function.value_list(fe_face_values.get_quadrature_points(), neumann_values);

            for (const unsigned int q : fe_face_values.quadrature_point_indices()) {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                cell_rhs(i) += neumann_values[q] * fe_face_values.shape_value(i, q) * fe_face_values.JxW(q);

              // TODO: add neumann contribution on lhs (see weak formulation of
              // the problem)
            }
          }
        }
    }

    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
  }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}


template <unsigned int dim, unsigned int degree>
void SolverClass<dim, degree>::solve() {
  TimerOutput::Scope t(computing_timer, "Solve");

  computing_timer.enter_subsection("Solve: Preconditioner setup");

  // Preconditioner and smoother setup
  PrecType preconditioner;
  PrecType::AdditionalData data;
  data.elliptic = true;
  data.smoother_type = "Chebyshev";
  data.smoother_sweeps = 2;

  preconditioner.initialize(system_matrix, data);
  
  computing_timer.leave_subsection("Solve: Preconditioner setup");

  track_memory();

  // Measure 1 V-Cycle
  {
    TimerOutput::Scope timing(computing_timer, "Solve: 1 multigrid V-cycle");
    preconditioner.vmult(solution, system_rhs);
  }

  solution = 0.;

  SolverControl solver_control(1000, 1e-12 + 1e-8 * system_rhs.l2_norm());
  {
    TimerOutput::Scope timing(computing_timer, "Solve: CG/GMRES");

    // SolverType::AdditionalData gmres_data(false, 50);
    SolverType solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
  }

  constraints.distribute(solution);
  
  pcout << "\tNumber of CG/GMRES iterations: " << solver_control.last_step() << std::endl;
}


template <unsigned int dim, unsigned int degree>
void SolverClass<dim, degree>::refine_grid(const VectorType &locally_relevant_solution) {
  TimerOutput::Scope t(computing_timer, "Refine");

  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

  // Neumann map
  std::map<types::boundary_id, const Function<dim> *> neumann_map;
  // neumann_map[5] = &neumann_function;

  KellyErrorEstimator<dim>::estimate(
      dof_handler, QGauss<dim - 1>(fe.degree + 1), neumann_map,
      locally_relevant_solution, estimated_error_per_cell);

  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
      triangulation, estimated_error_per_cell, 0.3, 0.03);
  triangulation.execute_coarsening_and_refinement();
}


template <unsigned int dim, unsigned int degree>
void SolverClass<dim, degree>::output_results(const unsigned int cycle, const VectorType &locally_relevant_solution) {
  TimerOutput::Scope timing(computing_timer, "Output results");

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(locally_relevant_solution, "solution");

  data_out.build_patches();

  DataOutBase::VtkFlags flags;
  flags.compression_level = DataOutBase::VtkFlags::best_speed;
  data_out.set_flags(flags);

  data_out.write_vtu_with_pvtu_record("../out/", "solution", cycle,
                                      mpi_communicator, 2, 1);
}


template <unsigned int dim, unsigned int degree>
void SolverClass<dim, degree>::estimate_error(const unsigned int cycle, const VectorType &locally_relevant_solution) {
  Vector<float> difference_per_cell(triangulation.n_active_cells());

  // Fill difference_per_cell and compute L2 error
  VectorTools::integrate_difference(
      dof_handler, locally_relevant_solution, exact_solution,
      difference_per_cell, QGauss<dim>(fe.degree + 1), VectorTools::L2_norm);
  const double L2_error = VectorTools::compute_global_error(
      triangulation, difference_per_cell, VectorTools::L2_norm);

  // Fill difference_per_cell and compute H1 seminorm error
  VectorTools::integrate_difference(dof_handler, locally_relevant_solution,
                                    exact_solution, difference_per_cell,
                                    QGauss<dim>(fe.degree + 1),
                                    VectorTools::H1_seminorm);
  const double H1_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::H1_seminorm);

  const unsigned int n_active_cells = triangulation.n_active_cells();
  const unsigned int n_dofs = dof_handler.n_dofs();

  // Record all quantities in the convergence table
  convergence_table.add_value("cycles", cycle);
  convergence_table.add_value("cells", n_active_cells);
  convergence_table.add_value("dofs", n_dofs);
  convergence_table.add_value("L2", L2_error);
  convergence_table.add_value("H1", H1_error);
}

/// Formats and writes the convergence and error tables to LaTeX files.
template <unsigned int dim, unsigned int degree>
void SolverClass<dim, degree>::print_tables() {
  convergence_table.set_precision("L2", 3);
  convergence_table.set_precision("H1", 3);

  convergence_table.set_scientific("L2", true);
  convergence_table.set_scientific("H1", true);

  convergence_table.set_tex_caption("cells", "\\# cells");
  convergence_table.set_tex_caption("dofs", "\\# dofs");
  convergence_table.set_tex_caption("L2", "@f$L^2@f$-error");
  convergence_table.set_tex_caption("H1", "@f$H^1@f$-error");

  convergence_table.set_tex_format("cells", "r");
  convergence_table.set_tex_format("dofs", "r");

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
    convergence_table.add_column_to_supercolumn("cells", "n cells");

    std::vector<std::string> new_order;
    new_order.emplace_back("n cells");
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
    pcout << "===========================================" << std::endl;
    pcout << "Polynomial degree " << degree << std::endl;
    pcout << "Running with " << n_mpi_procs << " MPI process(es)" << std::endl;
    pcout << "===========================================" << std::endl;
  }

  init_mesh();

  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle) {
    pcout << "Cycle " << cycle << ':' << std::endl;

    setup_system();

    if (dof_handler.n_dofs() > 200000000)
      break;

    pcout << "\tNumber of active cells: " << triangulation.n_global_active_cells() << std::endl;
    pcout << "\tNumber of dofs: " << dof_handler.n_dofs() << std::endl;

    assemble_system();
    solve();

    {
      // Create relevant dof vector for post-processing
      VectorType relevant_solution;
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

} // namespace MatrixBased
