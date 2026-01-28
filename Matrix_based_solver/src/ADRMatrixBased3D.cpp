#include "ADRMatrixBased3D.hpp"

void
ADRMatrixBased3D::setup()
{
  std::cout << "===============================================" << std::endl;

  // Create the mesh.
  {
    std::cout << "Initializing the mesh" << std::endl;

    GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
    triangulation.refine_global(3);   
    set_boundary_ids();

    std::cout << "  Number of elements = " << triangulation.n_active_cells()
              << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    std::cout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_Q<dim>>(r);

    std::cout << "  Degree                     = " << fe->degree << std::endl;
    std::cout << "  DoFs per cell              = " << fe->dofs_per_cell
              << std::endl;

    quadrature = std::make_unique<QGauss<dim>>(r + 1);

    std::cout << "  Quadrature points per cell = " << quadrature->size()
              << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    std::cout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(triangulation);
    dof_handler.distribute_dofs(*fe);

    std::cout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    std::cout << "Initializing the linear system" << std::endl;

    std::cout << "  Initializing the sparsity pattern" << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    std::cout << "  Initializing the system matrix" << std::endl;
    system_matrix.reinit(sparsity_pattern);

    std::cout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(dof_handler.n_dofs());
    std::cout << "  Initializing the solution vector" << std::endl;
    solution.reinit(dof_handler.n_dofs());
  }
}

void
ADRMatrixBased3D::assemble()
{
  std::cout << "===============================================" << std::endl;

  std::cout << "  Assembling the linear system" << std::endl;

  // Number of local DoFs for each element.
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(*fe,
                                 QGauss<dim-1>(r+1),
                                 update_values | update_quadrature_points |
                                   update_normal_vectors | update_JxW_values);

  const unsigned int n_qf = fe_face_values.n_quadrature_points;

  // Local matrix and vector.
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Reset the global matrix and vector, just in case.
  system_matrix = 0.0;
  system_rhs    = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
           const auto  x_q      = fe_values.quadrature_point(q);

          const double mu_loc    = mu.value(x_q);
          const double gamma_loc = gamma.value(x_q);
          const double f_loc     = f.value(x_q);     
          const Tensor<1,dim> beta_loc = beta.value(x_q);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) += mu_loc *                     //
                                       fe_values.shape_grad(i, q) * //
                                       fe_values.shape_grad(j, q) * //
                                       fe_values.JxW(q);

                  cell_matrix(i, j) += gamma_loc *                   //
                                       fe_values.shape_value(i, q) * //
                                       fe_values.shape_value(j, q) * //
                                       fe_values.JxW(q);

                  cell_matrix(i, j) += -(beta_loc * fe_values.shape_grad(i, q)) *
                               fe_values.shape_value(j, q) *
                               fe_values.JxW(q);
                }

              cell_rhs(i) += f_loc *                       //
                             fe_values.shape_value(i, q) * //
                             fe_values.JxW(q);
            }
       }

      for (unsigned int face = 0; face < cell->n_faces(); ++face)
        if (cell->face(face)->at_boundary())
          {
            const auto bid = cell->face(face)->boundary_id();

            if (neumann_ids.count(bid))
              {
                fe_face_values.reinit(cell, face);

                for (unsigned int q = 0; q < n_qf; ++q)
                  {
                    const auto xq = fe_face_values.quadrature_point(q);
                    const auto n  = fe_face_values.normal_vector(q);
                    const double mu_loc = mu.value(xq);
                    const double h_loc  = neumann_bc.value(xq);
                    const auto beta_loc = beta.value(xq);
                    const double beta_n = beta_loc * n;   

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                
                        cell_rhs(i) += mu_loc* h_loc *
                                      fe_face_values.shape_value(i, q) *
                                      fe_face_values.JxW(q);

                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                          cell_matrix(i, j) += beta_n *
                                              fe_face_values.shape_value(i, q) *
                                              fe_face_values.shape_value(j, q) *
                                              fe_face_values.JxW(q);
                      }
                  }
              }
          }


      cell->get_dof_indices(dof_indices);

      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  // Dirichlet boundary conditions.
  {
    std::map<types::global_dof_index, double> boundary_values;

    for (const auto bid : dirichlet_ids)
    VectorTools::interpolate_boundary_values(dof_handler,
                                           bid,
                                           dirichlet_bc,
                                           boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, true);
  }
}

void
ADRMatrixBased3D::solve()
{
  std::cout << "===============================================" << std::endl;

  // PreconditionIdentity preconditioner;

  // PreconditionJacobi preconditioner;
  // preconditioner.initialize(system_matrix);

  // PreconditionSOR preconditioner;
  // preconditioner.initialize(
  //   system_matrix,
  //   PreconditionSOR<SparseMatrix<double>>::AdditionalData(1.0));

  PreconditionSSOR preconditioner;
  preconditioner.initialize(
    system_matrix, PreconditionSSOR<SparseMatrix<double>>::AdditionalData(1.0));

  ReductionControl solver_control(/* maxiter = */ 10000,
                                  /* tolerance = */ 1.0e-16,
                                  /* reduce = */ 1.0e-6);

  std::cout << "  Solving the linear system" << std::endl;

  if (use_advection) // oppure un check: beta != 0
  {
    SolverGMRES<Vector<double>> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    std::cout << "  " << solver_control.last_step() << " GMRES iterations\n";
  }
  else
  {
    SolverCG<Vector<double>> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    std::cout << "  " << solver_control.last_step() << " CG iterations\n";
  }
}

void ADRMatrixBased3D::output() const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "u");
  data_out.build_patches();

  std::ofstream out("solution.vtu");
  data_out.write_vtu(out);
}

double
ADRMatrixBased3D::compute_error(const VectorTools::NormType &norm_type,
                                 const Function<dim> &exact_solution) const
{
  const QGaussSimplex<dim> quadrature_error(r + 2);

  FE_SimplexP<dim> fe_linear(1);
  MappingFE        mapping(fe_linear);

  Vector<double> error_per_cell(triangulation.n_active_cells());
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  const double error =
    VectorTools::compute_global_error(triangulation, error_per_cell, norm_type);

  return error;
}

void ADRMatrixBased3D::set_boundary_ids()
  {
  for (const auto &cell : triangulation.active_cell_iterators())
    for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
        {
           cell->face(f)->set_boundary_id(1);
        }
  }
