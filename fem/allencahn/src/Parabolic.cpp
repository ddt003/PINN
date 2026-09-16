#include "Parabolic.hpp"

void
Parabolic::setup()
{
  TimerOutput::Scope t(computing_timer, "1. Setup system");
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    GridGenerator::subdivided_hyper_cube(mesh_serial, N, 0.0, 1.0, true);
    const std::string mesh_file_name = "mesh-" + std::to_string(N) + ".vtk";
    GridOut           grid_out;
    std::ofstream     grid_out_file(mesh_file_name);
    grid_out.write_vtk(mesh_serial, grid_out_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;

#ifdef NEUMANN
    quadrature_boundary = std::make_unique<QGaussSimplex<dim - 1>>(r + 1);

    std::cout << "  Quadrature points per boundary cell = "
              << quadrature_boundary->size() << std::endl;
#endif //NEUMANN

  }

  pcout << "-----------------------------------------------" << std::endl;

  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
    constraints.clear();
    constraints.reinit(locally_relevant_dofs, locally_owned_dofs);
    DoFTools::make_periodicity_constraints(dof_handler, 0, 1, 0, constraints);
    constraints.close();

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity, constraints);
    sparsity.compress();

    pcout << "  Initializing the matrices" << std::endl;
    mass_matrix.reinit(sparsity);
    stiffness_matrix.reinit(sparsity);
    lhs_matrix.reinit(sparsity);
    rhs_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

    solution_old_time.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    newton_update.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  }
}

void Parabolic::assemble_newton_system()
{
  TimerOutput::Scope t(computing_timer, "2. Assembly");
  lhs_matrix = 0.0;
  system_rhs = 0.0;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe, *quadrature,
                          update_values | update_gradients | update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  std::vector<double>         u_k_values(n_q);
  std::vector<Tensor<1, dim>> u_k_grads(n_q);
  std::vector<double>         u_old_values(n_q);

  const double eps = 0.01;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned()) continue;

      fe_values.reinit(cell);
      cell_matrix = 0.0;
      cell_rhs = 0.0;

      fe_values.get_function_values(solution_owned, u_k_values);
      fe_values.get_function_gradients(solution_owned, u_k_grads);
      fe_values.get_function_values(solution_old_time, u_old_values);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          double u_k   = u_k_values[q];
          double u_old = u_old_values[q];

          double reazione = (2.0 / eps) * u_k * (1.0 - u_k) * (1.0 - 2.0 * u_k);
          double derivata_reazione = (2.0 / eps) * (1.0 - 6.0 * u_k + 6.0 * u_k * u_k);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) += (
                      fe_values.shape_value(i, q) * fe_values.shape_value(j, q)
                      + deltat * eps * fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q)
                      + deltat * derivata_reazione * fe_values.shape_value(i, q) * fe_values.shape_value(j, q)
                  ) * fe_values.JxW(q);
                }

              cell_rhs(i) -= (
                  (u_k - u_old) * fe_values.shape_value(i, q)
                  + deltat * eps * u_k_grads[q] * fe_values.shape_grad(i, q)
                  + deltat * reazione * fe_values.shape_value(i, q)
              ) * fe_values.JxW(q);
            }
        }
      cell->get_dof_indices(dof_indices);
      
      constraints.distribute_local_to_global(cell_matrix, cell_rhs, dof_indices, lhs_matrix, system_rhs);
    }
  lhs_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}

void Parabolic::solve_newton()
{
  const unsigned int max_newton_iter = 15;
  const double       tol             = 1e-8;

  for (unsigned int iter = 0; iter < max_newton_iter; ++iter)
    {
      assemble_newton_system();
      {
      TimerOutput::Scope t(computing_timer, "3. Solve linear system");
      double residual_norm = system_rhs.l2_norm();
      pcout << "    Newton iter " << iter << " - Residual: " << residual_norm << std::endl;
      if (residual_norm < tol) break;

      SolverControl solver_control(1000, 1e-11);
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);
      TrilinosWrappers::PreconditionILU preconditioner;
      preconditioner.initialize(lhs_matrix, TrilinosWrappers::PreconditionILU::AdditionalData(1.0));

      solver.solve(lhs_matrix, newton_update, system_rhs, preconditioner);
      
      constraints.distribute(newton_update);
      
      solution_owned += newton_update;
      }
    }
  solution = solution_owned; 
}

void
Parabolic::output(const unsigned int &time_step) const
{
  static double cumulative_eval_time = 0.0; 

  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "u");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  auto start_eval = std::chrono::high_resolution_clock::now();

  data_out.build_patches();

  auto end_eval = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff_eval = end_eval - start_eval;
  cumulative_eval_time += diff_eval.count();

  data_out.write_vtu_with_pvtu_record(
    "./", "output", time_step, MPI_COMM_WORLD, 3);

  if (time_step == 50) {
      pcout << "\n===============================================" << std::endl;
      pcout << ">>> Evaluation Time: " 
            << cumulative_eval_time << " s <<<" << std::endl;
      pcout << "===============================================\n" << std::endl;
  }
}

void
Parabolic::solve()
{
  pcout << "===============================================" << std::endl;

  time = 0.0;

  {
    pcout << "Applying the initial condition" << std::endl;

    VectorTools::interpolate(dof_handler, u_0, solution_owned);
    constraints.distribute(solution_owned);
    solution = solution_owned;

    output(0);
    pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;

  while (time < T - 0.5 * deltat)
    {
      time += deltat;
      ++time_step;

      pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << time << ":" << std::flush;

      solution_old_time = solution_owned;
      newton_update = 0.0;
      solve_newton();
      output(time_step);
    }
}

#ifdef CONVERGENCE
double
Parabolic::compute_error(const VectorTools::NormType &norm_type)
{
  TimerOutput::Scope t(computing_timer, "5. Compute Error");
  FE_SimplexP<dim> fe_linear(1);
  MappingFE        mapping(fe_linear);

  const QGaussSimplex<dim> quadrature_error = QGaussSimplex<dim>(r + 2);

  exact_solution.set_time(time);

  Vector<double> error_per_cell;
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  const double error =
    VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}

#endif //CONVERGENCE