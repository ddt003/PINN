#include "Elliptic.hpp"

void
Elliptic::setup()
{
  TimerOutput::Scope t(computing_timer, "1. Setup system");
  std::cout << "===============================================" << std::endl;

  {

    GridGenerator::subdivided_hyper_cube(mesh, N, 0.0, 1.0, true);
    const std::string mesh_file_name = "mesh-" + std::to_string(N) + ".vtk";
    GridOut           grid_out;
    std::ofstream     grid_out_file(mesh_file_name);
    grid_out.write_vtk(mesh, grid_out_file);

    std::cout << "  Number of elements = " << mesh.n_active_cells() << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  {
    std::cout << "Initializing the finite element space" << std::endl;
    fe = std::make_unique<FE_Q<dim>>(r);
    quadrature = std::make_unique<QGauss<dim>>(r + 1);

    std::cout << "  Quadrature points per cell = " << quadrature->size() << std::endl;

#ifdef NEUMANN
    quadrature_boundary = std::make_unique<QGauss<dim - 1>>(r + 1);
    std::cout << "  Quadrature points per boundary cell = " << quadrature_boundary->size() << std::endl;
#endif

  }

  std::cout << "-----------------------------------------------" << std::endl;

  {
    std::cout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    std::cout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

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
Elliptic::assemble()
{
  TimerOutput::Scope t(computing_timer, "2. Assembly");  
  std::cout << "===============================================" << std::endl;

  std::cout << "  Assembling the linear system" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(
    *fe,
    *quadrature,
    update_values | update_gradients | update_quadrature_points |
      update_JxW_values);


  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_matrix = 0.0;
  system_rhs    = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double mu = diffusion_coefficient.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) += mu                               // mu(x)
                                       * fe_values.shape_grad(i, q)     // (I)
                                       * fe_values.shape_grad(j, q)     // (II)
                                       * fe_values.JxW(q);              // (III)

                }

              cell_rhs(i) += forcing_term.value(fe_values.quadrature_point(q)) *
                             fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);

      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  {
    std::map<types::global_dof_index, double> boundary_values;

    Functions::ZeroFunction<dim> bc_function;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    for(unsigned int i=0; i<6; ++i) {
        boundary_functions[i] = &bc_function;
    }

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, true);
  }

}

void
Elliptic::solve()
{
  TimerOutput::Scope t(computing_timer, "3. Solve linear system");
  std::cout << "===============================================" << std::endl;

  SolverControl solver_control(20000, 1e-8);

    
  SolverCG<Vector<double>> solver(solver_control);

  SparseILU<double> preconditioner;
  preconditioner.initialize(system_matrix, SparseILU<double>::AdditionalData());

  std::cout << "  Solving the linear system" << std::endl;

  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  std::cout << "  " << solver_control.last_step() << "CG iterations"
            << std::endl;
}

void
Elliptic::output() const
{
  TimerOutput::Scope t(computing_timer, "4. Output");
  std::cout << "===============================================" << std::endl;

  DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler, solution, "solution");

  data_out.build_patches();

  const std::string           output_file_name =
    "output-" + std::to_string(N) + ".vtk";
  std::ofstream output_file(output_file_name);
  data_out.write_vtk(output_file);

  std::cout << "Output written to " << output_file_name << std::endl;

  std::cout << "===============================================" << std::endl;
}


#ifdef CONVERGENCE
double
Elliptic::compute_error(const VectorTools::NormType &norm_type) const
{
  TimerOutput::Scope t(computing_timer, "5. Compute Error");
  const QGauss<dim> quadrature_error(r + 2);

  Vector<double> error_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(MappingQGeneric<dim>(r),
                                    dof_handler,
                                    solution,
                                    ExactSolution(),
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  const double error =
    VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}
#endif