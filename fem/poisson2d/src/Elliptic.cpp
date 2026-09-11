#include "Elliptic.hpp"

void
Elliptic::setup()
{
  std::cout << "===============================================" << std::endl;

  // Create the mesh.
  {
    //std::cout << "Initializing the mesh from " << mesh_file_name << std::endl;

    // Read the mesh from file:
/*     GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh);
    std::ifstream mesh_file(mesh_file_name);
    grid_in.read_msh(mesh_file); */

    GridGenerator::subdivided_hyper_cube(mesh, N, 0.0, 1.0, true);
    const std::string mesh_file_name = "mesh-" + std::to_string(N) + ".vtk";
    GridOut           grid_out;
    std::ofstream     grid_out_file(mesh_file_name);
    grid_out.write_vtk(mesh, grid_out_file);

    std::cout << "  Number of elements = " << mesh.n_active_cells() << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    std::cout << "Initializing the finite element space" << std::endl;
    // We use the FE_SimplexP class here, that is suitable for triangular (or tetrahedral) meshes.
    //fe = std::make_unique<FE_SimplexP<dim>>(r);
    fe = std::make_unique<FE_Q<dim>>(r);


    std::cout << "  Degree                     = " << fe->degree << std::endl;
    std::cout << "  DoFs per cell              = " << fe->dofs_per_cell << std::endl;

    // Construct the quadrature formula of the appopriate degree of exactness.
    //quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);
    quadrature = std::make_unique<QGauss<dim>>(r + 1);

    std::cout << "  Quadrature points per cell = " << quadrature->size() << std::endl;

#ifdef NEUMANN
    quadrature_boundary = std::make_unique<QGauss<dim - 1>>(r + 1);
    std::cout << "  Quadrature points per boundary cell = " << quadrature_boundary->size() << std::endl;
#endif

  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    std::cout << "Initializing the DoF handler" << std::endl;

    // Initialize the DoF handler with the mesh we constructed.
    dof_handler.reinit(mesh);

    // "Distribute" the degrees of freedom.
    dof_handler.distribute_dofs(*fe);

    std::cout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    std::cout << "Initializing the linear system" << std::endl;

    // We first initialize a "sparsity pattern"
    std::cout << "  Initializing the sparsity pattern" << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    // Then, we use the sparsity pattern to initialize the system matrix
    std::cout << "  Initializing the system matrix" << std::endl;
    system_matrix.reinit(sparsity_pattern);

    // Finally, we initialize the right-hand side and solution vectors.
    std::cout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(dof_handler.n_dofs());
    std::cout << "  Initializing the solution vector" << std::endl;
    solution.reinit(dof_handler.n_dofs());
  }
}

void
Elliptic::assemble()
{
  std::cout << "===============================================" << std::endl;

  std::cout << "  Assembling the linear system" << std::endl;

  // Number of local DoFs for each element.
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size();

  // FEValues instance. This object allows to compute basis functions, their
  // derivatives, the reference-to-current element mapping and its
  // derivatives on all quadrature points of all elements.
  FEValues<dim> fe_values(
    *fe,
    *quadrature,
    // Here we specify what quantities we need FEValues to compute on
    // quadrature points. For our test, we need:
    // - the values of shape functions (update_values);
    // - the derivative of shape functions (update_gradients);
    // - the position of quadrature points (update_quadrature_points);
    // - the product J_c(x_q)*w_q (update_JxW_values).
    update_values | update_gradients | update_quadrature_points |
      update_JxW_values);

#ifdef NEUMANN
  FEFaceValues<dim> fe_values_boundary(*fe,
                                       *quadrature_boundary,
                                       update_values |
                                         update_quadrature_points |
                                         update_JxW_values);
#endif //NEUMANN

  // Local matrix and right-hand side vector. We will overwrite them for each element within the loop.
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  // We will use this vector to store the global indices of the DoFs of the current element within the loop.
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Reset the global matrix and vector, just in case.
  system_matrix = 0.0;
  system_rhs    = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      // Reinitialize the FEValues object on current element.
      fe_values.reinit(cell);

      // We reset the cell matrix and vector (discarding any leftovers from
      // previous element).
      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double mu = diffusion_coefficient.value(fe_values.quadrature_point(q));
          //const double sigma = reaction_coefficient.value(fe_values.quadrature_point(q));
#ifdef TRANSPORT_COEFFICIENT
          Vector<double> transport_coefficient_loc(dim);
          transport_coefficient.vector_value(fe_values.quadrature_point(q),
                                    transport_coefficient_loc);

          Tensor<1, dim> transport_coefficient_tensor;
          for (unsigned int d = 0; d < dim; ++d)
            transport_coefficient_tensor[d] = transport_coefficient_loc[d];
#endif

          // Here we iterate over *local* DoF indices.
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) += mu                               // mu(x)
                                       * fe_values.shape_grad(i, q)     // (I)
                                       * fe_values.shape_grad(j, q)     // (II)
                                       * fe_values.JxW(q);              // (III)

#ifdef TRANSPORT_COEFFICIENT
                  cell_matrix(i, j) -= scalar_product(transport_coefficient_tensor,
                                                      fe_values.shape_grad(j,q))
                                    * fe_values.shape_value(i,q)
                                    * fe_values.JxW(q);
#endif //TRANSPORT_COEFFICIENT

#ifdef REACTION_COEFFICIENT
                  cell_matrix(i, j) +=
                    sigma *                            // sigma(x)
                    fe_values.shape_value(i, q) *      // phi_i
                    fe_values.shape_value(j, q) *      // phi_j
                    fe_values.JxW(q);                  // dx
#endif //REACTION_COEFFICIENT
                }

              cell_rhs(i) += forcing_term.value(fe_values.quadrature_point(q)) *
                             fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

#ifdef NEUMANN
      // If the cell is adjacent to the boundary...
      if (cell->at_boundary())
        {
          // ...we loop over its edges (referred to as faces in the deal.II jargon).
          for (unsigned int face_number = 0; face_number < cell->n_faces();
               ++face_number)
            {
              // If current face lies on the boundary, and its boundary ID (or
              // tag) is that of one of the Neumann boundaries, we assemble the
              // boundary integral.
              if (cell->face(face_number)->at_boundary() && (face_number==0 || face_number ==1 || face_number==3))
                {
                  fe_values_boundary.reinit(cell, face_number);

                  for (unsigned int q = 0; q < quadrature_boundary->size(); ++q)
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      cell_rhs(i) +=
                        function_h.value(
                          fe_values_boundary.quadrature_point(q)) * // h(xq)
                        fe_values_boundary.shape_value(i, q) *      // v(xq)
                        fe_values_boundary.JxW(q);                  // Jq wq
                }
            }
        }
#endif //NEUMANN

      // At this point the local matrix and vector are constructed: we
      // need to sum them into the global matrix and vector. To this end,
      // we need to retrieve the global indices of the DoFs of current
      // cell.
      cell->get_dof_indices(dof_indices);

      // Then, we add the local matrix and vector into the corresponding
      // positions of the global matrix and vector.
      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  {
    // We construct a map that stores, for each DoF corresponding to a Dirichlet
    // condition, the corresponding value. E.g., if the Dirichlet condition is
    // u_i = b_i, the map will contain the pair (i, b_i).
    std::map<types::global_dof_index, double> boundary_values;

    // This object represents our boundary data as a real-valued function (that
    // always evaluates to zero). Other functions may require to implement a
    // custom class derived from dealii::Function<dim>.
    Functions::ZeroFunction<dim> bc_function;

    // Then, we build a map that, for each boundary tag, stores a pointer to the
    // corresponding boundary function.
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    boundary_functions[2] = &bc_function;

    // interpolate_boundary_values fills the boundary_values map.
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    // Finally, we modify the linear system to apply the boundary conditions.
    // This replaces the equations for the boundary DoFs with the corresponding
    // u_i = 0 equations.
    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, true);
  }

}

void
Elliptic::solve()
{
  std::cout << "===============================================" << std::endl;

  // Here we specify the maximum number of iterations of the iterative solver,
  // and its tolerance.
  SolverControl solver_control(20000, 1e-10);

  // Since the system matrix is symmetric and positive definite, we solve the
  // system using the conjugate gradient method.
  //SolverCG<Vector<double>> solver(solver_control);

    
  SolverCG<Vector<double>> solver(solver_control);

  // Jacobi preconditioner
  SparseILU<double> preconditioner;
  preconditioner.initialize(system_matrix, SparseILU<double>::AdditionalData());

/*   // Sor preconditioner 
  PreconditionSOR preconditioner;
  preconditioner.initialize(
    system_matrix, PreconditionSOR<SparseMatrix<double>>::AdditionalData(1.0)); */


  std::cout << "  Solving the linear system" << std::endl;
  // We don't use any preconditioner for now, so we pass the identity matrix
  // as preconditioner.
  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  std::cout << "  " << solver_control.last_step() << "GMRES iterations"
            << std::endl;
}

void
Elliptic::output() const
{
  std::cout << "===============================================" << std::endl;

  // The DataOut class manages writing the results to a file.
  DataOut<dim> data_out;

  // It can write multiple variables (defined on the same mesh) to a single
  // file. Each of them can be added by calling add_data_vector, passing the
  // associated DoFHandler and a name.
  data_out.add_data_vector(dof_handler, solution, "solution");

  // Once all vectors have been inserted, call build_patches to finalize the
  // DataOut object, preparing it for writing to file.
  data_out.build_patches();

  // Then, use one of the many write_* methods to write the file in an
  // appropriate format.
  //const std::filesystem::path mesh_path(mesh_file_name);
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
  // The error is an integral, and we approximate that integral using a
  // quadrature formula. To make sure we are accurate enough, we use a
  // quadrature formula with one node more than what we used in assembly.
  const QGauss<dim> quadrature_error(r + 2);

  // for dim>=2
  FE_SimplexP<dim> fe_linear(1);
  MappingFE        mapping(fe_linear);

  // First we compute the norm on each element, and store it in a vector.
  Vector<double> error_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(MappingQGeneric<dim>(r),
                                    dof_handler,
                                    solution,
                                    ExactSolution(),
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  // Then, we add out all the cells.
  const double error =
    VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}
#endif