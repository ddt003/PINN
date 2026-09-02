#ifndef PARABOLIC_HPP
#define PARABOLIC_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>


#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;

//#define NEUMANN
//#define CONVERGENCE
//#define TRANSPORT_COEFFICIENT
//#define REACTION_COEFFICIENT

// Class representing the non-linear diffusion problem.
class Parabolic
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 1;

  // Function for the mu coefficient.
  class FunctionMu : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.01;
    }
  };

#ifdef TRANSPORT_COEFFICIENT
  // transportin coefficient.
  class TransportCoefficient : public Function<dim>
  {
  public:
    virtual void
    vector_value(const Point<dim> & /*p*/,
                 Vector<double> &values) const override
    {
      for (unsigned int i = 0; i < dim - 1; ++i)
        values[i] = 0.0;

      values[dim - 1] = -g;
    }

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      if (component == dim - 1)
        return -g;
      else
        return 0.0;
    }

  protected:
    const double g = 0.5;
  };
#endif //TRANSPORT_COEFFICIENT

#ifdef REACTION_COEFFICIENT
  // Reaction coefficient.
  class ReactionCoefficient : public Function<dim>
  {
  public:
    // Constructor.
    ReactionCoefficient()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 1.0;
    }
  };
#endif //REACTION_COEFFICIENT

  // Function for the forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  // Function for the initial condition.
  class FunctionU0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.25 * std::sin(2.0 * M_PI * p[0]) 
           + 0.25 * std::sin(16.0 * M_PI * p[0]) 
           + 0.5;    }
  };

  // Function for the Dirichlet boundary.
  class FunctionG : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &/*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

#ifdef NEUMANN
   // Neumann boundary conditions.
  class FunctionH : public Function<dim>
  {
  public:
    // Constructor.
    FunctionH()
    {}

    // Evaluation:
    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/ = 0) const
    {
      return p[1];
    }
  };
#endif //NEUMANN


#ifdef CONVERGENCE
  // Exact solution.
  class ExactSolution : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return std::sin(5 * M_PI * get_time()) * std::sin(2 * M_PI * p[0]) *
             std::sin(3 * M_PI * p[1]) * std::sin(4 * M_PI * p[2]);
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;

      // duex / dx
      result[0] = 2 * M_PI * std::sin(5 * M_PI * get_time()) *
                  std::cos(2 * M_PI * p[0]) * std::sin(3 * M_PI * p[1]) *
                  std::sin(4 * M_PI * p[2]);

      // duex / dy
      result[1] = 3 * M_PI * std::sin(5 * M_PI * get_time()) *
                  std::sin(2 * M_PI * p[0]) * std::cos(3 * M_PI * p[1]) *
                  std::sin(4 * M_PI * p[2]);

      // duex / dz
      result[2] = 4 * M_PI * std::sin(5 * M_PI * get_time()) *
                  std::sin(2 * M_PI * p[0]) * std::sin(3 * M_PI * p[1]) *
                  std::cos(4 * M_PI * p[2]);

      return result;
    }
  };
#endif //CONVERGENCE

  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  Parabolic(const unsigned int &N_,
       const unsigned int &r_,
       const double       &T_,
       const double       &deltat_,
       const double       &theta_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , T(T_)
    , N(N_)
    , r(r_)
    , deltat(deltat_)
    , theta(theta_)
    , mesh(MPI_COMM_WORLD)
  {}

  // Initialization.
  void
  setup();

  // Solve the problem.
  void
  solve();

#ifdef CONVERGENCE
  // Compute the error.
  double
  compute_error(const VectorTools::NormType &norm_type);
#endif //CONVERGENCE

protected:
  // Assemble the mass and stiffness matrices.
  void
  assemble_newton_system();

  // Assemble the right-hand side of the problem.
  void
  solve_newton();

  // Output.
  void
  output(const unsigned int &time_step) const;

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  // mu coefficient.
  FunctionMu mu;

#ifdef TRANSPORT_COEFFICIENT
  TransportCoefficient transport_coefficient;
#endif //TRANSPORT_COEFFICIENT


#ifdef REACTION_COEFFICIENT
  // Reaction coefficient.
  ReactionCoefficient sigma;
#endif //REACTION_COEFFICIENT

  // Forcing term.
  ForcingTerm forcing_term;
  
  // Initial condition.
  FunctionU0 u_0;

  // g(x).
  FunctionG function_g;

#ifdef CONVERGENCE
  // Exact solution.
  ExactSolution exact_solution;
#endif //CONVERGENCE

  // Current time.
  double time;

  // Final time.
  const double T;

  // Discretization. ///////////////////////////////////////////////////////////

  // Mesh file name.
  const unsigned int N;

  // Polynomial degree.
  const unsigned int r;

  // Time step.
  const double deltat;

  // Theta parameter of the theta method.
  const double theta;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> constraints;

  // Mass matrix M / deltat.
  TrilinosWrappers::SparseMatrix mass_matrix;

  // Stiffness matrix A.
  TrilinosWrappers::SparseMatrix stiffness_matrix;

  // Matrix on the left-hand side (M / deltat + theta A).
  TrilinosWrappers::SparseMatrix lhs_matrix;

  // Matrix on the right-hand side (M / deltat - (1 - theta) A).
  TrilinosWrappers::SparseMatrix rhs_matrix;

  // Right-hand side vector in the linear system.
  TrilinosWrappers::MPI::Vector system_rhs;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;

  TrilinosWrappers::MPI::Vector solution_old_time; 

  TrilinosWrappers::MPI::Vector newton_update;

#ifdef NEUMANN
  // Quadrature formula used on boundary lines.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

  // h(x).
  FunctionH function_h;
#endif //NEUMANN
};

#endif