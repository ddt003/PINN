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
#include <deal.II/base/timer.h>
#include <chrono>

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;


class Parabolic
{
public:
  static constexpr unsigned int dim = 1;

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
  class FunctionH : public Function<dim>
  {
  public:
    FunctionH()
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/ = 0) const
    {
      return p[1];
    }
  };
#endif //NEUMANN


#ifdef CONVERGENCE
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

      result[0] = 2 * M_PI * std::sin(5 * M_PI * get_time()) *
                  std::cos(2 * M_PI * p[0]) * std::sin(3 * M_PI * p[1]) *
                  std::sin(4 * M_PI * p[2]);

      result[1] = 3 * M_PI * std::sin(5 * M_PI * get_time()) *
                  std::sin(2 * M_PI * p[0]) * std::cos(3 * M_PI * p[1]) *
                  std::sin(4 * M_PI * p[2]);

      result[2] = 4 * M_PI * std::sin(5 * M_PI * get_time()) *
                  std::sin(2 * M_PI * p[0]) * std::sin(3 * M_PI * p[1]) *
                  std::cos(4 * M_PI * p[2]);

      return result;
    }
  };
#endif //CONVERGENCE

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
    , computing_timer(pcout, dealii::TimerOutput::summary, dealii::TimerOutput::wall_times)
  {}

  void
  setup();

  void
  solve();

#ifdef CONVERGENCE
  double
  compute_error(const VectorTools::NormType &norm_type);
#endif //CONVERGENCE

protected:
  void
  assemble_newton_system();

  void
  solve_newton();

  void
  output(const unsigned int &time_step) const;


  const unsigned int mpi_size;

  const unsigned int mpi_rank;

  ConditionalOStream pcout;

  FunctionMu mu;

  ForcingTerm forcing_term;
  
  FunctionU0 u_0;

  FunctionG function_g;

  double time;

  const double T;

  const unsigned int N;

  const unsigned int r;

  const double deltat;

  const double theta;

  parallel::fullydistributed::Triangulation<dim> mesh;

  std::unique_ptr<FiniteElement<dim>> fe;

  std::unique_ptr<Quadrature<dim>> quadrature;

  DoFHandler<dim> dof_handler;

  IndexSet locally_owned_dofs;

  IndexSet locally_relevant_dofs;

  AffineConstraints<double> constraints;

  TrilinosWrappers::SparseMatrix mass_matrix;

  TrilinosWrappers::SparseMatrix stiffness_matrix;

  TrilinosWrappers::SparseMatrix lhs_matrix;

  TrilinosWrappers::SparseMatrix rhs_matrix;

  TrilinosWrappers::MPI::Vector system_rhs;

  TrilinosWrappers::MPI::Vector solution_owned;

  TrilinosWrappers::MPI::Vector solution;

  TrilinosWrappers::MPI::Vector solution_old_time; 

  TrilinosWrappers::MPI::Vector newton_update;

  mutable TimerOutput computing_timer;
};

#endif