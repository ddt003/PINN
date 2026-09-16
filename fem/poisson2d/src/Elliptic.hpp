#ifndef ELLIPTIC
#define ELLIPTIC

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/timer.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>

#define NEUMANN
//#define CONVERGENCE

using namespace dealii;

class Elliptic
{
public:
  static constexpr unsigned int dim = 2;

  class DiffusionCoefficient : public Function<dim>
  {
  public:
    DiffusionCoefficient()
    {}

    virtual double
    value(const Point<dim> & /*p*/, const unsigned int /*component*/ = 0) const override
    {
      return 1.0;
    }
  };


  class ForcingTerm : public Function<dim>
  {
  public:
    ForcingTerm()
    {}

    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      const double x = p[0];
      const double y = p[1];
      
      double f = 2.0 * (
          std::pow(x, 4) * (3.0 * y - 2.0) +
          std::pow(x, 3) * (4.0 - 6.0 * y) +
          std::pow(x, 2) * (6.0 * std::pow(y, 3) - 12.0 * std::pow(y, 2) + 9.0 * y - 2.0) -
          6.0 * x * std::pow(y - 1.0, 2) * y +
          std::pow(y - 1.0, 2) * y
      );
      
      return -f; 
    }
  };

  class FunctionG : public Function<dim>
  {
  public:
    FunctionG()
    {}

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return std::exp(-1.0);
    }
  };

#ifdef NEUMANN
  class FunctionH : public Function<dim>
  {
  public:
    FunctionH()
    {}

    virtual double
    value(const Point<dim> & /*p*/, const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };
#endif //NEUMANN


#ifdef CONVERGENCE
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution()
    {}

    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      const double x = p[0];
      const double y = p[1];
      
      return (x * x) * ((x - 1.0) * (x - 1.0)) * y * ((y - 1.0) * (y - 1.0));    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;
      const double x = p[0];
      const double y = p[1];

      result[0] = (4.0 * x * x * x - 6.0 * x * x + 2.0 * x) * (y * (y - 1.0) * (y - 1.0));
      
      result[1] = (x * x * (x - 1.0) * (x - 1.0)) * (3.0 * y * y - 4.0 * y + 1.0);
      return result;
    }
  };
  #endif //CONVERGENCE

  Elliptic(const unsigned int &N_, const unsigned int &r_)
    : N(N_)
    , r(r_)
    , computing_timer(std::cout, dealii::TimerOutput::summary, dealii::TimerOutput::wall_times)  
  {}

  void
  setup();

  void
  assemble();

  void
  solve();

  void
  output() const;

#ifdef CONVERGENCE
  double
  compute_error(const VectorTools::NormType &norm_type) const;
#endif

protected:
  const unsigned int N;

  const unsigned int r;

  DiffusionCoefficient diffusion_coefficient;

  ForcingTerm forcing_term;

  FunctionG function_g;

  Triangulation<dim> mesh;

  std::unique_ptr<FiniteElement<dim>> fe;

  std::unique_ptr<Quadrature<dim>> quadrature;

  DoFHandler<dim> dof_handler;

  SparsityPattern sparsity_pattern;

  SparseMatrix<double> system_matrix;

  Vector<double> system_rhs;

  Vector<double> solution;

  mutable TimerOutput computing_timer;

#ifdef NEUMANN
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

  FunctionH function_h;
#endif //NEUMANN

};

#endif