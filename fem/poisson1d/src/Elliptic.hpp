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

#define CONVERGENCE

using namespace dealii;

class Elliptic
{
public:
  static constexpr unsigned int dim = 1;

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
      return -1.0*(4*p[0]*p[0]*p[0] - 6*p[0])*std::exp(-1.0*p[0]*p[0]);

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
      return p[0]*std::exp(-1.0*p[0]*p[0]);
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;

      result[0] =
        std::exp(-1.0*p[0]*p[0])*(1-2*p[0]*p[0]);

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

};

#endif