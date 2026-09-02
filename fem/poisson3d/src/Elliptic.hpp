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

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>

//#define NEUMANN
//#define CONVERGENCE
//#define TRANSPORT_COEFFICIENT
//#define REACTION_COEFFICIENT

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class Elliptic
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  // Diffusion coefficient
  class DiffusionCoefficient : public Function<dim>
  {
  public:
    // Constructor.
    DiffusionCoefficient()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/, const unsigned int /*component*/ = 0) const override
    {
      return 1.0;
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
      values[0] = 1.0;
      values[1] = 1.0;
    }

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 1.0;
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

  // Forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    ForcingTerm() {}
    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
      return 3.0 * M_PI * M_PI * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]) * std::sin(M_PI * p[2]);
    }
  };

  // Dirichlet boundary conditions.
  class FunctionG : public Function<dim>
  {
  public:
    // Constructor.
    FunctionG()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return std::exp(-1.0);
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
    value(const Point<dim> & /*p*/, const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };
#endif //NEUMANN


#ifdef CONVERGENCE
  // Exact solution.
  class ExactSolution : public Function<dim>
  {
  public:
    // Constructor.
    ExactSolution()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return std::cos(2.0 * M_PI * p[0]) * std::cos(4.0 * M_PI * p[1]);
    }

    // Gradient evaluation.
    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;

      result[0] =
        -2.0 * M_PI * std::sin(2.0 * M_PI * p[0]) * std::cos(4.0 * M_PI * p[1]);
      result[1] =
        -4.0 * M_PI * std::cos(2.0 * M_PI * p[0]) * std::sin(4.0 * M_PI * p[1]);

      return result;
    }
  };
  #endif //CONVERGENCE

  // Constructor.
  Elliptic(const unsigned int &N_, const unsigned int &r_)
    : N(N_)
    , r(r_)
  {}

  // Initialization.
  void
  setup();

  // System assembly.
  void
  assemble();

  // System solution.
  void
  solve();

  // Output.
  void
  output() const;

#ifdef CONVERGENCE
  // Compute the error.
  double
  compute_error(const VectorTools::NormType &norm_type) const;
#endif

protected:
  // Path to the mesh file.
  const unsigned int N;
  //const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Diffusion coefficient.
  DiffusionCoefficient diffusion_coefficient;
  
#ifdef TRANSPORT_COEFFICIENT
  TransportCoefficient transport_coefficient;
#endif //TRANSPORT_COEFFICIENT


#ifdef REACTION_COEFFICIENT
  // Reaction coefficient.
  ReactionCoefficient reaction_coefficient;
#endif //REACTION_COEFFICIENT

  // Forcing term.
  ForcingTerm forcing_term;

  // g(x).
  FunctionG function_g;

  // Triangulation.
  Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // Sparsity pattern.
  SparsityPattern sparsity_pattern;

  // System matrix.
  SparseMatrix<double> system_matrix;

  // System right-hand side.
  Vector<double> system_rhs;

  // System solution.
  Vector<double> solution;

#ifdef NEUMANN
  // Quadrature formula used on boundary lines.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

  // h(x).
  FunctionH function_h;
#endif //NEUMANN

};

#endif