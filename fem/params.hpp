#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>


namespace homogeneous {
  
using namespace dealii;

template <int dim>
class RightHandSide : public Function<dim> {
public:
  RightHandSide() : Function<dim>() {}

  virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
    const double x = p[0];
    // const double y = p[1];
    // const double z = p[2];

    return (-4*x*x*x + 6*x) * std::exp(-x*x);
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p, const unsigned int = 0) const {
    const VectorizedArray<number> x = p[0];
    // const VectorizedArray<number> y = p[1];
    // const VectorizedArray<number> z = p[2];

    return (-4*x*x*x + 6*x) * std::exp(-x*x);
  }
};


template <int dim>
class DiffusionCoefficient : public Function<dim> {
public:
  DiffusionCoefficient() : Function<dim>() {}

  virtual double value(const Point<dim> &, const unsigned int = 0) const override {
    return 1.0;
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &, const unsigned int = 0) const {
    return VectorizedArray<number>(number(1.0));
  }
};


template <int dim>
class AdvectionCoefficient : public Function<dim> {
public:
  AdvectionCoefficient() : Function<dim>(dim) {}

  virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override {
    values[0] = 0.0;
  }

  template <typename number>
  Tensor<1, dim, VectorizedArray<number>> value(const Point<dim, VectorizedArray<number>> &p) const {
    Tensor<1, dim, VectorizedArray<number>> result;
    result[0] = 0.0;
    return result;
  }

  void tensor_value_list(const std::vector<Point<dim>> &points, std::vector<Tensor<1, dim>> &values) const {
    for (unsigned int p = 0; p < points.size(); ++p)
      values[p][0] = 0.0;
  }
};



template <int dim>
class ReactionCoefficient : public Function<dim> {
public:
  ReactionCoefficient() : Function<dim>() {}

  virtual double value(const Point<dim> &, const unsigned int = 0) const override {
    return 0.0;
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &, const unsigned int = 0) const {
    return VectorizedArray<number>(number(0.0));
  }
};


template <int dim>
class DirichletBoundaryInlet : public Function<dim> {
public:
  DirichletBoundaryInlet() : Function<dim>() {}

  virtual double value(const Point<dim> &, const unsigned int = 0) const override {
    return std::exp(-1.0);
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &, const unsigned int = 0) const {
    return VectorizedArray<number>(number(std::exp(-1.0)));
  }
};


template <int dim>
class DirichletBoundaryWalls : public Function<dim> {
public:
  DirichletBoundaryWalls() : Function<dim>() {}

  virtual double value(const Point<dim> &, const unsigned int = 0) const override {
    return 0.0;
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &, const unsigned int = 0) const {
    return VectorizedArray<number>(number(0.0));
  }
};


template <int dim>
class NeumannBoundaryValues : public Function<dim> {
public:
  NeumannBoundaryValues() : Function<dim>() {}

  virtual double value(const Point<dim> &, const unsigned int = 0) const override {
    return 0.0;
  }

  template <typename number>
  VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &, const unsigned int = 0) const {
    return VectorizedArray<number>(number(0.0));
  }
};


// ------
// exact solution
// ------
template <int dim>
class ExactSolution : public Function<dim> {
public:
  ExactSolution() : Function<dim>() {}

  virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
    return p[0] * std::exp(-p[0] * p[0]);
  }

  virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int = 0) const override {
    Tensor<1, dim> grad;
    grad[0] = (1 - 2 * p[0] * p[0]) * std::exp(-p[0] * p[0]);
    return grad;
  }
};

} // namespace homogeneous

#endif
