#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "Elliptic.hpp"

#ifdef CONVERGENCE
int
main(int /*argc*/, char * /*argv*/[])
{
  ConvergenceTable table;
  const std::vector<unsigned int>     N = {100, 200, 400, 800};
  const unsigned int             degree = 1;

  std::ofstream convergence_file("convergence.csv");
  convergence_file << "h,eL2,eH1" << std::endl;
  for (unsigned int i = 0; i < N.size(); ++i)
    {
      Elliptic problem(N[i], degree);

      problem.setup();
      problem.assemble();
      problem.solve();
      problem.output();

      const double error_L2 = problem.compute_error(VectorTools::L2_norm);
      const double error_H1 = problem.compute_error(VectorTools::H1_norm);

      double h=1.0/N[i];

      table.add_value("h", h);
      table.add_value("L2", error_L2);
      table.add_value("H1", error_H1);

      convergence_file << h /*h*/ << "," << error_L2 << "," << error_H1
                       << std::endl;
    }

  table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
  table.set_scientific("L2", true);
  table.set_scientific("H1", true);
  table.write_text(std::cout);

  return 0;
}
#endif //CONVERGENCE

#ifndef CONVERGENCE
int
main(int /*argc*/, char * /*argv*/[])
{
  const unsigned int N = 2000;
  const unsigned int r = 1;

  Elliptic problem(N, r);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}
#endif //CONVERGENCE