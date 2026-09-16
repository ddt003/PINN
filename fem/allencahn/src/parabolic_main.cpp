#include <fstream>

#include "Parabolic.hpp"

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::vector<unsigned int> N_vector = {32, 128, 512, 2048};
  const unsigned int degree         = 1;

  const double T      = 0.05;
  const double deltat = 0.001;
  const double theta  = 1.0;

for (unsigned int j=0; j<N_vector.size(); ++j ) {
  std::cout << "N = " << N_vector[j] << std::endl;
  Parabolic problem(N_vector[j], degree, T, deltat, theta);

  problem.setup();
  problem.solve();
}
  return 0;
}
