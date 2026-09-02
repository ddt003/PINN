#include <fstream>

#include "Parabolic.hpp"

#ifdef CONVERGENCE
// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int               mpi_rank =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  const unsigned int degree = 2;

  const double T     = 1.0;
  const double theta = 0.5;

  const std::vector<double> deltat_vector = {0.1, 0.05, 0.025, 0.0125};
  const std::vector<unsigned int> N = {10, 20, 40};
  const std::vector<std::string> meshes = {"../mesh/mesh-square-h0.100000.msh",
                                           "../mesh/mesh-square-h0.050000.msh",
                                           "../mesh/mesh-square-h0.025000.msh",
                                           "../mesh/mesh-square-h0.012500.msh"};
  std::vector<double> errors_L2;
  std::vector<double> errors_H1;
  //for (unsigned int i = 0; i < meshes.size(); ++i) {
  for (const auto &n : N) {
    for (const auto &deltat : deltat_vector)
      {
        //Parabolic problem(meshes[j], degree, T, deltat, theta);
        Parabolic problem(n, degree, T, deltat, theta);

        problem.setup();
        problem.solve();

        errors_L2.push_back(problem.compute_error(VectorTools::L2_norm));
        errors_H1.push_back(problem.compute_error(VectorTools::H1_norm));
      }
  }
  // Print the errors and estimate the convergence order.
  if (mpi_rank == 0)
    {
      std::cout << "==============================================="
                << std::endl;

      std::ofstream convergence_file("convergence.csv");
      convergence_file << "dt,eL2,eH1" << std::endl;
    for (unsigned int j=0; j<N.size() ; ++j){ //meshes.size()
      std::cout << "N = " << N[j] << std::endl; //meshes
      for (unsigned int i = 0; i < deltat_vector.size(); ++i)
        {
          
          convergence_file << deltat_vector[i] << "," << errors_L2[deltat_vector.size() * j + i] << ","
                           << errors_H1[deltat_vector.size()*j + i] << std::endl;

          std::cout << std::scientific << "dt = " << std::setw(4)
                    << std::setprecision(2) << deltat_vector[i];

          std::cout << std::scientific << " | eL2 = " << errors_L2[deltat_vector.size()* j +i];

          // Estimate the convergence order.
          if (i > 0)
            {
              const double p =
                std::log(errors_L2[deltat_vector.size()*j + i] / errors_L2[deltat_vector.size()*j + i - 1]) /
                std::log(deltat_vector[i] / deltat_vector[i - 1]);

              std::cout << " (" << std::fixed << std::setprecision(2)
                        << std::setw(4) << p << ")";
            }
          else
            std::cout << " (  - )";

          std::cout << std::scientific << " | eH1 = " << errors_H1[deltat_vector.size()* j + i];

          // Estimate the convergence order.
          if (i > 0)
            {
              const double p =
                std::log(errors_H1[deltat_vector.size()*j + i] / errors_H1[deltat_vector.size() * j + i - 1]) /
                std::log(deltat_vector[i] / deltat_vector[i - 1]);

              std::cout << " (" << std::fixed << std::setprecision(2)
                        << std::setw(4) << p << ")";
            }
          else
            std::cout << " (  - )";

          std::cout << "\n";
        }
      }  
    }

  return 0;
}
#endif //CONVERGENCE

#ifndef CONVERGENCE
// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  //const std::string  mesh_file_name = "../mesh/mesh-square-h0.100000.msh";
  const unsigned int N = 2048;
  const unsigned int degree         = 1;

  const double T      = 0.05;
  const double deltat = 0.001;
  const double theta  = 1.0;

  //Parabolic problem(mesh_file_name, degree, T, deltat, theta);
  Parabolic problem(N, degree, T, deltat, theta);
  
  problem.setup();
  problem.solve();

  return 0;
}
#endif //CONVERGENCE