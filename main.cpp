#include "simulator.hpp"
#include <iostream>

const float DT = 0.001f;

int main()
{
  Simulator sim(DT);

  for (int i = 0, N = static_cast<int>(1.0f / DT); i < N; i++) {
    Eigen::Vector3f acc = sim.getAcc(i);
    Eigen::Vector3f omega = sim.getOmega(i);
    Eigen::Vector3f pos = sim.getGps(i);
    Eigen::Vector3f ans = sim.getTruePos(i);

    Eigen::Vector3f ekf;
    ekf.setZero();

    // time, GPS, EKF, answer
    std::cout << i << " " << pos.transpose() << " " << ekf.transpose() << " " << ans.transpose() << std::endl;
  }
}