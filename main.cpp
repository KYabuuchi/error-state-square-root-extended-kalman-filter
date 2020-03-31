#include "ekf.hpp"
#include "simulator.hpp"
#include <iostream>

const float DT = 0.001f;

int main()
{
  Simulator sim(DT);
  EKF ekf(DT);


  for (int i = 0, N = static_cast<int>(1.0f / DT); i < N; i++) {
    Eigen::Vector3f acc = sim.getAcc(i);
    Eigen::Vector3f omega = sim.getOmega(i);
    Eigen::Vector3f pos = sim.getGps(i);
    Eigen::Quaternionf qua = sim.getOrientation(i);
    Eigen::Vector3f ans = sim.getTruePos(i);

    ekf.predict(acc, omega);
    ekf.observe(pos, qua);
    Eigen::Vector3f ekf_pos = ekf.getPos();

    // time, GPS, EKF, answer
    std::cout << i << " " << pos.transpose() << " " << ekf_pos.transpose() << " " << ans.transpose() << std::endl;
  }
}