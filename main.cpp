#include "ekf.hpp"
#include "simulator.hpp"
#include <iostream>

const float DT = 0.001f;

int main()
{
  Simulator sim(DT);
  EKF ekf(DT);
  EKF imu(DT);

  for (int i = 0, N = static_cast<int>(1.0f / DT); i < N; i++) {
    Eigen::Vector3f acc = sim.sensorAcc(i);
    Eigen::Vector3f omega = sim.sensorOmega(i);
    Eigen::Vector3f pos = sim.sensorPos(i);
    Eigen::Quaternionf qua = sim.sensorOrientation(i);
    Eigen::Vector3f gt = sim.groundTruth(i);

    // standard kalman filter
    ekf.predict(acc, omega);
    ekf.observe(pos, qua);
    Eigen::Vector3f ekf_pos = ekf.getPos();

    // only prediction
    imu.predict(acc, omega);
    Eigen::Vector3f imu_only_pos = imu.getPos();

    // time, GPS, EKF, Grount-Truth, only-predict
    std::cout << i << " " << pos.transpose() << " " << ekf_pos.transpose() << " " << gt.transpose() << " " << imu_only_pos.transpose() << std::endl;
  }
}