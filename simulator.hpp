#pragma once
#include <Eigen/Dense>
#include <random>
#include <vector>

class Simulator
{
public:
  Simulator(const float DT)
  {
    Eigen::Vector3f true_pos;
    Eigen::Vector3f true_vel;
    Eigen::Quaternionf true_q;
    const Eigen::Vector3f gravity(0, 0, 9.8);

    true_pos.setZero();
    true_vel.setZero();
    true_q.setIdentity();

    for (int i = 0, N = static_cast<int>(1.0f / DT); i < N; i++) {
      const Eigen::Matrix3f R = true_q.toRotationMatrix();

      Eigen::Vector3f global_acc, global_omega;
      global_acc << 13 * std::cos(static_cast<float>(i) * 0.0037), 17 * std::sin(static_cast<float>(i) * 0.0011), 7 * std::sin(static_cast<float>(i) * 0.0053);
      global_omega << 3 * std::sin(static_cast<float>(i) * 0.019), 7 * std::cos(static_cast<float>(i) * 0.073), 5 * std::sin(static_cast<float>(i) * 0.0053);

      Eigen::Vector3f sensor_acc, sensor_omega;
      sensor_acc = R.transpose() * (global_acc + gravity);
      sensor_omega = global_omega;

      true_pos += DT * true_vel + DT * DT * 0.5f * (R * sensor_acc - gravity);
      true_vel += DT * (R * sensor_acc - gravity);
      true_q = true_q * exp(sensor_omega * DT);

      sensor_acc_data.push_back(sensor_acc);
      sensor_omega_data.push_back(sensor_omega);
      true_pos_data.push_back(true_pos);
      true_q_data.push_back(true_q);
    }
  }
  // ローカル加速度
  Eigen::Vector3f sensorAcc(int i) { return sensor_acc_data[i] + 4.0f * Eigen::Vector3f::Random(); }

  // ローカル角速度
  Eigen::Vector3f sensorOmega(int i) { return sensor_omega_data[i] + 4.0f * Eigen::Vector3f::Random(); }

  // グローバル位置
  Eigen::Vector3f sensorPos(float i) { return true_pos_data[i] + 0.3f * Eigen::Vector3f::Random(); }

  // グローバル回転
  Eigen::Quaternionf sensorOrientation(float i) { return true_q_data[i] * randQ(); }

  // グローバル位置
  Eigen::Vector3f groundTruth(float i) { return true_pos_data[i]; }

private:
  Eigen::Quaternionf exp(const Eigen::Vector3f& v)
  {
    float norm = v.norm();
    float c = std::cos(norm / 2);
    float s = std::sin(norm / 2);
    Eigen::Vector3f n = s * v.normalized();
    return Eigen::Quaternionf(c, n.x(), n.y(), n.z());
  }

  Eigen::Quaternionf randQ()
  {
    std::normal_distribution<> norm(0.0, 1.57);
    return Eigen::Quaternionf(Eigen::AngleAxisf(norm(mt), Eigen::Vector3f::Random()));
  }

  std::mt19937 mt;

  std::vector<Eigen::Vector3f> sensor_acc_data;
  std::vector<Eigen::Vector3f> sensor_omega_data;
  std::vector<Eigen::Vector3f> true_pos_data;
  std::vector<Eigen::Quaternionf> true_q_data;
};