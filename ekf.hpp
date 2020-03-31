#pragma once
#include <Eigen/Dense>
#include <iostream>

class EKF
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EKF(float DT) : gravity(0, 0, 9.8f), DT(DT)
  {
    // state variable
    pos.setZero();
    vel.setZero();
    qua.setIdentity();

    // variance covariance matrix
    Eigen::MatrixXf P = Eigen::MatrixXf::Identity(9, 9);
    Pu = P.llt().matrixU();

    // drive noise
    Eigen::MatrixXf L, Q;
    L.setZero(9, 6);
    L.block(3, 0, 3, 3).setIdentity();
    L.bottomRightCorner(3, 3).setIdentity();
    Q.setZero(6, 6);
    Q.topLeftCorner(3, 3) = Eigen::Matrix3f::Identity() * 0.1;
    Q.bottomRightCorner(3, 3) = Eigen::Matrix3f::Identity() * 0.1;
    Eigen::MatrixXf V = L * Q * L.transpose();
    Vu = V.ldlt().matrixU();

    // observe noise
    Eigen::MatrixXf W;
    W.setZero(7, 7);
    W.topLeftCorner(3, 3) = 0.1 * Eigen::Matrix3f::Identity();      // position noise
    W.bottomRightCorner(4, 4) = 0.1 * Eigen::Matrix4f::Identity();  // rotation noise
    Wu = W.ldlt().matrixU();
  }

  void predict(const Eigen::Vector3f& acc, const Eigen::Vector3f& omega)
  {
    Eigen::Matrix3f R = qua.toRotationMatrix();
    Eigen::Quaternionf dq = exp(omega * DT);

    // Predict state
    Eigen::Vector3f nominal_acc = R * acc - gravity;
    pos += vel * DT + 0.5 * nominal_acc * DT * DT;
    vel += nominal_acc * DT;
    qua = qua * dq;

    // Propagate uncertainty
    Eigen::MatrixXf F = calcF(qua, acc, DT);
    Eigen::MatrixXf QR(18, 9);
    QR.topRows(9) = Pu * F.transpose();
    QR.bottomRows(9) = Vu;
    Eigen::MatrixXf Q = QR.householderQr().householderQ();
    Pu = (Q.transpose() * QR).topRows(9);
  }

  void observe(const Eigen::Vector3f& obs_p, const Eigen::Quaternionf& obs_q)
  {
    // error vector          (7)
    Eigen::VectorXf error = toVec(obs_p, obs_q) - toVec(pos, qua);

    // observation jacobian  (7x9)
    Eigen::MatrixXf H = calcH(qua);

    // QR (7+9,7+9)
    Eigen::MatrixXf QR(16, 16);
    QR.setZero();
    QR.topLeftCorner(7, 7) = Wu;
    QR.bottomLeftCorner(9, 7) = Pu * H.transpose();  // 9x9 9x7
    QR.bottomRightCorner(9, 9) = Pu;

    Eigen::MatrixXf Q = QR.householderQr().householderQ();
    Eigen::MatrixXf R = Q.transpose() * QR;
    Eigen::MatrixXf Su = R.topLeftCorner(7, 7);
    Eigen::MatrixXf Kt = R.topRightCorner(7, 9);
    Pu = R.bottomRightCorner(9, 9);

    Eigen::MatrixXf Sui = Su.triangularView<Eigen::Upper>().solve(Eigen::MatrixXf::Identity(7, 7));
    Eigen::VectorXf dx = (Kt.transpose() * Sui) * error;  // 9x7 7x7 7x1
    Eigen::Quaternionf dq = exp(dx.bottomRows(3));

    pos = pos + dx.topRows(3);
    vel = vel + dx.block(3, 0, 3, 1);
    qua = qua * dq;
  }

  Eigen::Vector3f getPos() const { return pos; }

private:
  Eigen::Quaternionf exp(const Eigen::Vector3f& v)
  {
    float norm = v.norm();
    float c = std::cos(norm / 2);
    float s = std::sin(norm / 2);
    Eigen::Vector3f n = s * v.normalized();
    return Eigen::Quaternionf(c, n.x(), n.y(), n.z());
  }

  Eigen::VectorXf toVec(const Eigen::Vector3f& p, const Eigen::Quaternionf& q)
  {
    Eigen::VectorXf x(7);
    x.topRows(3) = p;
    x(3) = q.w();
    x(4) = q.x();
    x(5) = q.y();
    x(6) = q.z();
    return x;
  }

  Eigen::MatrixXf calcH(const Eigen::Quaternionf& q)
  {
    Eigen::MatrixXf Q(4, 3);
    // clang-format off
    Q << -q.x() , -q.y() , -q.z() ,
          q.w() , -q.z() ,  q.y() ,
          q.z() ,  q.w() , -q.x() ,
         -q.y() ,  q.x() ,  q.w() ;
    // clang-format on
    Q *= 0.5;

    Eigen::MatrixXf H = Eigen::MatrixXf::Zero(7, 9);
    H.topLeftCorner(3, 3).setIdentity();
    H.bottomRightCorner(4, 3) = Q;
    return H;
  }


  // 伝搬行列 (9x9)
  Eigen::MatrixXf calcF(const Eigen::Quaternionf& q, const Eigen::Vector3f& acc, float dt)
  {
    Eigen::MatrixXf F = Eigen::MatrixXf::Identity(9, 9);
    F.block(0, 3, 3, 3) = Eigen::Matrix3f::Identity(3, 3) * dt;
    F.block(3, 6, 3, 3) = -hat(q.toRotationMatrix() * acc) * dt;
    return F;
  }

  Eigen::Matrix3f hat(const Eigen::Vector3f& vec)
  {
    Eigen::Matrix3f A;
    // clang-format off
    A <<
          0, -vec(2),  vec(1),
     vec(2),       0, -vec(0),
    -vec(1),  vec(0),       0;
    // clang-format on
    return A;
  }

  const Eigen::Vector3f gravity;
  const float DT;

  // drive noise
  Eigen::MatrixXf Vu;
  // observe noise
  Eigen::MatrixXf Wu;

  // nominal state
  Eigen::Vector3f pos, vel;
  Eigen::Quaternionf qua;

  // upper trianguler matrix of variance covariance matrix
  Eigen::MatrixXf Pu;  // 9x9
};
