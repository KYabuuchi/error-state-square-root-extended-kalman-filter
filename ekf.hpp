#pragma once
#include <Eigen/Dense>
#include <iostream>

class EKF
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3f getPos() const { return pos; }
  Eigen::Vector3f getVel() const { return vel; }
  Eigen::MatrixXf getCov() const { return Pu.transpose() * Pu; }

  EKF(float DT) : gravity(0, 0, 9.8f), DT(DT)
  {
    // states
    pos.setZero();
    vel.setZero();
    qua.setIdentity();

    // variance covariance matrix of error state
    Eigen::MatrixXf P = 0.2 * Eigen::MatrixXf::Identity(9, 9);
    Pu = P.llt().matrixU();

    // variance covariance matrix of IMU
    Eigen::MatrixXf Q;
    Q.setZero(6, 6);
    Q.topLeftCorner(3, 3) = Eigen::Matrix3f::Identity() * 0.1;
    Q.bottomRightCorner(3, 3) = Eigen::Matrix3f::Identity() * 0.1;
    Eigen::MatrixXf L;
    L.setZero(9, 6);
    L.block(3, 0, 3, 3).setIdentity();
    L.bottomRightCorner(3, 3).setIdentity();

    // driving noise
    Eigen::MatrixXf V = L * Q * L.transpose();
    Vu = V.ldlt().matrixU();

    // observation noise
    Eigen::MatrixXf W;
    W.setZero(7, 7);
    W.topLeftCorner(3, 3) = 0.1 * Eigen::Matrix3f::Identity();      // position noise
    W.bottomRightCorner(4, 4) = 0.1 * Eigen::Matrix4f::Identity();  // rotation noise
    Wu = W.ldlt().matrixU();
  }

  void predict(const Eigen::Vector3f& acc, const Eigen::Vector3f& omega)
  {
    Eigen::Matrix3f Rotate = qua.toRotationMatrix();
    Eigen::Quaternionf dq = exp(omega * DT);

    // update state vector
    Eigen::Vector3f nominal_acc = Rotate * acc - gravity;
    pos += vel * DT + 0.5 * nominal_acc * DT * DT;
    vel += nominal_acc * DT;
    qua = qua * dq;

    // propagation jacobian
    Eigen::MatrixXf F = calcF(qua, acc, DT);  // (9x9)

    // QR decompose
    Eigen::MatrixXf QR(18, 9);
    QR.topRows(9) = Pu * F.transpose();
    QR.bottomRows(9) = Vu * DT;
    Eigen::MatrixXf Q = QR.householderQr().householderQ();  // (18x18)
    Eigen::MatrixXf R = Q.transpose() * QR;                 // (18x9)

    // update variance covanriance matrix
    Pu = R.topRows(9);  // (9x9)
  }

  void observe(const Eigen::Vector3f& obs_p, const Eigen::Quaternionf& obs_q)
  {
    // inovation residual vector (7x1)
    Eigen::VectorXf error = toVec(obs_p, obs_q) - toVec(pos, qua);

    // observation jacobian (7x9)
    Eigen::MatrixXf H = calcH(qua);

    // QR decompose
    Eigen::MatrixXf QR(16, 16);
    QR.setZero();
    QR.topLeftCorner(7, 7) = Wu;
    QR.bottomLeftCorner(9, 7) = Pu * H.transpose();  // (9x7)=(9x9)*(9x7)
    QR.bottomRightCorner(9, 9) = Pu;
    Eigen::MatrixXf Q = QR.householderQr().householderQ();  // (16x16)
    Eigen::MatrixXf R = Q.transpose() * QR;                 // (16x16)

    Eigen::MatrixXf Kt = R.topRightCorner(7, 9);             // (7x9) upper trianguler matrix of the modified Kalman gain
    Eigen::MatrixXf Su = R.topLeftCorner(7, 7);              // (7x7) upper trianguler matrix of the inovation matrix
    Eigen::MatrixXf Sui = Su.triangularView<Eigen::Upper>()  // (7x7) inverse of the upper trianguler of the inovation matrix
                              .solve(Eigen::MatrixXf::Identity(7, 7));

    // update variance covariance matrix
    Pu = R.bottomRightCorner(9, 9);

    Eigen::VectorXf dx = (Kt.transpose() * Sui) * error;  // (9x1)=(9x7)*(7x7)*(7x1)

    // updata state vector
    Eigen::Quaternionf dq = exp(dx.bottomRows(3));
    pos = pos + dx.topRows(3);
    vel = vel + dx.block(3, 0, 3, 1);
    qua = qua * dq;
  }

private:
  // exponential image from so(3) to SO(3)
  Eigen::Quaternionf exp(const Eigen::Vector3f& v)
  {
    float norm = v.norm();
    float c = std::cos(norm / 2);
    float s = std::sin(norm / 2);
    Eigen::Vector3f n = s * v.normalized();
    return Eigen::Quaternionf(c, n.x(), n.y(), n.z());
  }

  // convert to a vector
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

  // calculate observation jacobian (7x9)
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

  // calculate propagation jacobian (9x9)
  Eigen::MatrixXf calcF(const Eigen::Quaternionf& q, const Eigen::Vector3f& acc, float dt)
  {
    Eigen::MatrixXf F = Eigen::MatrixXf::Identity(9, 9);
    F.block(0, 3, 3, 3) = Eigen::Matrix3f::Identity(3, 3) * dt;
    F.block(3, 6, 3, 3) = -hat(q.toRotationMatrix() * acc) * dt;
    return F;
  }

  // hat operator
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

  // upper trianguler matrix of variance covariance matrix (9x9)
  Eigen::MatrixXf Pu;
  // upper trianguler matrix of drive noise (9x9)
  Eigen::MatrixXf Vu;
  // upper trianguler matrix of observe noise (7x7)
  Eigen::MatrixXf Wu;

  // nominal state
  Eigen::Vector3f pos, vel;
  Eigen::Quaternionf qua;
};
