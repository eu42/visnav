/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <cmath>
#include <iostream>

#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {

const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

template <class T>
Eigen::Matrix<T, 3, 3> hat_operator(const Eigen::Matrix<T, 3, 1>& x) {
  Eigen::Matrix<T, 3, 3> x_hat;
  // clang-format off
  x_hat <<
      0, -x(2), x(1),
      x(2), 0, -x(0),
      -x(1), x(0), 0;
  // clang-format on
  return x_hat;
}

template <class T>
std::tuple<double, Eigen::Matrix<T, 3, 1>> get_angle_and_axis(
    const Eigen::Matrix<T, 3, 1>& x) {
  // get angle and axis from given vector
  double theta = x.norm();

  Eigen::Matrix<T, 3, 1> a;
  if (theta == 0) {
    a << 1, 1, 1;
  } else {
    a = x / theta;
  }

  return {theta, a};
}

// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(
    const Eigen::Matrix<T, 3, 1>& phi) {
  auto [theta, a] = get_angle_and_axis(phi);
  Eigen::Matrix<T, 3, 3> a_hat = hat_operator(a);

  double cos_theta = cos(theta);
  double sin_theta = sin(theta);

  return cos_theta * I + (1 - cos_theta) * a * a.transpose() +
         sin_theta * a_hat;
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {
  Eigen::Matrix<T, 3, 1> res = Eigen::MatrixXd::Zero(3, 1);

  double theta = acos((mat.trace() - 1) / 2);

  if (theta == 0) return res;

  Eigen::Matrix<T, 3, 1> diff_r_rT;
  diff_r_rT << mat(2, 1) - mat(1, 2), mat(0, 2) - mat(2, 0),
      mat(1, 0) - mat(0, 1);

  return theta * (1 / (2 * sin(theta))) * diff_r_rT;
}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(
    const Eigen::Matrix<T, 6, 1>& xi) {
  Eigen::Matrix<T, 4, 4> res = Eigen::Matrix4d::Identity();

  Eigen::Matrix<T, 3, 1> ro = xi.head(3);
  Eigen::Matrix<T, 3, 1> phi = xi.tail(3);
  Eigen::Matrix<T, 3, 3> exp_phi = user_implemented_expmap(phi);

  auto [theta, a] = get_angle_and_axis(phi);

  if (theta == 0) return res;

  Eigen::Matrix<T, 3, 3> a_hat = hat_operator(a);

  double cos_theta = cos(theta);
  double sin_theta_by_theta = sin(theta) / theta;

  Eigen::Matrix<T, 3, 3> jacobian;
  jacobian = sin_theta_by_theta * I +
             (1 - sin_theta_by_theta) * a * a.transpose() +
             (1 - cos_theta) / theta * a_hat;

  res.block(0, 0, 3, 3) = exp_phi;
  res.block(0, 3, 3, 1) = jacobian * ro;

  return res;
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {
  Eigen::Matrix<T, 6, 1> res = Eigen::MatrixXd::Zero(6, 1);

  Eigen::Matrix<T, 3, 3> r = mat.block(0, 0, 3, 3);
  Eigen::Matrix<T, 3, 1> t = mat.block(0, 3, 3, 1);

  Eigen::Matrix<T, 3, 1> phi = user_implemented_logmap(r);

  auto [theta, a] = get_angle_and_axis(phi);

  if (theta == 0) return res;

  Eigen::Matrix<T, 3, 3> a_hat = hat_operator(a);

  double cos_theta = cos(theta);
  double sin_theta_by_theta = sin(theta) / theta;

  Eigen::Matrix<T, 3, 3> jacobian;
  jacobian = sin_theta_by_theta * I +
             (1 - sin_theta_by_theta) * a * a.transpose() +
             (1 - cos_theta) / theta * a_hat;

  Eigen::Matrix<T, 3, 1> ro = jacobian.colPivHouseholderQr().solve(t);

  res.block(0, 0, 3, 1) = ro;
  res.block(3, 0, 3, 1) = phi;
  return res;
}

}  // namespace visnav
