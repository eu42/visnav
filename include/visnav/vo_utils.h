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

#include <set>

#include <visnav/calibration.h>
#include <visnav/common_types.h>
#include <visnav/keypoints.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

namespace visnav {

void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double cam_z_threshold,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {
  projected_points.clear();
  projected_track_ids.clear();

  // project landmarks to the image plane using the current
  // locations of the cameras.
  for (const auto& [track_id, landmark] : landmarks) {
    // convert to camera coordinate frame
    Eigen::Vector3d p_3d = current_pose.inverse() * landmark.p;

    // Ignore all points that are behind the camera
    if (p_3d.z() < cam_z_threshold) continue;

    Eigen::Vector2d p_2d = cam->project(p_3d);

    // Ignore all points that project outside of the image plane.
    if (p_2d.x() < 0 || p_2d.x() >= w_image || p_2d.y() < 0 ||
        p_2d.y() >= h_image)
      continue;

    projected_points.push_back(p_2d);
    projected_track_ids.push_back(track_id);
  }
}

void find_matches_landmarks(
    const KeypointsData& kdl, const Landmarks& landmarks,
    const Corners& feature_corners,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    const std::vector<TrackId>& projected_track_ids,
    const double match_max_dist_2d, const int feature_match_max_dist,
    const double feature_match_test_next_best, MatchData& md) {
  md.matches.clear();

  // Find the matches between projected landmarks and detected
  // keypoints in the current frame.

  double match_max_dist_sq = match_max_dist_2d * match_max_dist_2d;

  // For every detected keypoint search for matches inside a circle with radius
  // match_max_dist_2d around the point location.
  for (size_t i = 0; i < kdl.corners.size(); ++i) {
    int best_match = -1;
    int min_dist = MAX_DISTANCE;
    int second_dist = MAX_DISTANCE;

    const Eigen::Vector2d& p_kp = kdl.corners[i];
    const Descriptor& desc_kp = kdl.corner_descriptors.at(i);

    for (size_t j = 0; j < projected_points.size(); ++j) {
      const Eigen::Vector2d& p_proj = projected_points[j];
      const TrackId track_id = projected_track_ids[j];

      // prevent sqrt operation by using squared_norm
      if ((p_kp - p_proj).squaredNorm() > match_max_dist_sq) continue;

      // For every landmark the distance is the minimal distance between
      // the descriptor of the current point and descriptors of all observations
      // of the landmarks.
      int min_dist_landmark = MAX_DISTANCE;
      for (const auto& [tcid, f_id] : landmarks.at(track_id).obs) {
        Descriptor desc_o =
            feature_corners.at(tcid).corner_descriptors.at(f_id);

        int dist = hamming_distance(desc_kp, desc_o);
        min_dist_landmark = std::min(dist, min_dist_landmark);
      }

      if (min_dist_landmark < min_dist) {
        second_dist = min_dist;
        min_dist = min_dist_landmark;
        best_match = track_id;
      } else if (min_dist_landmark < second_dist) {
        second_dist = min_dist_landmark;
      }
    }
    // The feature_match_max_dist and feature_match_test_next_best
    // should be used to filter outliers.
    if (best_match != -1 && min_dist < feature_match_max_dist &&
        min_dist * feature_match_test_next_best <= second_dist) {
      md.matches.emplace_back(i, best_match);
    }
  }
}

void localize_camera(const std::shared_ptr<AbstractCamera<double>>& cam,
                     const KeypointsData& kdl, const Landmarks& landmarks,
                     const double reprojection_error_pnp_inlier_threshold_pixel,
                     const MatchData& md, Sophus::SE3d& T_w_c,
                     std::vector<int>& inliers) {
  inliers.clear();

  if (md.matches.size() == 0) {
    T_w_c = Sophus::SE3d();
    return;
  }

  // Find the pose (T_w_c) and the inliers using the landmark
  // to keypoints matches and PnP. This should be similar to the
  // localize_camera in exercise 4 but in this execise we don't explicitelly
  // have tracks.

  opengv::bearingVectors_t bearing_vec;
  opengv::points_t points;
  for (const auto& [f_id, track_id] : md.matches) {
    Eigen::Vector3d p_3d_unprojected = cam->unproject(kdl.corners[f_id]);
    const Eigen::Vector3d& p_3d = landmarks.at(track_id).p;

    bearing_vec.push_back(p_3d_unprojected);
    points.push_back(p_3d);
  }

  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearing_vec, points);

  // create a Ransac object
  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      ransac;

  // create an AbsolutePoseSacProblem
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::KNEIP));

  // run ransac

  ransac.sac_model_ = absposeproblem_ptr;
  double focal_length = 500;
  double threshold =
      1.0 -
      cos(atan(reprojection_error_pnp_inlier_threshold_pixel / focal_length));
  ransac.threshold_ = threshold;
  ransac.computeModel();

  // optimize using all inliers
  ransac.sac_model_->optimizeModelCoefficients(
      ransac.inliers_, ransac.model_coefficients_, ransac.model_coefficients_);

  // apply threshold to ransac.sac_model_
  // directly populate `inliers`
  ransac.sac_model_->selectWithinDistance(ransac.model_coefficients_,
                                          ransac.threshold_, inliers);

  Eigen::Matrix3d R = ransac.model_coefficients_.block(0, 0, 3, 3);
  Eigen::Vector3d t = ransac.model_coefficients_.block(0, 3, 3, 1);

  T_w_c = Sophus::SE3d(R, t);
}

void add_new_landmarks(const TimeCamId tcidl, const TimeCamId tcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Sophus::SE3d& T_w_c0, const Calibration& calib_cam,
                       const std::vector<int> inliers,
                       const MatchData& md_stereo, const MatchData& md,
                       Landmarks& landmarks, TrackId& next_landmark_id) {
  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // Add new landmarks and observations. Here md_stereo
  // contains stereo matches for the current frame and md contains landmark
  // to map matches for the left camera (camera 0). The inliers vector
  // contains all inliers in md that were used to compute the pose T_w_c0.

  // track stereo observations that are added to landmarks to find observations
  // not added in order to triangulate and add them
  std::set<std::pair<FeatureId, FeatureId>> added_inliers;

  // For all inliers add the observations to the existing landmarks.
  for (int inlier_id : inliers) {
    const auto& [f_id, track_id] = md.matches[inlier_id];

    Landmark& landmark = landmarks[track_id];

    landmark.obs.emplace(tcidl, f_id);

    // if the left point is in md_stereo.inliers then add both observations
    for (const auto& [f_idl, f_idr] : md_stereo.inliers) {
      if (f_idl == f_id) {
        landmark.obs.emplace(tcidr, f_idr);

        // track stereo observations that are added to existing landmarks
        added_inliers.emplace(f_idl, f_idr);

        break;
      }
    }
  }

  // For all stereo observations that were not added to the existing landmarks
  // triangulate and add new landmarks.

  // create bearing vectors
  opengv::bearingVectors_t bearing_vec_0, bearing_vec_1;
  for (const auto& i : md_stereo.inliers) {
    // inlier is not added to landmarks
    if (added_inliers.find(i) == added_inliers.end()) {
      auto& [f_idl, f_idr] = i;
      bearing_vec_0.push_back(
          calib_cam.intrinsics[tcidl.second]->unproject(kdl.corners[f_idl]));
      bearing_vec_1.push_back(
          calib_cam.intrinsics[tcidr.second]->unproject(kdr.corners[f_idr]));
    }
  }

  // create the central relative adapter
  opengv::relative_pose::CentralRelativeAdapter adapter(
      bearing_vec_0, bearing_vec_1, t_0_1, R_0_1);

  int j = 0;
  for (auto i : md_stereo.inliers) {
    // inlier is not added to landmarks
    if (added_inliers.find(i) == added_inliers.end()) {
      Landmark l;

      // triangulated point at world coordinate frame
      l.p = T_w_c0 * opengv::triangulation::triangulate(adapter, j++);

      // add observations to the new landmark
      auto& [f_idl, f_idr] = i;
      l.obs.emplace(tcidl, f_idl);
      l.obs.emplace(tcidr, f_idr);

      // add new landmark to `landmarks`

      // Here next_landmark_id is a running index of the landmarks, so after
      // adding a new landmark you should always increase next_landmark_id
      // by 1.
      landmarks.emplace(next_landmark_id++, l);
    }
  }
}

void remove_old_keyframes(const TimeCamId tcidl, const int max_num_kfs,
                          Cameras& cameras, Cameras& old_cameras,
                          Landmarks& landmarks, Landmarks& old_landmarks,
                          std::set<FrameId>& kf_frames) {
  kf_frames.emplace(tcidl.first);

  // Remove old cameras and observations if the number of
  // keyframe pairs (left and right image is a pair) is larger than
  // max_num_kfs.

  // The ids of all the keyframes that are currently in the optimization should
  // be stored in kf_frames.
  while (kf_frames.size() > max_num_kfs) {
    // kf_frames has a chronological order, start to remove from the beginning
    const FrameId frame_id = *kf_frames.begin();

    kf_frames.erase(frame_id);

    // Removed keyframes should be removed from cameras
    TimeCamId tcid0(frame_id, 0);
    TimeCamId tcid1(frame_id, 1);
    // keep removed cameras in a separate variable for later use
    // only keep left camera
    old_cameras[tcid0] = cameras.at(tcid0);
    cameras.erase(tcid0);
    cameras.erase(tcid1);

    for (auto& [track_id, landmark] : landmarks) {
      auto& obs = landmark.obs;

      obs.erase(tcid0);
      obs.erase(tcid1);

      // landmarks with no left observations should be moved to old_landmarks.
      if (obs.size() == 0) {
        old_landmarks.emplace(track_id, landmark);
      }
    }
  }

  for (const auto& [track_id, landmark] : old_landmarks) {
    landmarks.erase(track_id);
  }
}
}  // namespace visnav
