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

#include <gtest/gtest.h>

#include <fstream>

#include "visnav/map_utils.h"
#include "visnav/vo_utils.h"

using namespace visnav;

const int MATCH_THRESHOLD = 70;
const double DIST_2_BEST = 1.2;

const std::string calib_path = "../../test/ex4_test_data/calib.json";

const std::string map_localize_path =
    "../../test/ex4_test_data/map_localize_camera.cereal";

const std::string matches_path = "../../test/ex5_test_data/matches.cereal";
const std::string projections_path =
    "../../test/ex5_test_data/projections.cereal";

void load_calib(const std::string& calib_path, Calibration& calib_cam) {
  std::ifstream os(calib_path, std::ios::binary);

  if (os.is_open()) {
    cereal::JSONInputArchive archive(os);
    archive(calib_cam);
  } else {
    ASSERT_TRUE(false) << "could not load camera ";
  }
}

void test_pose_equal(const Sophus::SE3d& pose_ref, const Sophus::SE3d& pose,
                     const double epsilon = 1e-6) {
  const Sophus::SE3d diff = pose_ref.inverse() * pose;
  EXPECT_LE(diff.log().norm(), epsilon);
}

TEST(Ex5TestSuite, ProjectLandmarks) {
  Calibration calib_cam;
  Corners feature_corners;
  Matches feature_matches;
  FeatureTracks feature_tracks;
  FeatureTracks outlier_tracks;
  Cameras cameras;
  Landmarks landmarks;

  load_calib(calib_path, calib_cam);

  load_map_file(map_localize_path, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras, landmarks);

  const TimeCamId tcid(3, 0);
  Sophus::SE3d current_pose = cameras[tcid].T_w_c;
  current_pose.translation() += Eigen::Vector3d(0.05, -0.03, 0.06);
  current_pose.so3() *= Sophus::SO3d::exp(Eigen::Vector3d(-0.02, 0.01, -0.03));

  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> pp,
      pp_loaded;
  std::vector<TrackId> ptid, ptid_loaded;

  {
    std::ifstream os(projections_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(pp_loaded);
    archive(ptid_loaded);
  }

  const double cam_z_threshold = 0.1;
  project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                    cam_z_threshold, pp, ptid);

  ASSERT_EQ(pp_loaded.size(), pp.size());
  ASSERT_EQ(ptid_loaded.size(), ptid.size());

  std::map<TrackId, Eigen::Vector2d> loaded_map;

  for (size_t i = 0; i < ptid_loaded.size(); i++) {
    loaded_map[ptid_loaded[i]] = pp_loaded[i];
  }

  for (size_t i = 0; i < pp_loaded.size(); i++) {
    Eigen::Vector2d ref = loaded_map[ptid[i]];
    EXPECT_TRUE(ref.isApprox(pp[i])) << "ref: " << ref << ", pp: " << pp[i];
  }
}

TEST(Ex5TestSuite, FindMatchesLandmarks) {
  Calibration calib_cam;
  Corners feature_corners;
  Matches feature_matches;
  FeatureTracks feature_tracks;
  FeatureTracks outlier_tracks;
  Cameras cameras;
  Landmarks landmarks;

  load_calib(calib_path, calib_cam);

  load_map_file(map_localize_path, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras, landmarks);

  MatchData md_loaded;

  {
    std::ifstream os(matches_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(md_loaded);
  }

  std::set<std::pair<FeatureId, FeatureId>> match_set_loaded(
      md_loaded.matches.begin(), md_loaded.matches.end());

  const TimeCamId tcid(3, 0);
  const Sophus::SE3d current_pose = Sophus::SE3d(
      cameras[tcid].T_w_c.so3() *
          Sophus::SO3d::exp(Eigen::Vector3d(-0.02, 0.01, -0.03)),
      cameras[tcid].T_w_c.translation() + Eigen::Vector3d(0.05, -0.03, 0.06));

  const double cam_z_threshold = 0.1;
  const double match_max_dist_2d = 20.0;

  // first check with the camera already being in the map (it should also work
  // in that case)
  {
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        projected_points;
    std::vector<TrackId> projected_track_ids;
    project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                      cam_z_threshold, projected_points, projected_track_ids);

    KeypointsData kdl = feature_corners[tcid];
    MatchData md;

    find_matches_landmarks(kdl, landmarks, feature_corners, projected_points,
                           projected_track_ids, match_max_dist_2d,
                           MATCH_THRESHOLD, DIST_2_BEST, md);

    std::set<std::pair<FeatureId, FeatureId>> match_set(md.matches.begin(),
                                                        md.matches.end());

    EXPECT_EQ(match_set_loaded.size(), match_set.size());

    for (const auto& pair : match_set_loaded) {
      EXPECT_TRUE(match_set.count(pair) > 0)
          << "Match (" << pair.first << "," << pair.second
          << ") from md_loaded.matches is missing in md.matches.";
    }

    for (const auto& pair : match_set) {
      EXPECT_TRUE(match_set_loaded.count(pair) > 0)
          << "Match (" << pair.first << "," << pair.second
          << ") in md.matches is too much (not present in md_loaded.matches).";
    }
  }

  // now remove the camera from the map and check again without the camera in
  // the map (the set of matches we expect is smaller in that case)
  {
    // the camera, observations, and expected matches that we know we don't get
    // in that case
    cameras.erase(tcid);
    for (auto& kv : landmarks) {
      kv.second.obs.erase(tcid);
    }
    for (auto p : {std::pair<FeatureId, FeatureId>({32, 1732}),
                   {36, 1732},
                   {65, 1787},
                   {89, 571},
                   {193, 227}}) {
      match_set_loaded.erase(p);
    }

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        projected_points;
    std::vector<TrackId> projected_track_ids;
    project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                      cam_z_threshold, projected_points, projected_track_ids);

    KeypointsData kdl = feature_corners[tcid];
    MatchData md;

    find_matches_landmarks(kdl, landmarks, feature_corners, projected_points,
                           projected_track_ids, match_max_dist_2d,
                           MATCH_THRESHOLD, DIST_2_BEST, md);

    std::set<std::pair<FeatureId, FeatureId>> match_set(md.matches.begin(),
                                                        md.matches.end());

    EXPECT_EQ(match_set_loaded.size(), match_set.size());

    for (const auto& pair : match_set_loaded) {
      EXPECT_TRUE(match_set.count(pair) > 0)
          << "Match (" << pair.first << "," << pair.second
          << ") from md_loaded.matches is missing in md.matches.";
    }

    for (const auto& pair : match_set) {
      EXPECT_TRUE(match_set_loaded.count(pair) > 0)
          << "Match (" << pair.first << "," << pair.second
          << ") in md.matches is too much (not present in md_loaded.matches).";
    }
  }
}

TEST(Ex5TestSuite, LocalizeCamera) {
  Calibration calib_cam;
  Corners feature_corners;
  Matches feature_matches;
  FeatureTracks feature_tracks;
  FeatureTracks outlier_tracks;
  Cameras cameras;
  Landmarks landmarks;

  load_calib(calib_path, calib_cam);

  load_map_file(map_localize_path, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras, landmarks);

  const TimeCamId tcid(3, 0);
  Sophus::SE3d current_pose = cameras[tcid].T_w_c;
  current_pose.translation() += Eigen::Vector3d(0.05, -0.03, 0.06);
  current_pose.so3() *= Sophus::SO3d::exp(Eigen::Vector3d(-0.02, 0.01, -0.03));

  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      projected_points;
  std::vector<TrackId> projected_track_ids;

  const double cam_z_threshold = 0.1;
  project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                    cam_z_threshold, projected_points, projected_track_ids);

  KeypointsData kdl = feature_corners[tcid];

  double match_max_dist_2d = 20.0;

  MatchData md;
  find_matches_landmarks(kdl, landmarks, feature_corners, projected_points,
                         projected_track_ids, match_max_dist_2d,
                         MATCH_THRESHOLD, DIST_2_BEST, md);

  // std::cerr << "matches " << md.matches.size() << std::endl;

  const double reprojection_error_pnp_inlier_threshold_pixel = 3.0;
  std::vector<int> inliers;
  Sophus::SE3d T_w_c;

  localize_camera(calib_cam.intrinsics[0], kdl, landmarks,
                  reprojection_error_pnp_inlier_threshold_pixel, md, T_w_c,
                  inliers);

  // std::cerr << "inliers " << inliers.size() << std::endl;

  test_pose_equal(cameras.at(tcid).T_w_c, T_w_c, 2e-2);
}

TEST(Ex5TestSuite, AddNewLandmarks) {
  Calibration calib_cam;
  Corners feature_corners;
  Matches feature_matches;
  FeatureTracks feature_tracks;
  FeatureTracks outlier_tracks;
  Cameras cameras;
  Landmarks landmarks;

  load_calib(calib_path, calib_cam);

  load_map_file(map_localize_path, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras, landmarks);

  const TimeCamId tcidl(3, 0);
  const TimeCamId tcidr(3, 1);

  // Remove all observations for those 2 frames
  std::set<TrackId> lm_empty;
  for (auto& kv_lm : landmarks) {
    kv_lm.second.obs.erase(tcidl);
    kv_lm.second.obs.erase(tcidr);

    if (kv_lm.second.obs.empty()) {
      lm_empty.emplace(kv_lm.first);
    }
  }

  for (const TrackId tid : lm_empty) {
    landmarks.erase(tid);
  }

  Sophus::SE3d current_pose = cameras[tcidl].T_w_c;
  current_pose.translation() += Eigen::Vector3d(0.05, -0.03, 0.06);
  current_pose.so3() *= Sophus::SO3d::exp(Eigen::Vector3d(-0.02, 0.01, -0.03));

  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      projected_points;
  std::vector<TrackId> projected_track_ids;

  const double cam_z_threshold = 0.1;
  project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                    cam_z_threshold, projected_points, projected_track_ids);

  const KeypointsData& kdl = feature_corners[tcidl];
  const KeypointsData& kdr = feature_corners[tcidr];

  double match_max_dist_2d = 20.0;

  MatchData md;
  find_matches_landmarks(kdl, landmarks, feature_corners, projected_points,
                         projected_track_ids, match_max_dist_2d,
                         MATCH_THRESHOLD, DIST_2_BEST, md);

  const MatchData& md_stereo = feature_matches[std::make_pair(tcidl, tcidr)];

  const double reprojection_error_pnp_inlier_threshold_pixel = 3.0;
  std::vector<int> inliers;
  Sophus::SE3d T_w_c;

  localize_camera(calib_cam.intrinsics[0], kdl, landmarks,
                  reprojection_error_pnp_inlier_threshold_pixel, md, T_w_c,
                  inliers);

  size_t landmarks_prev_size = landmarks.size();

  // set distinguishable landmark id for new landmarks
  const TrackId offset = 100000;
  TrackId next_landmark_id = 100000 + landmarks.size();
  add_new_landmarks(tcidl, tcidr, kdl, kdr, T_w_c, calib_cam, inliers,
                    md_stereo, md, landmarks, next_landmark_id);

  const size_t landmarks_new_size = landmarks.size();
  const size_t new_landmarks_added = landmarks_new_size - landmarks_prev_size;
  //  std::cerr << "landmarks_added " << landmarks_added << std::endl;
  //  std::cerr << "md_stereo.inliers.size() " << md_stereo.inliers.size()
  //            << std::endl;

  int new_landmark_ids = 0;
  int obs_left_cam_added = 0;
  for (const auto& kv_lm : landmarks) {
    if (kv_lm.first > offset) {
      // landmark with new id
      EXPECT_EQ(kv_lm.second.obs.size(), 2);
      EXPECT_TRUE(kv_lm.second.obs.find(tcidl) != kv_lm.second.obs.end());
      EXPECT_TRUE(kv_lm.second.obs.find(tcidr) != kv_lm.second.obs.end());
      new_landmark_ids++;
    } else {
      // previously existing landmark
      if (kv_lm.second.obs.find(tcidl) != kv_lm.second.obs.end()) {
        obs_left_cam_added++;
      }
    }
  }

  EXPECT_EQ(new_landmarks_added, new_landmark_ids)
      << "Wrong ids used for new landmarks?";

  // compute the number of features in the new stereo frame pair that we expect
  // to either be added as observations to existing landmarks, or be added as a
  // new landmark.
  std::set<int> expected_new_observations;
  for (const int i : inliers) {
    expected_new_observations.insert(md.matches[i].first);
  }

  std::set<int> expected_new_landmarks;
  for (const auto& kv : md_stereo.inliers) {
    if (!expected_new_observations.count(kv.first)) {
      expected_new_landmarks.insert(kv.first);
    }
  }

  // compare expected counts to actually added landmarks / observations
  EXPECT_EQ(expected_new_observations.size(), obs_left_cam_added);
  EXPECT_EQ(expected_new_landmarks.size(), new_landmarks_added);

  EXPECT_GT(obs_left_cam_added, 10);

  test_pose_equal(cameras.at(tcidl).T_w_c, T_w_c, 2e-2);
}

TEST(Ex5TestSuite, RemoveOldKFs) {
  Calibration calib_cam;
  Corners feature_corners;
  Matches feature_matches;
  FeatureTracks feature_tracks;
  FeatureTracks outlier_tracks;
  Cameras cameras, old_cameras;
  Landmarks landmarks;

  load_calib(calib_path, calib_cam);

  load_map_file(map_localize_path, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras, landmarks);

  Landmarks landmarks_loaded, old_landmarks;
  landmarks_loaded = landmarks;

  std::set<FrameId> kf_frames;
  for (const auto& kv : cameras) {
    kf_frames.emplace(kv.first.first);
  }

  const TimeCamId tcid(kf_frames.size(), 0);

  // std::cerr << "kf_frames.size() " << kf_frames.size() << std::endl;

  int max_num_kfs = 2;
  remove_old_keyframes(tcid, max_num_kfs, cameras, old_cameras, landmarks,
                       old_landmarks, kf_frames);

  // Check that we have right number of kfs.
  EXPECT_EQ(max_num_kfs, int(kf_frames.size()));

  // All cameras are in kf_frames
  for (const auto& kv : cameras) {
    EXPECT_TRUE(kf_frames.find(kv.first.first) != kf_frames.end());
  }

  EXPECT_EQ(old_landmarks.size() + landmarks.size(), landmarks_loaded.size());

  // Check that there are no observations from removed cameras.
  for (const auto& kv : landmarks) {
    for (const auto& obs_kv : kv.second.obs) {
      EXPECT_TRUE(kf_frames.find(obs_kv.first.first) != kf_frames.end());
    }
  }
}
