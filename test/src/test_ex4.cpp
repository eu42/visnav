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

#include "visnav/map_utils.h"

using namespace visnav;

const std::string calib_path = "../../test/ex4_test_data/calib.json";

const std::string map_triangulate_path =
    "../../test/ex4_test_data/map_landmark_triangulation.cereal";

const std::string map_init_path =
    "../../test/ex4_test_data/map_initialization_pre_opt.cereal";

const std::string map_localize_path =
    "../../test/ex4_test_data/map_localize_camera.cereal";

const std::string map_optimized_path =
    "../../test/ex4_test_data/map_localize_camera_optimized.cereal";

void load_calib(const std::string& calib_path, Calibration& calib_cam) {
  std::ifstream os(calib_path, std::ios::binary);

  if (os.is_open()) {
    cereal::JSONInputArchive archive(os);
    archive(calib_cam);
  } else {
    ASSERT_TRUE(false) << "could not load camera ";
  }
}

void test_cameras_equal(const Cameras& cameras_ref, const Cameras& cameras) {
  ASSERT_EQ(cameras_ref.size(), cameras.size());
  for (const auto& kv : cameras_ref) {
    ASSERT_TRUE(cameras.count(kv.first) > 0)
        << "Expected camera " << kv.first.first << "_" << kv.first.second
        << " in map.";
    const Sophus::SE3d diff =
        kv.second.T_w_c.inverse() * cameras.at(kv.first).T_w_c;
    EXPECT_LE(diff.log().norm(), 1e-6)
        << "Pose of camera " << kv.first.first << "_" << kv.first.second
        << " inaccurate.";
  }
}

void test_landmark_equal(const Landmark& lm_ref, const Landmark& lm,
                         const TrackId lm_id) {
  const double lm_dist = (lm_ref.p - lm.p).norm();
  EXPECT_LE(lm_dist, 0.05) << "Position of lm " << lm_id << " inaccurate";
  EXPECT_EQ(lm_ref.obs, lm.obs)
      << "List of observations for lm " << lm_id << " doesn't match.";
}

void test_landmarks_equal(const Landmarks& landmarks_ref,
                          const Landmarks& landmarks) {
  ASSERT_EQ(landmarks_ref.size(), landmarks.size());
  for (const auto& kv : landmarks_ref) {
    ASSERT_TRUE(landmarks.count(kv.first) > 0);
    test_landmark_equal(kv.second, landmarks.at(kv.first), kv.first);
  }
}

TEST(Ex4TestSuite, Triangulate) {
  Calibration calib_cam;
  Corners feature_corners;
  Matches feature_matches;
  FeatureTracks feature_tracks;
  FeatureTracks outlier_tracks;
  Cameras cameras;
  Landmarks landmarks, landmarks_ref;

  load_calib(calib_path, calib_cam);

  load_map_file(map_triangulate_path, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras, landmarks_ref);

  const TimeCamId tcid0(0, 1);
  const TimeCamId tcid1(1, 1);

  // copy landmarks and remove landmarks shared by the two cameras
  landmarks = landmarks_ref;
  for (const auto& kv : landmarks_ref) {
    if (kv.second.obs.count(tcid0) > 0 && kv.second.obs.count(tcid1) > 0) {
      landmarks.erase(kv.first);
    }
  }

  // TODO: make sure to check that this would fail if triangualte landmarks adds
  // feature observations of cameras not yet in the map

  add_new_landmarks_between_cams(tcid0, tcid1, calib_cam, feature_corners,
                                 feature_tracks, cameras, landmarks);

  test_landmarks_equal(landmarks_ref, landmarks);
}

TEST(Ex4TestSuite, InitializeScene) {
  Calibration calib_cam;
  Corners feature_corners;
  Matches feature_matches;
  FeatureTracks feature_tracks;
  FeatureTracks outlier_tracks;
  Cameras cameras, cameras_ref;
  Landmarks landmarks, landmarks_ref;

  load_calib(calib_path, calib_cam);

  load_map_file(map_init_path, feature_corners, feature_matches, feature_tracks,
                outlier_tracks, cameras_ref, landmarks_ref);

  const TimeCamId tcid0(0, 0);
  const TimeCamId tcid1(0, 1);

  initialize_scene_from_stereo_pair(tcid0, tcid1, calib_cam, feature_corners,
                                    feature_tracks, cameras, landmarks);

  test_cameras_equal(cameras_ref, cameras);
  test_landmarks_equal(landmarks_ref, landmarks);
}

TEST(Ex4TestSuite, LocalizeCamera) {
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

  std::vector<TrackId> shared_tracks;
  GetSharedTracks(tcid, feature_tracks, landmarks, shared_tracks);

  const double reprojection_error_pnp_inlier_threshold_pixel = 3.0;
  std::vector<TrackId> inlier_track_ids;
  Sophus::SE3d T_w_c;
  localize_camera(tcid, shared_tracks, calib_cam, feature_corners,
                  feature_tracks, landmarks,
                  reprojection_error_pnp_inlier_threshold_pixel, T_w_c,
                  inlier_track_ids);

  std::set<TrackId> inlier_set(inlier_track_ids.begin(),
                               inlier_track_ids.end());

  EXPECT_EQ(151, shared_tracks.size());
  EXPECT_GE(145, inlier_track_ids.size());
  EXPECT_LE(120, inlier_track_ids.size());
  // some true outlier matches:
  EXPECT_FALSE(inlier_set.count(2114) > 0);
  EXPECT_FALSE(inlier_set.count(1260) > 0);
  // some badly localized landmarks:
  EXPECT_FALSE(inlier_set.count(1866) > 0);
  EXPECT_FALSE(inlier_set.count(1158) > 0);
  EXPECT_FALSE(inlier_set.count(674) > 0);

  // TODO: also check that inlier track ids is a subset of shared_tracks (to
  // catch people that return e.g. the inices of the ransac point vector).

  // TODO: check Kahn error and make sure it is covered in
  // the test

  // TODO: make Kahn's and Nail's initial version fail. Also localize a camera
  // that is not yet in the map. See
  // https://gitlab9.in.tum.de/visnav_ws18/w0035/visnav_ws18/merge_requests/8/diffs#note_2111

  const Sophus::SE3d diff = cameras.at(tcid).T_w_c.inverse() * T_w_c;
  EXPECT_LE(diff.log().norm(), 0.02) << "Pose of camera " << tcid.first << "_"
                                     << tcid.second << " inaccurate.";
}

TEST(Ex4TestSuite, BundleAdjustment) {
  Calibration calib_cam;
  Corners feature_corners;
  Matches feature_matches;
  FeatureTracks feature_tracks;
  FeatureTracks outlier_tracks;
  Cameras cameras, cameras_ref;
  Landmarks landmarks, landmarks_ref;

  load_calib(calib_path, calib_cam);

  load_map_file(map_optimized_path, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras_ref, landmarks_ref);

  load_map_file(map_localize_path, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras, landmarks);

  BundleAdjustmentOptions ba_options;
  ba_options.optimize_intrinsics = false;
  ba_options.use_huber = true;
  ba_options.huber_parameter = 1.0;
  ba_options.max_num_iterations = 20;
  ba_options.verbosity_level = 0;
  std::set<TimeCamId> fixed_cameras = {{0, 0}, {0, 1}};

  bundle_adjustment(feature_corners, ba_options, fixed_cameras, calib_cam,
                    cameras, landmarks);

  test_cameras_equal(cameras_ref, cameras);
  test_landmarks_equal(landmarks_ref, landmarks);

  // TODO: check with and w/o huber, check with different hhuber parameters,
  // check with different fixed cameras (in particular two of the same
  // intrinsics)
}
