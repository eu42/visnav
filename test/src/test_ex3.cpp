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

#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>

#include "visnav/keypoints.h"
#include "visnav/matching_utils.h"

#include "visnav/serialization.h"

#include <fstream>

using namespace visnav;

const int NUM_FEATURES = 1500;
const int MATCH_THRESHOLD = 70;
const double DIST_2_BEST = 1.2;

const std::string img0_path = "../../test/ex3_test_data/0_0.jpg";
const std::string img1_path = "../../test/ex3_test_data/0_1.jpg";

const std::string kd0_path = "../../test/ex3_test_data/kd0.json";
const std::string kd1_path = "../../test/ex3_test_data/kd1.json";

const std::string matches_stereo_path =
    "../../test/ex3_test_data/matches_stereo.json";
const std::string matches_path = "../../test/ex3_test_data/matches.json";

const std::string calib_path = "../../test/ex3_test_data/calib.json";

TEST(Ex3TestSuite, KeypointAngles) {
  pangolin::ManagedImage<uint8_t> img0 = pangolin::LoadImage(img0_path);
  pangolin::ManagedImage<uint8_t> img1 = pangolin::LoadImage(img1_path);

  KeypointsData kd0, kd1, kd0_loaded, kd1_loaded;

  detectKeypointsAndDescriptors(img0, kd0, NUM_FEATURES, true);
  detectKeypointsAndDescriptors(img1, kd1, NUM_FEATURES, true);

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0_loaded);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd1_loaded);
  }

  ASSERT_TRUE(kd0_loaded.corner_angles.size() == kd0.corner_angles.size());
  ASSERT_TRUE(kd1_loaded.corner_angles.size() == kd1.corner_angles.size());

  for (size_t i = 0; i < kd0_loaded.corner_angles.size(); i++) {
    ASSERT_TRUE(std::abs(kd0_loaded.corner_angles[i] - kd0.corner_angles[i]) <
                1e-8);
  }

  for (size_t i = 0; i < kd1_loaded.corner_angles.size(); i++) {
    ASSERT_TRUE(std::abs(kd1_loaded.corner_angles[i] - kd1.corner_angles[i]) <
                1e-8);
  }
}

TEST(Ex3TestSuite, KeypointDescriptors) {
  pangolin::ManagedImage<uint8_t> img0 = pangolin::LoadImage(img0_path);
  pangolin::ManagedImage<uint8_t> img1 = pangolin::LoadImage(img1_path);

  KeypointsData kd0, kd1, kd0_loaded, kd1_loaded;

  detectKeypointsAndDescriptors(img0, kd0, NUM_FEATURES, true);
  detectKeypointsAndDescriptors(img1, kd1, NUM_FEATURES, true);

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0_loaded);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd1_loaded);
  }

  ASSERT_TRUE(kd0_loaded.corner_descriptors.size() ==
              kd0.corner_descriptors.size());
  ASSERT_TRUE(kd1_loaded.corner_descriptors.size() ==
              kd1.corner_descriptors.size());

  for (size_t i = 0; i < kd0_loaded.corner_descriptors.size(); i++) {
    ASSERT_TRUE((kd0_loaded.corner_descriptors[i] ^ kd0.corner_descriptors[i])
                    .count() == 0);
  }

  for (size_t i = 0; i < kd1_loaded.corner_descriptors.size(); i++) {
    ASSERT_TRUE((kd1_loaded.corner_descriptors[i] ^ kd1.corner_descriptors[i])
                    .count() == 0);
  }
}

TEST(Ex3TestSuite, DescriptorMatching) {
  MatchData md, md_loaded;
  KeypointsData kd0_loaded, kd1_loaded;

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0_loaded);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd1_loaded);
  }

  matchDescriptors(kd0_loaded.corner_descriptors, kd1_loaded.corner_descriptors,
                   md.matches, MATCH_THRESHOLD, DIST_2_BEST);

  {
    std::ifstream os(matches_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(md_loaded);
  }

  ASSERT_TRUE(md_loaded.matches.size() == md.matches.size())
      << "md_loaded.matches.size() " << md_loaded.matches.size()
      << " md.matches.size() " << md.matches.size();

  for (size_t i = 0; i < md_loaded.matches.size(); i++) {
    ASSERT_TRUE(md_loaded.matches[i] == md.matches[i]);
  }
}

TEST(Ex3TestSuite, KeypointsAll) {
  pangolin::ManagedImage<uint8_t> img0 = pangolin::LoadImage(img0_path);
  pangolin::ManagedImage<uint8_t> img1 = pangolin::LoadImage(img1_path);

  MatchData md, md_loaded;
  KeypointsData kd0, kd1, kd0_loaded, kd1_loaded;

  detectKeypointsAndDescriptors(img0, kd0, NUM_FEATURES, true);
  detectKeypointsAndDescriptors(img1, kd1, NUM_FEATURES, true);

  matchDescriptors(kd0.corner_descriptors, kd1.corner_descriptors, md.matches,
                   MATCH_THRESHOLD, DIST_2_BEST);

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0_loaded);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd1_loaded);
  }

  {
    std::ifstream os(matches_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(md_loaded);
  }

  ASSERT_TRUE(md_loaded.matches.size() == md.matches.size())
      << "md_loaded.matches.size() " << md_loaded.matches.size()
      << " md.matches.size() " << md.matches.size();

  for (size_t i = 0; i < md_loaded.matches.size(); i++) {
    ASSERT_TRUE(md_loaded.matches[i] == md.matches[i]);
  }
}

TEST(Ex3TestSuite, EpipolarInliers) {
  Calibration calib;

  MatchData md, md_loaded;
  KeypointsData kd0_loaded, kd1_loaded;

  {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib);
    } else {
      ASSERT_TRUE(false) << "could not load camera ";
    }
  }

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0_loaded);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd1_loaded);
  }

  {
    std::ifstream os(matches_stereo_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(md_loaded);
  }

  // Essential matrix
  Eigen::Matrix3d E;
  Sophus::SE3d T_0_1 = calib.T_i_c[0].inverse() * calib.T_i_c[1];

  computeEssential(T_0_1, E);

  md.matches = md_loaded.matches;
  findInliersEssential(kd0_loaded, kd1_loaded, calib.intrinsics[0],
                       calib.intrinsics[1], E, 1e-3, md);

  ASSERT_TRUE(md_loaded.inliers.size() == md.inliers.size())
      << "md_loaded.inliers.size() " << md_loaded.inliers.size()
      << " md.inliers.size() " << md.inliers.size();

  for (size_t i = 0; i < md_loaded.inliers.size(); i++) {
    ASSERT_TRUE(md_loaded.inliers[i] == md.inliers[i]);
  }
}

TEST(Ex3TestSuite, RansacInliers) {
  Calibration calib;

  MatchData md, md_loaded;
  KeypointsData kd0_loaded, kd1_loaded;

  {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib);
    } else {
      ASSERT_TRUE(false) << "could not load camera ";
    }
  }

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0_loaded);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd1_loaded);
  }

  {
    std::ifstream os(matches_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(md_loaded);
  }

  md.matches = md_loaded.matches;
  findInliersRansac(kd0_loaded, kd1_loaded, calib.intrinsics[0],
                    calib.intrinsics[1], 1e-5, 20, md);

  // Translation is only determined up to scale, so normalize before comparison.
  const double dist = (md_loaded.T_i_j.translation().normalized() -
                       md.T_i_j.translation().normalized())
                          .norm();
  const double angle = md_loaded.T_i_j.unit_quaternion().angularDistance(
      md.T_i_j.unit_quaternion());

  const int inlier_count_diff =
      std::abs(int(md_loaded.inliers.size()) - int(md.inliers.size()));

  std::set<std::pair<int, int>> md_loaded_inliers(md_loaded.inliers.begin(),
                                                  md_loaded.inliers.end()),
      md_inliers(md.inliers.begin(), md.inliers.end()), md_itersection_inliers;

  // compute set intersection
  size_t max_size = std::max(md_loaded_inliers.size(), md_inliers.size());

  std::set_intersection(
      md_loaded_inliers.begin(), md_loaded_inliers.end(), md_inliers.begin(),
      md_inliers.end(),
      std::inserter(md_itersection_inliers, md_itersection_inliers.begin()));

  double intersection_fraction =
      double(md_itersection_inliers.size()) / max_size;
  ASSERT_TRUE(intersection_fraction > 0.99)
      << "intersection_fraction " << intersection_fraction;

  ASSERT_TRUE(inlier_count_diff < 20) << "inlier " << inlier_count_diff;
  ASSERT_TRUE(dist < 0.05) << "dist " << dist;
  ASSERT_TRUE(angle < 0.01) << "angle " << angle;
}
