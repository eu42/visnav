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

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/tbb.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <visnav/keypoints.h>
#include <visnav/map_utils.h>
#include <visnav/matching_utils.h>

#include <visnav/gui_helper.h>
#include <visnav/tracks.h>

#include <visnav/serialization.h>

using namespace visnav;

///////////////////////////////////////////////////////////////////////////////
/// Declarations
///////////////////////////////////////////////////////////////////////////////

void draw_image_overlay(pangolin::View& v, size_t cam_id);
void change_display_to_image(const TimeCamId& tcid);
void draw_scene();
void load_data(const std::string& path, const std::string& calib_path,
               int max_images = 0);
void save_map();
void load_map();
void clear_keypoints();
void clear_matches();
void clear_tracks();
void clear_map();
bool next_step();
void summary();
void detect_keypoints();
void match_stereo();
void match_all();
void unreachable();
void build_tracks();
void initialize_scene();
void compute_camera_candidate_set();
void add_next_camera();
void add_new_landmarks();
void optimize();
void compute_projections();
bool is_landmark_outlier(TrackId track_id);
void remove_outlier_landmarks();

///////////////////////////////////////////////////////////////////////////////
/// Constants
///////////////////////////////////////////////////////////////////////////////

constexpr int UI_WIDTH = 200;
constexpr int NUM_CAMS = 2;
const std::string corners_path = "corners.cereal";
const std::string matches_path = "matches.cereal";
const std::string map_path = "map.cereal";

///////////////////////////////////////////////////////////////////////////////
/// Variables
///////////////////////////////////////////////////////////////////////////////

/// intrinsic calibration
Calibration calib_cam;

/// loaded images
tbb::concurrent_unordered_map<TimeCamId, pangolin::ManagedImage<uint8_t>>
    images;

/// timestamps for all stereo pairs
std::vector<FrameId> timestamps;

/// detected feature locations and descriptors
Corners feature_corners;

/// pairwise feature matches
Matches feature_matches;

/// all inlier or unmapped feature tracks
FeatureTracks feature_tracks;

/// landmarks that have been removed from the map as outliers
FeatureTracks outlier_tracks;

/// camera poses in the current map
Cameras cameras;

/// landmark positions and feature observations in current map
Landmarks landmarks;

/// camera candidates keeps a list of potential cameras to add to the scene in
/// one go and also keeps track of the stage we are currently in (adding
/// cameras, adding landmarks, optimizing, etc)
CameraCandidates camera_candidates;

/// cashed info on reprojected landmarks; recomputed every time time from
/// cameras, landmarks, and feature_tracks; used for visualization and
/// determining outliers; indexed by images (shared with track_projections)
ImageProjections image_projections;

/// same as image_projections, but indexed by tracks (shared with
/// image_projections)
TrackProjections track_projections;

///////////////////////////////////////////////////////////////////////////////
/// GUI parameters
///////////////////////////////////////////////////////////////////////////////

// The following GUI elements can be enabled / disabled from the main panel by
// switching the prefix from "ui" to "hidden" or vice verca. This way you can
// show only the elements you need / want for development.

pangolin::Var<bool> ui_show_hidden("ui.show_extra_options", false, false, true);

//////////////////////////////////////////////
/// Image display options

pangolin::Var<int> show_frame1("ui.show_frame1", 0, 0, 1500);
pangolin::Var<int> show_cam1("ui.show_cam1", 0, 0, NUM_CAMS - 1);
pangolin::Var<int> show_frame2("ui.show_frame2", 0, 0, 1500);
pangolin::Var<int> show_cam2("ui.show_cam2", 1, 0, NUM_CAMS - 1);
pangolin::Var<bool> lock_frames("ui.lock_frames", true, false, true);
pangolin::Var<bool> show_detected("ui.show_detected", true, false, true);
pangolin::Var<bool> show_matches("ui.show_matches", true, false, true);
pangolin::Var<bool> show_inliers("ui.show_inliers", true, false, true);
pangolin::Var<bool> show_tracks("ui.show_tracks", true, false, true);
pangolin::Var<bool> show_reprojections("ui.show_reprojections", true, false,
                                       true);
pangolin::Var<bool> show_outlier_observations("ui.show_outlier_obs", false,
                                              false, true);
pangolin::Var<bool> show_ids("ui.show_ids", false, false, true);
pangolin::Var<bool> show_epipolar("hidden.show_epipolar", false, false, true);
pangolin::Var<bool> show_cameras3d("hidden.show_cameras", true, false, true);
pangolin::Var<bool> show_points3d("hidden.show_points", true, false, true);

//////////////////////////////////////////////
/// Feature extraction and matching options

pangolin::Var<int> num_features_per_image("hidden.num_features", 1500, 10,
                                          5000);
pangolin::Var<bool> rotate_features("hidden.rotate_features", true, false,
                                    true);
pangolin::Var<int> feature_match_max_dist("hidden.match_max_dist", 70, 1, 255);
pangolin::Var<double> feature_match_test_next_best("hidden.match_next_best",
                                                   1.2, 1, 4);
pangolin::Var<double> relative_pose_ransac_thresh("hidden.5pt_thresh", 5e-5,
                                                  1e-10, 1, true);
pangolin::Var<int> relative_pose_ransac_min_inliers("hidden.5pt_min_inlier", 16,
                                                    1, 100);

//////////////////////////////////////////////
/// Track building options

pangolin::Var<int> min_track_length("hidden.min_track_length", 3, 2, 20);

//////////////////////////////////////////////
/// Adding cameras and landmarks options

// first try to add a set of images with high inlier count to map at once
pangolin::Var<int> desired_localization_inlier_count("hidden.ideal_loc_inl", 40,
                                                     1, 100);
pangolin::Var<int> desired_inlier_max_cameras_to_add("hidden.ideal_max_cam", 15,
                                                     1, 50);

// if that failed, try with lower threshold, but add fewer at once
pangolin::Var<int> minimal_localization_inlier_count("hidden.min_loc_inl", 10,
                                                     1, 100);
pangolin::Var<int> minimal_inlier_max_cameras_to_add("hidden.min_max_cam", 2, 1,
                                                     50);

pangolin::Var<bool> always_add_all_observations("hidden.add_loc_outlier", false,
                                                false, true);

pangolin::Var<double> reprojection_error_pnp_inlier_threshold_pixel(
    "hidden.pnp_inlier_thresh", 3.0, 0.1, 10);

//////////////////////////////////////////////
/// Bundle Adjustment Options

pangolin::Var<bool> ba_optimize_intrinsics("hidden.ba_opt_intrinsics", false,
                                           false, true);
pangolin::Var<int> ba_verbose("hidden.ba_verbose", 1, 0, 2);

pangolin::Var<double> reprojection_error_huber_pixel("hidden.ba_huber_width",
                                                     1.0, 0.1, 10);

//////////////////////////////////////////////
/// Outlier removal Options

// Note: the distance thresholds can be in meters, since we initialize the
// scene with a metric distance between the initial stereo pair from camera
// calibration.

pangolin::Var<double> reprojection_error_outlier_threshold_normal_pixel(
    "hidden.outlier_repr", 3.0, 0.1, 10);
pangolin::Var<double> reprojection_error_outlier_threshold_huge_pixel(
    "hidden.outlier_repr_huge", 40.0, 0.1, 100);
pangolin::Var<double> camera_center_distance_outlier_threshold_meter(
    "hidden.outlier_dist", 0.1, 0.0, 1.0);
pangolin::Var<double> z_coordinate_outlier_threshold_meter("hidden.outlier_z",
                                                           0.05, -1.0, 1.0);

///////////////////////////////////////////////////////////////////////////////
/// GUI buttons
///////////////////////////////////////////////////////////////////////////////

// if you enable this, a hint for what should be the next step is printed
pangolin::Var<bool> show_next_step_hint("ui.next_step_hint", false, false,
                                        true);

// if you enable this, next_step is called repeatedly until completion
pangolin::Var<bool> continue_next("ui.continue_next", false, false, true);

using Button = pangolin::Var<std::function<void(void)>>;

Button next_step_btn("ui.next_step", &next_step);

Button detect_keypoints_btn("ui.detect_keypoints", &detect_keypoints);

Button match_stereo_btn("ui.match_stereo", &match_stereo);

Button match_all_btn("ui.match_all", &match_all);

Button unreachable_btn("hidden.unreachable", &unreachable);

Button build_tracks_btn("ui.build_tracks", &build_tracks);

Button init_scene_btn("ui.init_scene", &initialize_scene);

Button candidates_btn("ui.camera_candidates", &compute_camera_candidate_set);

Button add_camera_btn("ui.add_camera", &add_next_camera);

Button add_landmarks_btn("ui.add_landmarks", &add_new_landmarks);

Button optimize_btn("ui.optimize", &optimize);

Button remove_outlier_btn("ui.remove_outliers", &remove_outlier_landmarks);

Button save_map_btn("ui.save_map", &save_map);

Button load_map_btn("ui.load_map", &load_map);

///////////////////////////////////////////////////////////////////////////////
/// GUI and Boilerplate Implementation
///////////////////////////////////////////////////////////////////////////////

// Parse parameters, load data, and create GUI window and event loop (or
// process everything in non-gui mode).
int main(int argc, char** argv) {
  bool show_gui = true;
  std::string dataset_path;
  std::string cam_calib = "opt_calib.json";
  int max_frames = 0;

  CLI::App app{"App description"};

  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--dataset-path", dataset_path, "Dataset path")->required();
  app.add_option("--cam-calib", cam_calib,
                 "Path to camera calibration. Default: opt_calib.json.");
  app.add_option(
      "--max-frames", max_frames,
      "Maximum number of frames to load. 0 means load all. Default: 0.");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  load_data(dataset_path, cam_calib, max_frames);

  if (show_gui) {
    pangolin::CreateWindowAndBind("Main", 1800, 1000);

    glEnable(GL_DEPTH_TEST);

    // main parent display for images and 3d viewer
    pangolin::View& main_view =
        pangolin::Display("main")
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
            .SetLayout(pangolin::LayoutEqualVertical);

    // parent display for images
    pangolin::View& img_view_display =
        pangolin::Display("images").SetLayout(pangolin::LayoutEqual);
    main_view.AddDisplay(img_view_display);

    // main ui panel
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    // extra options panel
    pangolin::View& hidden_panel = pangolin::CreatePanel("hidden").SetBounds(
        0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH),
        pangolin::Attach::Pix(2 * UI_WIDTH));
    ui_show_hidden.Meta().gui_changed = true;

    // 2D image views
    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    while (img_view.size() < NUM_CAMS) {
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

      size_t idx = img_view.size();
      img_view.push_back(iv);

      img_view_display.AddDisplay(*iv);
      iv->extern_draw_function =
          std::bind(&draw_image_overlay, std::placeholders::_1, idx);
    }

    // 3D visualization (initial camera view optimized to see full map)
    pangolin::OpenGlRenderState camera(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
        pangolin::ModelViewLookAt(-3.4, -3.7, -8.3, 2.1, 0.6, 0.2,
                                  pangolin::AxisNegY));

    pangolin::View& display3D =
        pangolin::Display("scene")
            .SetAspect(-640 / 480.0)
            .SetHandler(new pangolin::Handler3D(camera));
    main_view.AddDisplay(display3D);

    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if (ui_show_hidden.GuiChanged()) {
        hidden_panel.Show(ui_show_hidden);
        const int panel_width = ui_show_hidden ? 2 * UI_WIDTH : UI_WIDTH;
        main_view.SetBounds(0.0, 1.0, pangolin::Attach::Pix(panel_width), 1.0);
      }

      display3D.Activate(camera);
      glClearColor(0.95f, 0.95f, 0.95f, 1.0f);  // light gray background

      draw_scene();

      img_view_display.Activate();

      if (lock_frames) {
        // in case of locking frames, chaning one should change the other
        if (show_frame1.GuiChanged()) {
          change_display_to_image(std::make_pair(show_frame1, 0));
          change_display_to_image(std::make_pair(show_frame1, 1));
        } else if (show_frame2.GuiChanged()) {
          change_display_to_image(std::make_pair(show_frame2, 0));
          change_display_to_image(std::make_pair(show_frame2, 1));
        }
      }

      if (show_frame1.GuiChanged() || show_cam1.GuiChanged()) {
        size_t frame_id = show_frame1;
        size_t cam_id = show_cam1;

        TimeCamId tcid;
        tcid.first = frame_id;
        tcid.second = cam_id;
        if (images.find(tcid) != images.end()) {
          pangolin::GlPixFormat fmt;
          fmt.glformat = GL_LUMINANCE;
          fmt.gltype = GL_UNSIGNED_BYTE;
          fmt.scalable_internal_format = GL_LUMINANCE8;

          img_view[0]->SetImage(images[tcid], fmt);
        } else {
          img_view[0]->Clear();
        }
      }

      if (show_frame2.GuiChanged() || show_cam2.GuiChanged()) {
        size_t frame_id = show_frame2;
        size_t cam_id = show_cam2;

        TimeCamId tcid;
        tcid.first = frame_id;
        tcid.second = cam_id;
        if (images.find(tcid) != images.end()) {
          pangolin::GlPixFormat fmt;
          fmt.glformat = GL_LUMINANCE;
          fmt.gltype = GL_UNSIGNED_BYTE;
          fmt.scalable_internal_format = GL_LUMINANCE8;

          img_view[1]->SetImage(images[tcid], fmt);
        } else {
          img_view[1]->Clear();
        }
      }

      pangolin::FinishFrame();

      if (continue_next) {
        // stop if there is nothing left to do
        continue_next = next_step();
      } else {
        // if the gui is just idling, make sure we don't burn too much CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }
    // print summary before closing
    summary();
  } else {
    // non-gui mode: do all steps, then save map
    while (next_step()) {
      // nop
    }
    save_map();
  }

  return 0;
}

// Visualize features and related info on top of the image views
void draw_image_overlay(pangolin::View& v, size_t view_id) {
  UNUSED(v);

  size_t frame_id = view_id == 0 ? show_frame1 : show_frame2;
  size_t cam_id = view_id == 0 ? show_cam1 : show_cam2;

  TimeCamId tcid = std::make_pair(frame_id, cam_id);

  float text_row = 20;

  if (show_detected) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 0.0);  // red
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (feature_corners.find(tcid) != feature_corners.end()) {
      const KeypointsData& cr = feature_corners.at(tcid);

      for (size_t i = 0; i < cr.corners.size(); i++) {
        Eigen::Vector2d c = cr.corners[i];
        double angle = cr.corner_angles[i];
        pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

        Eigen::Vector2d r(3, 0);
        Eigen::Rotation2Dd rot(angle);
        r = rot * r;

        pangolin::glDrawLine(c, c + r);
      }

      pangolin::GlFont::I()
          .Text("Detected %d corners", cr.corners.size())
          .Draw(5, text_row);

    } else {
      glLineWidth(1.0);

      pangolin::GlFont::I().Text("Corners not processed").Draw(5, text_row);
    }
    text_row += 20;
  }

  if (show_matches || show_inliers) {
    glLineWidth(1.0);
    glColor3f(0.0, 0.0, 1.0);  // blue
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    size_t o_frame_id = (view_id == 0 ? show_frame2 : show_frame1);
    size_t o_cam_id = (view_id == 0 ? show_cam2 : show_cam1);

    TimeCamId o_tcid = std::make_pair(o_frame_id, o_cam_id);

    int idx = -1;

    auto it = feature_matches.find(std::make_pair(tcid, o_tcid));

    if (it != feature_matches.end()) {
      idx = 0;
    } else {
      it = feature_matches.find(std::make_pair(o_tcid, tcid));
      if (it != feature_matches.end()) {
        idx = 1;
      }
    }

    if (idx >= 0 && show_matches) {
      if (feature_corners.find(tcid) != feature_corners.end()) {
        const KeypointsData& cr = feature_corners.at(tcid);

        for (size_t i = 0; i < it->second.matches.size(); i++) {
          size_t c_idx = idx == 0 ? it->second.matches[i].first
                                  : it->second.matches[i].second;

          Eigen::Vector2d c = cr.corners[c_idx];
          double angle = cr.corner_angles[c_idx];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          Eigen::Vector2d r(3, 0);
          Eigen::Rotation2Dd rot(angle);
          r = rot * r;

          pangolin::glDrawLine(c, c + r);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
          }
        }

        pangolin::GlFont::I()
            .Text("Detected %d matches", it->second.matches.size())
            .Draw(5, text_row);
        text_row += 20;
      }
    }

    glColor3f(0.0, 1.0, 0.0);  // green

    if (idx >= 0 && show_inliers) {
      if (feature_corners.find(tcid) != feature_corners.end()) {
        const KeypointsData& cr = feature_corners.at(tcid);

        for (size_t i = 0; i < it->second.inliers.size(); i++) {
          size_t c_idx = idx == 0 ? it->second.inliers[i].first
                                  : it->second.inliers[i].second;

          Eigen::Vector2d c = cr.corners[c_idx];
          double angle = cr.corner_angles[c_idx];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          Eigen::Vector2d r(3, 0);
          Eigen::Rotation2Dd rot(angle);
          r = rot * r;

          pangolin::glDrawLine(c, c + r);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
          }
        }

        pangolin::GlFont::I()
            .Text("Detected %d inliers", it->second.inliers.size())
            .Draw(5, text_row);
        text_row += 20;
      }
    }
  }

  if (show_tracks) {
    glLineWidth(1.0);
    glColor3f(0.0, 0.4, 0.4);  // dark teal
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (feature_corners.find(tcid) != feature_corners.end()) {
      const KeypointsData& cr = feature_corners.at(tcid);
      if (!(feature_tracks.empty() && outlier_tracks.empty())) {
        std::vector<TrackId> track_ids;
        std::vector<TrackId> outlier_track_ids;
        GetTracksInImage(tcid, feature_tracks, track_ids);
        GetTracksInImage(tcid, outlier_tracks, outlier_track_ids);

        size_t num_obs = 0;
        size_t num_outlier_obs = 0;

        for (TrackId track_id : track_ids) {
          if (landmarks.count(track_id) > 0) {
            // highlight tracks already in map
            if (landmarks.at(track_id).outlier_obs.count(tcid) > 0) {
              // track contained in map, but this observation is outlier
              glColor3f(0.9, 0.0, 0.9);  // purple
              ++num_outlier_obs;
            } else {
              glColor3f(0.0, 0.9, 0.9);  // bright teal
              ++num_obs;
            }
          } else {
            // not yet in map
            glColor3f(0.0, 0.4, 0.4);  // dark teal
          }
          const FeatureId feature_id = feature_tracks.at(track_id).at(tcid);
          const Eigen::Vector2d c = cr.corners[feature_id];
          const double angle = cr.corner_angles[feature_id];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          Eigen::Vector2d r(3, 0);
          const Eigen::Rotation2Dd rot(angle);
          r = rot * r;

          pangolin::glDrawLine(c, c + r);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", track_id).Draw(c[0], c[1]);
          }
        }

        // outliers gray
        glColor3f(0.5, 0.5, 0.5);  // grey
        for (TrackId track_id : outlier_track_ids) {
          const FeatureId feature_id = outlier_tracks.at(track_id).at(tcid);
          const Eigen::Vector2d c = cr.corners[feature_id];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", track_id).Draw(c[0], c[1]);
          }
        }

        glColor3f(0.0, 0.4, 0.4);  // dark teal
        pangolin::GlFont::I()
            .Text(
                "Contains %u tracks (%u obs, %u outlier obs, %u removed "
                "outliers)",
                track_ids.size() + outlier_track_ids.size(), num_obs,
                num_outlier_obs, outlier_track_ids.size())
            .Draw(5, text_row);
      } else {
        glLineWidth(1.0);

        pangolin::GlFont::I().Text("Tracks not built").Draw(5, text_row);
      }
    } else {
      glLineWidth(1.0);

      pangolin::GlFont::I().Text("Corners not processed").Draw(5, text_row);
    }
    text_row += 20;
  }

  if (show_reprojections) {
    if (image_projections.count(tcid) > 0) {
      glLineWidth(1.0);
      glColor3f(1.0, 0.0, 0.0);  // red
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      const size_t num_points = image_projections.at(tcid).obs.size();
      double error_sum = 0;
      size_t num_outliers = 0;

      // count up and draw all inlier projections
      for (const auto& lm_proj : image_projections.at(tcid).obs) {
        error_sum += lm_proj->reprojection_error;

        if (lm_proj->outlier_flags != OutlierNone) {
          // outlier point
          glColor3f(1.0, 0.0, 0.0);  // red
          ++num_outliers;
        } else if (lm_proj->reprojection_error >
                   reprojection_error_huber_pixel) {
          // close to outlier point
          glColor3f(1.0, 0.5, 0.0);  // orange
        } else {
          // clear inlier point
          glColor3f(1.0, 1.0, 0.0);  // yellow
        }
        pangolin::glDrawCirclePerimeter(lm_proj->point_reprojected, 3.0);
        pangolin::glDrawLine(lm_proj->point_measured,
                             lm_proj->point_reprojected);
      }

      // only draw outlier projections
      if (show_outlier_observations) {
        glColor3f(1.0, 0.0, 0.0);  // red
        for (const auto& lm_proj : image_projections.at(tcid).outlier_obs) {
          pangolin::glDrawCirclePerimeter(lm_proj->point_reprojected, 3.0);
          pangolin::glDrawLine(lm_proj->point_measured,
                               lm_proj->point_reprojected);
        }
      }

      glColor3f(1.0, 0.0, 0.0);  // red
      pangolin::GlFont::I()
          .Text("Average repr. error (%u points, %u new outliers): %.2f",
                num_points, num_outliers, error_sum / num_points)
          .Draw(5, text_row);
      text_row += 20;
    }
  }

  if (show_epipolar) {
    glLineWidth(1.0);
    glColor3f(0.0, 1.0, 1.0);  // bright teal
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    size_t o_frame_id = (view_id == 0 ? show_frame2 : show_frame1);
    size_t o_cam_id = (view_id == 0 ? show_cam2 : show_cam1);

    TimeCamId o_tcid = std::make_pair(o_frame_id, o_cam_id);

    int idx = -1;

    auto it = feature_matches.find(std::make_pair(tcid, o_tcid));

    if (it != feature_matches.end()) {
      idx = 0;
    } else {
      it = feature_matches.find(std::make_pair(o_tcid, tcid));
      if (it != feature_matches.end()) {
        idx = 1;
      }
    }

    if (idx >= 0 && it->second.inliers.size() > 20) {
      Sophus::SE3d T_this_other =
          idx == 0 ? it->second.T_i_j : it->second.T_i_j.inverse();

      Eigen::Vector3d p0 = T_this_other.translation().normalized();

      int line_id = 0;
      for (double i = -M_PI_2 / 2; i <= M_PI_2 / 2; i += 0.05) {
        Eigen::Vector3d p1(0, sin(i), cos(i));

        if (idx == 0) p1 = it->second.T_i_j * p1;

        p1.normalize();

        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
            line;
        for (double j = -1; j <= 1; j += 0.001) {
          line.emplace_back(calib_cam.intrinsics[cam_id]->project(
              p0 * j + (1 - std::abs(j)) * p1));
        }

        Eigen::Vector2d c = calib_cam.intrinsics[cam_id]->project(p1);
        pangolin::GlFont::I().Text("%d", line_id).Draw(c[0], c[1]);
        line_id++;

        pangolin::glDrawLineStrip(line);
      }
    }
  }
}

// Update the image views to a given image id
void change_display_to_image(const TimeCamId& tcid) {
  if (0 == tcid.second) {
    // left view
    show_cam1 = 0;
    show_frame1 = tcid.first;
    show_cam1.Meta().gui_changed = true;
    show_frame1.Meta().gui_changed = true;
  } else {
    // right view
    show_cam2 = tcid.second;
    show_frame2 = tcid.first;
    show_cam2.Meta().gui_changed = true;
    show_frame2.Meta().gui_changed = true;
  }
}

// Render the 3D viewer scene of cameras and points
void draw_scene() {
  const TimeCamId tcid1 = std::make_pair(show_frame1, show_cam1);
  const TimeCamId tcid2 = std::make_pair(show_frame2, show_cam2);

  const u_int8_t color_camera_left[3]{0, 125, 0};            // dark green
  const u_int8_t color_camera_right[3]{0, 0, 125};           // dark blue
  const u_int8_t color_points[3]{0, 0, 0};                   // black
  const u_int8_t color_selected_left[3]{0, 250, 0};          // green
  const u_int8_t color_selected_right[3]{0, 0, 250};         // blue
  const u_int8_t color_selected_both[3]{0, 250, 250};        // teal
  const u_int8_t color_outlier_points[3]{250, 0, 0};         // red
  const u_int8_t color_outlier_observation[3]{250, 0, 250};  // purple

  // render cameras
  if (show_cameras3d) {
    for (const auto& cam : cameras) {
      if (cam.first == tcid1) {
        render_camera(cam.second.T_w_c.matrix(), 3.0f, color_selected_left,
                      0.1f);
      } else if (cam.first == tcid2) {
        render_camera(cam.second.T_w_c.matrix(), 3.0f, color_selected_right,
                      0.1f);
      } else if (cam.first.second == 0) {
        render_camera(cam.second.T_w_c.matrix(), 2.0f, color_camera_left, 0.1f);
      } else {
        render_camera(cam.second.T_w_c.matrix(), 2.0f, color_camera_right,
                      0.1f);
      }
    }
  }

  // render points
  if (show_points3d && landmarks.size() > 0) {
    glPointSize(3.0);
    glBegin(GL_POINTS);
    for (const auto& kv_lm : landmarks) {
      const TrackId track_id = kv_lm.first;

      const bool in_cam_1 = kv_lm.second.obs.count(tcid1) > 0;
      const bool in_cam_2 = kv_lm.second.obs.count(tcid2) > 0;

      const bool outlier_in_cam_1 = kv_lm.second.outlier_obs.count(tcid1) > 0;
      const bool outlier_in_cam_2 = kv_lm.second.outlier_obs.count(tcid2) > 0;

      if (is_landmark_outlier(track_id)) {
        glColor3ubv(color_outlier_points);
      } else if (in_cam_1 && in_cam_2) {
        glColor3ubv(color_selected_both);
      } else if (in_cam_1) {
        glColor3ubv(color_selected_left);
      } else if (in_cam_2) {
        glColor3ubv(color_selected_right);
      } else if (outlier_in_cam_1 || outlier_in_cam_2) {
        glColor3ubv(color_outlier_observation);
      } else {
        glColor3ubv(color_points);
      }

      pangolin::glVertex(kv_lm.second.p);
    }
    glEnd();
  }
}

// Load images, calibration, and features / matches if available.
// If max_frames > 0, load at most that many frames (each frame consists of
// NUM_CAMS images)
void load_data(const std::string& dataset_path, const std::string& calib_path,
               const int max_images) {
  const std::string timestams_path = dataset_path + "/timestamps.txt";

  {
    std::ifstream times(timestams_path);

    int64_t timestamp;

    int id = 0;

    while (times && (max_images <= 0 || id < max_images)) {
      times >> timestamp;

      // ensure that we actually read a new timestamp (and not e.g. just newline
      // at the end of the file)
      if (times.fail()) {
        times.clear();
        std::string temp;
        times >> temp;
        if (temp.size() > 0) {
          std::cerr << "Skipping '" << temp << "' while reading times."
                    << std::endl;
        }
        continue;
      }

      timestamps.push_back(timestamp);

      for (int i = 0; i < NUM_CAMS; i++) {
        TimeCamId tcid(id, i);

        std::stringstream ss;
        ss << dataset_path << "/" << timestamp << "_" << i << ".jpg";
        pangolin::TypedImage img = pangolin::LoadImage(ss.str());
        images[tcid] = std::move(img);
      }

      id++;
    }

    std::cerr << "Loaded " << id << " images " << std::endl;
  }

  {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib_cam);

      if (calib_cam.intrinsics.size() >= NUM_CAMS) {
        std::cout << "Loaded camera from " << calib_path << std::endl;
      } else {
        std::cout << "Calibration " << calib_path << " only has "
                  << calib_cam.intrinsics.size() << " cameras, but we require "
                  << NUM_CAMS << "." << std::endl;
        std::abort();
      }
    } else {
      std::cerr << "could not load camera calibration " << calib_path
                << std::endl;
      std::abort();
    }
  }

  {
    std::ifstream os(corners_path, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryInputArchive archive(os);
      archive(feature_corners);
      if (feature_corners.size() == images.size()) {
        std::cout << "Loaded cached corners from " << corners_path << std::endl;
      } else {
        std::cerr << "Ignoring cached corners from " << corners_path
                  << " (contains corners for " << feature_corners.size()
                  << " images, but we have now loaded " << images.size()
                  << " images)." << std::endl;
        feature_corners.clear();
      }
    } else {
      std::cerr << "could not load cached corners from " << corners_path
                << std::endl;
    }
  }

  {
    std::ifstream os(matches_path, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryInputArchive archive(os);
      archive(feature_matches);
      std::set<TimeCamId> tcids_matches;
      for (const auto& kv : feature_matches) {
        tcids_matches.insert(kv.first.first);
        tcids_matches.insert(kv.first.second);
      }
      if (tcids_matches.size() == images.size()) {
        std::cout << "Loaded cached matches from " << matches_path << std::endl;
      } else {
        std::cerr << "Ignoring cached matches from " << matches_path
                  << " (contains matches for " << tcids_matches.size()
                  << " images, but we have now loaded " << images.size()
                  << " images)." << std::endl;
        feature_matches.clear();
      }
    } else {
      std::cerr << "could not load cached matches from " << matches_path
                << std::endl;
    }
  }

  show_frame1.Meta().range[1] = images.size() / NUM_CAMS - 1;
  show_frame1.Meta().gui_changed = true;
  show_frame2.Meta().range[1] = images.size() / NUM_CAMS - 1;
  show_frame2.Meta().gui_changed = true;
}

// save the whole map including features, matches, tracks, etc
void save_map() {
  save_map_file(map_path, feature_corners, feature_matches, feature_tracks,
                outlier_tracks, cameras, landmarks);
}

// load a saved map
void load_map() {
  // clear everything
  clear_keypoints();

  // load map
  load_map_file(map_path, feature_corners, feature_matches, feature_tracks,
                outlier_tracks, cameras, landmarks);

  // check loaded image data matches
  if (!(images.size() == feature_corners.size() ||
        feature_corners.size() == 0)) {
    std::cerr << "Warning: Count of loaded images (" << images.size()
              << ") and loaded map (" << feature_corners.size()
              << ") doesn't match. Unloading map." << std::endl;

    // clear everything
    clear_keypoints();
  }

  // compute projection cache
  compute_projections();
}

// delete features and dependent data structures
void clear_keypoints() {
  feature_corners.clear();
  clear_matches();
}

// delete feature matches and dependent data structures
void clear_matches() {
  feature_matches.clear();
  clear_tracks();
}

// delete feature tracks and dependent data structures
void clear_tracks() {
  feature_tracks.clear();
  outlier_tracks.clear();
  clear_map();
}

// delete map data structures
void clear_map() {
  cameras.clear();
  landmarks.clear();
  camera_candidates = CameraCandidates();
  image_projections.clear();
  track_projections.clear();
}

// reconstruction stage to string for output
std::string stage_to_str(const CameraCandidates::Stage stage) {
  switch (stage) {
    case CameraCandidates::ComputeCandidates:
      return "select new camera candidates";
    case CameraCandidates::AddCameras:
      return "add cameras";
    case CameraCandidates::AddLandmarks:
      return "add landmarks";
    case CameraCandidates::Optimize:
      return "optimize";
    case CameraCandidates::RemoveOutliers:
      return "remove outliers";
    case CameraCandidates::Done:
      return "admire the result";
  }

  return "Unreachable!";
}

// print the next suggested step on console if enabled
void print_proceed_to(const std::string& str) {
  if (show_next_step_hint) {
    std::cerr << "Proceed to " << str << "." << std::endl;
  }
}

// print the next suggested step on console if enabled
void print_proceed_to(const CameraCandidates::Stage stage) {
  if (show_next_step_hint) {
    std::cerr << "Proceed to " << stage_to_str(stage) << "." << std::endl;
  }
}

// print a warning if step was called out-of-order (can still be useful for
// experimenting)
void check_current_stage_equal_to(const CameraCandidates::Stage stage) {
  if (stage != camera_candidates.current_stage) {
    std::cerr << "Warning: You " << stage_to_str(stage)
              << " even though current stage is '"
              << stage_to_str(camera_candidates.current_stage) << "'."
              << std::endl;
  }
}

// Execute next step in the overall SfM pipeline. Call this repeatedly until it
// returns false for automatic execution.
bool next_step() {
  if (feature_corners.empty()) {
    detect_keypoints();
    return true;
  }

  if (feature_matches.empty()) {
    match_stereo();
    match_all();
    return true;
  }

  if (feature_tracks.empty()) {
    build_tracks();
    return true;
  }

  if (cameras.empty()) {
    initialize_scene();
    return true;
  }

  switch (camera_candidates.current_stage) {
    case CameraCandidates::ComputeCandidates:
      compute_camera_candidate_set();
      return true;
    case CameraCandidates::AddCameras:
      add_next_camera();
      return true;
    case CameraCandidates::AddLandmarks:
      add_new_landmarks();
      return true;
    case CameraCandidates::Optimize:
      optimize();
      return true;
    case CameraCandidates::RemoveOutliers:
      remove_outlier_landmarks();
      return true;
    case CameraCandidates::Done:
      summary();
      // the end
      return false;
  }

  std::cerr << "Unreachable!" << std::endl;
  return false;
}

// Print a summary of the built map.
void summary() {
  size_t num_obs = 0;
  size_t num_outlier_obs = 0;

  for (const auto& kv : landmarks) {
    num_obs += kv.second.obs.size();
    num_outlier_obs += kv.second.outlier_obs.size();
  }

  std::cerr << "The map has " << cameras.size() << " cameras and "
            << landmarks.size() << " landmarks with " << num_obs
            << " observations. " << outlier_tracks.size()
            << " landmarks were removed as outliers and " << num_outlier_obs
            << " observations were marked as outliers." << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
/// Here the algorithmically interesting implementation begins
///////////////////////////////////////////////////////////////////////////////

// Detect features and compute descriptors
void detect_keypoints() {
  clear_keypoints();

  for (const auto& kv : images) {
    KeypointsData kd;

    detectKeypointsAndDescriptors(kv.second, kd, num_features_per_image,
                                  rotate_features);

    feature_corners[kv.first] = kd;
  }

  {
    std::ofstream os(corners_path, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryOutputArchive archive(os);
      archive(feature_corners);
      std::cout << "Saved features as " << corners_path << std::endl;
    }
  }

  print_proceed_to("matching stereo pairs");
}

// Feature matching and inlier filtering for stereo pairs with known pose
void match_stereo() {
  clear_tracks();

  const int num_images = images.size() / NUM_CAMS;

  // Pose of camera 1 (right) w.r.t camera 0 (left)
  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];

  // Essential matrix
  Eigen::Matrix3d E;
  computeEssential(T_0_1, E);

  std::cerr << "Matching " << num_images << " stereo pairs..." << std::endl;

  int num_matches = 0;
  int num_inliers = 0;

  for (int i = 0; i < num_images; i++) {
    const TimeCamId tcid1(i, 0), tcid2(i, 1);

    MatchData md;
    md.T_i_j = T_0_1;

    const KeypointsData& kd1 = feature_corners[tcid1];
    const KeypointsData& kd2 = feature_corners[tcid2];

    matchDescriptors(kd1.corner_descriptors, kd2.corner_descriptors, md.matches,
                     feature_match_max_dist, feature_match_test_next_best);

    num_matches += md.matches.size();

    findInliersEssential(kd1, kd2, calib_cam.intrinsics[0],
                         calib_cam.intrinsics[1], E, 1e-3, md);

    num_inliers += md.inliers.size();

    feature_matches[std::make_pair(tcid1, tcid2)] = md;
  }

  std::cerr << "Matched " << num_images << " stereo pairs with " << num_inliers
            << " inlier matches (" << num_matches << " total)." << std::endl;

  {
    std::ofstream os(matches_path, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryOutputArchive archive(os);
      archive(feature_matches);
      std::cout << "Saved matches as " << matches_path << std::endl;
    }
  }

  print_proceed_to("matching all pairs");
}

// Feature matching and inlier filtering for non-stereo pairs with unknown pose
void match_all() {
  clear_tracks();

  std::vector<TimeCamId> keys;

  for (const auto& kv : images) keys.push_back(kv.first);

  std::vector<std::pair<int, int>> ids_to_match;

  for (size_t i = 0; i < keys.size(); i++) {
    for (size_t j = i + 1; j < keys.size(); j++) {
      // Do not add stereo pairs (have same timestamp)
      if (keys[i].first != keys[j].first) ids_to_match.emplace_back(i, j);
    }
  }

  std::cerr << "Matching " << ids_to_match.size() << " image pairs..."
            << std::endl;

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, ids_to_match.size()),
      [&](const tbb::blocked_range<size_t>& r) {
        for (size_t j = r.begin(); j != r.end(); ++j) {
          const TimeCamId& id1 = keys[ids_to_match[j].first];
          const TimeCamId& id2 = keys[ids_to_match[j].second];

          const KeypointsData& f1 = feature_corners[id1];
          const KeypointsData& f2 = feature_corners[id2];

          MatchData md;

          matchDescriptors(f1.corner_descriptors, f2.corner_descriptors,
                           md.matches, feature_match_max_dist,
                           feature_match_test_next_best);

          if (int(md.matches.size()) > relative_pose_ransac_min_inliers) {
            findInliersRansac(f1, f2, calib_cam.intrinsics[id1.second],
                              calib_cam.intrinsics[id2.second],
                              relative_pose_ransac_thresh,
                              relative_pose_ransac_min_inliers, md);
          }

          feature_matches[std::make_pair(id1, id2)] = md;
        }
      });

  //
  int num_matches = 0;
  int num_inliers = 0;

  for (const auto& kv : feature_matches) {
    num_matches += kv.second.matches.size();
    num_inliers += kv.second.inliers.size();
  }

  std::cerr << "Matched " << ids_to_match.size() << " image pairs with "
            << num_inliers << " inlier matches (" << num_matches << " total)."
            << std::endl;

  {
    std::ofstream os(matches_path, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryOutputArchive archive(os);
      archive(feature_matches);
      std::cout << "Saved matches as " << matches_path << std::endl;
    }
  }

  print_proceed_to("building feature tracks");
}

// Small helper to debug matching. Starting from the first camera, compute the
// set of reachable cameras based on the pairwise matches.
void unreachable() {
  std::set<TimeCamId> all_tcid;

  for (auto& kv : feature_corners) {
    all_tcid.emplace(kv.first);
  }

  std::cout << "all_tcid.size() " << all_tcid.size() << std::endl;

  std::set<TimeCamId> reachable_tcid;
  reachable_tcid.emplace(*all_tcid.begin());

  int num_added = 1;

  while (num_added > 0) {
    num_added = 0;

    for (auto& kv : feature_matches) {
      const TimeCamId& t1 = kv.first.first;
      const TimeCamId& t2 = kv.first.second;

      if (reachable_tcid.find(t1) != reachable_tcid.end() &&
          reachable_tcid.find(t2) == reachable_tcid.end() &&
          int(kv.second.inliers.size()) > minimal_localization_inlier_count) {
        reachable_tcid.emplace(t2);

        num_added++;
      }

      if (reachable_tcid.find(t2) != reachable_tcid.end() &&
          reachable_tcid.find(t1) == reachable_tcid.end() &&
          int(kv.second.inliers.size()) > minimal_localization_inlier_count) {
        reachable_tcid.emplace(t1);

        num_added++;
      }
    }
  }

  std::cout << "reachable_tcid.size() " << reachable_tcid.size() << std::endl;

  std::vector<TimeCamId> diff_tcid(all_tcid.size());
  auto it = std::set_difference(all_tcid.begin(), all_tcid.end(),
                                reachable_tcid.begin(), reachable_tcid.end(),
                                diff_tcid.begin());

  diff_tcid.resize(it - diff_tcid.begin());

  for (const TimeCamId& t : diff_tcid) {
    std::cout << "frame " << t.first << " cam " << t.second << std::endl;
  }
}

// Build feature tracks, that is the transitive closure of the pairwise matches
// with filtering of wrong or too short tracks.
void build_tracks() {
  clear_tracks();

  TrackBuilder trackBuilder;
  // Build: Efficient fusion of correspondences
  trackBuilder.Build(feature_matches);
  // Filter: Remove tracks that have conflict
  trackBuilder.Filter(min_track_length);
  // Export tree to usable data structure
  trackBuilder.Export(feature_tracks);

  // info
  size_t inlier_match_count = 0;
  for (const auto& it : feature_matches) {
    inlier_match_count += it.second.inliers.size();
  }

  size_t total_track_obs_count = 0;
  for (const auto& it : feature_tracks) {
    total_track_obs_count += it.second.size();
  }

  std::cerr << "Built " << feature_tracks.size() << " feature tracks from "
            << inlier_match_count << " matches. Average track length is "
            << total_track_obs_count / (double)feature_tracks.size() << "."
            << std::endl;

  print_proceed_to("initialize the scene");
}

// Select first stereo pair and initialize scene with the known transformation
// and triangulated features
void initialize_scene() {
  clear_map();

  const TimeCamId tcid0(0, 0);
  const TimeCamId tcid1(0, 1);

  if (!initialize_scene_from_stereo_pair(tcid0, tcid1, calib_cam,
                                         feature_corners, feature_tracks,
                                         cameras, landmarks)) {
    std::cerr << "Failed to initialize map with first stereo pair."
              << std::endl;
    return;
  }

  // update image views
  change_display_to_image(tcid0);
  change_display_to_image(tcid1);

  // update projection info cache
  compute_projections();

  // info
  std::cerr << "Initialized scene with " << cameras.size() << " cameras and "
            << landmarks.size() << " landmarks." << std::endl;

  // next step should be optimizating the map
  camera_candidates.current_stage = CameraCandidates::Optimize;
  print_proceed_to(camera_candidates.current_stage);
}

// Select a sorted list of next camera candidates fulfilling the given inlier
// count threshold
void next_camera_candidate_set(CameraCandidates& candidate_set) {
  candidate_set.cameras.clear();
  for (int i = 0; i < int(timestamps.size()); ++i) {
    for (int j = 0; j < NUM_CAMS; ++j) {
      const TimeCamId tcid = std::make_pair(i, j);

      // check if camera is not yet in map
      if (cameras.count(tcid) == 0) {
        // compute shared tracks of existing map and current image
        std::vector<TrackId> shared_tracks_curr;
        GetSharedTracks(tcid, feature_tracks, landmarks, shared_tracks_curr);

        // check if potential candidate
        if (int(shared_tracks_curr.size()) >=
            candidate_set.min_localization_inliers) {
          CameraCandidate candidate;
          candidate.tcid = tcid;
          candidate.shared_tracks = std::move(shared_tracks_curr);
          candidate_set.cameras.push_back(std::move(candidate));
        }
      }
    }
  }

  // sort by number of shared tracks in descending order
  std::sort(candidate_set.cameras.begin(), candidate_set.cameras.end(),
            [](const CameraCandidate& a, const CameraCandidate& b) {
              return a.shared_tracks.size() > b.shared_tracks.size();
            });
}

// Compute set of camera candidates to add to the map next, first with a high
// threhold and if unseccessful with a lower threshold.
void compute_camera_candidate_set() {
  // check current stage
  check_current_stage_equal_to(CameraCandidates::ComputeCandidates);

  // check previous stage and warn if we missed something
  size_t num_tried = 0;
  size_t num_added = 0;
  for (const auto& candidate : camera_candidates.cameras) {
    if (!candidate.tried) {
      // not trying is ok, since we will just select again
      //      std::cerr << "Camera " << candidate.tcid
      //                << " in previous set was not tried." << std::endl;
    } else {
      ++num_tried;
      if (candidate.camera_added) {
        ++num_added;
        if (!candidate.landmarks_added) {
          // not having added landmarks means we will never add them
          std::cerr << "Waring: Camera " << candidate.tcid
                    << " was added in previous set, but landmarks not added."
                    << std::endl;
        }
      }
    }
  }

  const int num_cameras_remaining = int(images.size()) - int(cameras.size());

  // either this is the first round, or we check that at least one camera was
  // added in last round (if any were tried)
  const bool previous_attempt_failed =
      camera_candidates.min_localization_inliers > 0 &&
      (num_tried > 0 && num_added == 0);

  // clear candidate set
  camera_candidates.cameras.clear();
  camera_candidates.current_stage = CameraCandidates::Done;

  if (num_cameras_remaining <= 0) {
    std::cerr << "Cannot select candidate set. All " << cameras.size()
              << " have already been added. That's it..." << std::endl;
    return;
  } else {
    // compute new candidate set with desired threshold
    if (!previous_attempt_failed) {
      // previous candidate set was successful or this is the first round, so
      // try with desired (high) threshold
      camera_candidates.min_localization_inliers =
          desired_localization_inlier_count;
      camera_candidates.max_cameras_to_add = desired_inlier_max_cameras_to_add;
      next_camera_candidate_set(camera_candidates);
    }

    if (camera_candidates.cameras.size() == 0) {
      if (!previous_attempt_failed) {
        // 2nd attempt with lower threshold
        std::cerr << "Did not find any camera candidates (shared track thresh: "
                  << camera_candidates.min_localization_inliers << ")."
                  << std::endl;
      } else if (camera_candidates.min_localization_inliers <=
                 minimal_localization_inlier_count) {
        // no luck previous round with lowest threshold --> no need to try again
        std::cerr
            << "Previous candidate set with minimal shared track threshold "
            << camera_candidates.min_localization_inliers
            << " didn't result in any added camera, so don't try again. "
               "There are "
            << num_cameras_remaining << " cameras left. That's it..."
            << std::endl;
        return;
      }

      // compute new candidate set with minimal threshold
      camera_candidates.min_localization_inliers =
          minimal_localization_inlier_count;
      camera_candidates.max_cameras_to_add = minimal_inlier_max_cameras_to_add;
      next_camera_candidate_set(camera_candidates);

      if (camera_candidates.cameras.size() == 0) {
        std::cerr << "Did not find any camera candidates (shared track thresh: "
                  << camera_candidates.min_localization_inliers
                  << "). There are " << num_cameras_remaining
                  << " cameras left. That's it..." << std::endl;
        return;
      }
    }
  }

  std::cerr << "Selected " << camera_candidates.cameras.size()
            << " camera candidates from " << num_cameras_remaining
            << " remaining cameras (shared track thresh: "
            << camera_candidates.min_localization_inliers << ")." << std::endl;

  // success, we have a new candidate set, so proceed to adding cameras
  camera_candidates.current_stage = CameraCandidates::AddCameras;
  print_proceed_to(camera_candidates.current_stage);
}

// This is intended to be called after the camera candidate set has been
// computed. Adds another camera from the candidate set to the scene.
void add_next_camera() {
  // check current stage
  check_current_stage_equal_to(CameraCandidates::AddCameras);

  // find next candidate to try
  CameraCandidate* candidate = nullptr;
  size_t i = 0;
  int num_added = 0;
  for (auto& c : camera_candidates.cameras) {
    if (c.camera_added) {
      ++num_added;
    } else if (!c.tried) {
      c.tried = true;
      candidate = &c;
      break;
    }
    ++i;
  }

  // only try to add camera if we have found a candidate and if we haven't
  // reached the maximum number of candidates to add in one go
  if (nullptr == candidate) {
    std::cerr << "No more candidates (out of "
              << camera_candidates.cameras.size() << ") to try. Total added "
              << num_added << "." << std::endl;
  } else if (num_added < camera_candidates.max_cameras_to_add) {
    // We have a new camera to add
    const TimeCamId tcid = candidate->tcid;

    // Localize camera in map with PnP
    std::vector<TrackId> inlier_track_ids;
    Sophus::SE3d T_w_c;
    localize_camera(tcid, candidate->shared_tracks, calib_cam, feature_corners,
                    feature_tracks, landmarks,
                    reprojection_error_pnp_inlier_threshold_pixel, T_w_c,
                    inlier_track_ids);

    // actually still take all observations if option is set
    if (always_add_all_observations) {
      inlier_track_ids = candidate->shared_tracks;
    }

    // Check if we have enough inliers.
    if (int(inlier_track_ids.size()) <
        camera_candidates.min_localization_inliers) {
      // cannot accept this camera
      std::cerr << "Cannot add camera " << tcid << " (" << i + 1 << " of "
                << camera_candidates.cameras.size() << ") with "
                << inlier_track_ids.size() << " localization inlier ("
                << candidate->shared_tracks.size() - inlier_track_ids.size()
                << " outlier ignored)." << std::endl;
    } else {
      // add camera to map
      cameras[tcid].T_w_c = T_w_c;
      candidate->camera_added = true;
      ++num_added;

      // Add landmark observations for inlier and outlier tracks
      std::set<TrackId> inlier_set(inlier_track_ids.begin(),
                                   inlier_track_ids.end());
      for (const TrackId track_id : candidate->shared_tracks) {
        if (inlier_set.count(track_id) > 0) {
          // inlier
          landmarks.at(track_id).obs[tcid] =
              feature_tracks.at(track_id).at(tcid);
        } else {
          // outlier
          landmarks.at(track_id).outlier_obs[tcid] =
              feature_tracks.at(track_id).at(tcid);
        }
      }

      // info
      std::cerr << "Camera " << tcid << " (" << i + 1 << " of "
                << camera_candidates.cameras.size()
                << ") added to map observing " << inlier_track_ids.size()
                << " landmarks ("
                << candidate->shared_tracks.size() - inlier_track_ids.size()
                << " outlier ignored)." << std::endl;

      // update projection info cache
      compute_projections();

      // Update image view
      change_display_to_image(tcid);
    }
  }

  // check if we should advance to the next stage
  bool more_to_add = true;
  if (i + 1 >= camera_candidates.cameras.size()) {
    // no more current camera candidates to try
    more_to_add = false;
  } else if (num_added >= camera_candidates.max_cameras_to_add) {
    // reached max number
    std::cerr << "Reached maximum number of " << num_added << " (out of "
              << camera_candidates.cameras.size()
              << ") cameras to add in one go." << std::endl;
    more_to_add = false;
  }

  // move to the next stage if no more candidates to try
  if (!more_to_add) {
    if (camera_candidates.num_cameras_added() > 0) {
      // cameras added, proceed to adding landmarks
      camera_candidates.current_stage = CameraCandidates::AddLandmarks;

    } else {
      // no cameras added, move to new canddiate set
      camera_candidates.current_stage = CameraCandidates::ComputeCandidates;
    }
  }

  print_proceed_to(camera_candidates.current_stage);
}

// This is intended to be called after all cameras from the candidate set have
// been added (or tried at least). Creates new landmarks from the new cameras.
void add_new_landmarks() {
  // check current stage
  check_current_stage_equal_to(CameraCandidates::AddLandmarks);

  // Determine next camera for which to add landmarks
  CameraCandidate* candidate = nullptr;
  size_t i = 0;
  for (auto& c : camera_candidates.cameras) {
    if (c.camera_added && !c.landmarks_added) {
      c.landmarks_added = true;
      candidate = &c;
      break;
    }
    ++i;
  }

  bool more_to_add = false;

  if (nullptr == candidate) {
    std::cerr << "No more cameras for which to add landmarks." << std::endl;
  } else {
    const TimeCamId tcid = candidate->tcid;

    // Find overlapping tracks that are not yet landmarks and add to scene.
    // We go through existing cams one by one to triangulate landmarks
    // pair-wise. If there are additional cameras in the existing map also
    // sharing the same track, we add observations after triangulation.
    int new_landmark_count = 0;
    for (const auto& kv : cameras) {
      const TimeCamId& tcid_existing = kv.first;
      if (tcid_existing != tcid) {
        new_landmark_count += add_new_landmarks_between_cams(
            tcid_existing, tcid, calib_cam, feature_corners, feature_tracks,
            cameras, landmarks);
      }
    }

    std::cerr << "Added " << new_landmark_count << " new landmarks for image "
              << tcid << "." << std::endl;

    // update projection info cache
    compute_projections();

    // return true if there are more cameras for which to add landmarks
    more_to_add = camera_candidates.num_landmarks_added() <
                  camera_candidates.num_cameras_added();
  }

  if (!more_to_add) {
    // proceed to optimization after all landmarks have been added
    camera_candidates.current_stage = CameraCandidates::Optimize;
  }

  print_proceed_to(camera_candidates.current_stage);
}

// Optimize the map with bundle adjustment
void optimize() {
  // check current stage
  check_current_stage_equal_to(CameraCandidates::Optimize);

  // info on how many cameras were added (special case: first optimization after
  // initialization)
  size_t num_obs = 0;
  for (const auto& kv : landmarks) {
    num_obs += kv.second.obs.size();
  }
  const int num_cameras_new = camera_candidates.min_localization_inliers == 0
                                  ? cameras.size()
                                  : camera_candidates.num_cameras_added();
  std::cerr << "Optimizing map with " << cameras.size() << " cameras ("
            << num_cameras_new << " new), " << landmarks.size()
            << " points and " << num_obs << " observations." << std::endl;

  // Fix first two cameras to fix SE3 and scale gauge. Making the whole second
  // camera constant is a bit suboptimal, since we only need 1 DoF, but it's
  // simple and the initial poses should be good from calibration.
  std::set<TimeCamId> fixed_cameras = {{0, 0}, {0, 1}};

  // Run bundle adjustment
  BundleAdjustmentOptions ba_options;
  ba_options.optimize_intrinsics = ba_optimize_intrinsics;
  ba_options.use_huber = true;
  ba_options.huber_parameter = reprojection_error_huber_pixel;
  ba_options.max_num_iterations = 20;
  ba_options.verbosity_level = ba_verbose;
  bundle_adjustment(feature_corners, ba_options, fixed_cameras, calib_cam,
                    cameras, landmarks, false);

  // Update project info cache
  compute_projections();

  // advance to next stage (unless it was called out-of-order)
  if (CameraCandidates::Optimize == camera_candidates.current_stage) {
    // Next stage: remove outliers
    camera_candidates.current_stage = CameraCandidates::RemoveOutliers;
  }

  print_proceed_to(camera_candidates.current_stage);
}

// helper for computing outlier flags for a projected landmark
void set_outlier_flags(ProjectedLandmark& lm_proj) {
  // 1. check for huge reprojection error
  if (lm_proj.reprojection_error >
      reprojection_error_outlier_threshold_huge_pixel) {
    lm_proj.outlier_flags |= OutlierReprojectionErrorHuge;
  }

  // 2. check for large reprojection error
  if (lm_proj.reprojection_error >
      reprojection_error_outlier_threshold_normal_pixel) {
    lm_proj.outlier_flags |= OutlierReprojectionErrorNormal;
  }

  // 3. check for landmarks that are too close to a camera center --> may
  // correspond to outlier matches or points stuck in local minima
  const double distance_to_camera = lm_proj.point_3d_c.norm();
  if (distance_to_camera < camera_center_distance_outlier_threshold_meter) {
    lm_proj.outlier_flags |= OutlierCameraDistance;
  }

  // 4. check for landmarks with too small z coordinate for some camera -->
  // may correspond to outlier matches or points stuck in local minima
  if (lm_proj.point_3d_c.z() < z_coordinate_outlier_threshold_meter) {
    lm_proj.outlier_flags |= OutlierZCoordinate;
  }
}

// Compute reprojections for all landmark observations for visualization and
// outlier removal.
void compute_projections() {
  image_projections.clear();
  track_projections.clear();

  for (const auto& kv_lm : landmarks) {
    const TrackId track_id = kv_lm.first;

    for (const auto& kv_obs : kv_lm.second.obs) {
      const TimeCamId& tcid = kv_obs.first;
      const Eigen::Vector2d p_2d_corner =
          feature_corners.at(tcid).corners[kv_obs.second];

      const Eigen::Vector3d p_c =
          cameras.at(tcid).T_w_c.inverse() * kv_lm.second.p;
      const Eigen::Vector2d p_2d_repoj =
          calib_cam.intrinsics.at(tcid.second)->project(p_c);

      ProjectedLandmarkPtr proj_lm(new ProjectedLandmark);
      proj_lm->track_id = track_id;
      proj_lm->point_measured = p_2d_corner;
      proj_lm->point_reprojected = p_2d_repoj;
      proj_lm->point_3d_c = p_c;
      proj_lm->reprojection_error = (p_2d_corner - p_2d_repoj).norm();

      set_outlier_flags(*proj_lm);

      image_projections[tcid].obs.push_back(proj_lm);
      track_projections[track_id][tcid] = proj_lm;
    }

    for (const auto& kv_obs : kv_lm.second.outlier_obs) {
      const TimeCamId& tcid = kv_obs.first;
      const Eigen::Vector2d p_2d_corner =
          feature_corners.at(tcid).corners[kv_obs.second];

      const Eigen::Vector3d p_c =
          cameras.at(tcid).T_w_c.inverse() * kv_lm.second.p;
      const Eigen::Vector2d p_2d_repoj =
          calib_cam.intrinsics.at(tcid.second)->project(p_c);

      ProjectedLandmarkPtr proj_lm(new ProjectedLandmark);
      proj_lm->track_id = track_id;
      proj_lm->point_measured = p_2d_corner;
      proj_lm->point_reprojected = p_2d_repoj;
      proj_lm->point_3d_c = p_c;
      proj_lm->reprojection_error = (p_2d_corner - p_2d_repoj).norm();

      image_projections[tcid].outlier_obs.push_back(proj_lm);
    }
  }
}

// Check if a given landmark is an outlier.
// Assumes that the track_id is currently a landmark.
bool is_landmark_outlier(const TrackId track_id) {
  // check if any of the observations has outlier flags
  for (const auto& kv : track_projections.at(track_id)) {
    if (kv.second->outlier_flags != OutlierNone) {
      return true;
    }
  }
  return false;
}

// Remove outlier landmarks from the map based on the flags from
// compute_projections. The normal reprojection error threshold is only used if
// no other types of outliers are present, since the later can distort the map
// heavily and cause large reprojection errors even for inlier observations.
// Afterwards we should optimize the map and look again for new or remaining
// outliers until no more are removed.
void remove_outlier_landmarks() {
  // check current stage
  check_current_stage_equal_to(CameraCandidates::RemoveOutliers);

  int num_repr_error_normal = 0;
  int num_repr_error_huge = 0;
  int num_camera_center_distance = 0;
  int num_z_coordinate = 0;

  // check if there are any outliers apart from the normal reprojection error
  bool any_severe_outliers = false;
  for (const auto& kv_tracks : track_projections) {
    for (const auto& kv_proj : kv_tracks.second) {
      if (kv_proj.second->outlier_flags & ~OutlierReprojectionErrorNormal) {
        any_severe_outliers = true;
        break;
      }
    }
    if (any_severe_outliers) {
      break;
    }
  }

  // Go through projections and remove marked outliers.
  // OutlierReprojectionErrorNormal is only removed if no other types of
  // outliers are present.
  for (const auto& kv_tracks : track_projections) {
    bool remove = false;
    bool repr_error_normal_counted = false;

    for (const auto& kv_proj : kv_tracks.second) {
      // 1. much too large repr. error
      if (kv_proj.second->outlier_flags & OutlierReprojectionErrorHuge) {
        ++num_repr_error_huge;
        remove = true;
        break;
      }

      // 2. too large repro. error (only if no other types)
      if (kv_proj.second->outlier_flags & OutlierReprojectionErrorNormal) {
        if (!repr_error_normal_counted) {
          ++num_repr_error_normal;
          repr_error_normal_counted = true;
        }
        if (!any_severe_outliers) {
          remove = true;
          break;
        }
      }

      // 3. too small distance to camera
      if (kv_proj.second->outlier_flags & OutlierCameraDistance) {
        remove = true;
        ++num_camera_center_distance;
        break;
      }

      // 4. too small z coordinate
      if (kv_proj.second->outlier_flags & OutlierZCoordinate) {
        remove = true;
        ++num_z_coordinate;
        break;
      }
    }

    if (remove) {
      // outlier observation --> remove landmark and flag feature track
      const auto iter = feature_tracks.find(kv_tracks.first);
      outlier_tracks.insert(*iter);
      feature_tracks.erase(iter);
      landmarks.erase(kv_tracks.first);
    }
  }

  // update projections
  compute_projections();

  // info and next step
  const int num_total = any_severe_outliers
                            ? (num_repr_error_huge +
                               num_camera_center_distance + num_z_coordinate)
                            : num_repr_error_normal;
  if (num_total > 0) {
    if (any_severe_outliers) {
      std::cerr << num_total << " outliers removed (" << num_repr_error_huge
                << " for huge repr. error (" << num_repr_error_normal
                << " not removed), " << num_camera_center_distance
                << " too close to camera center, " << num_z_coordinate
                << " too small z)." << std::endl;
    } else {
      std::cerr << num_total << " outliers removed for too large repr. error."
                << std::endl;
    }
  }

  // advance to next stage (unless it was called out-of-order)
  if (CameraCandidates::RemoveOutliers == camera_candidates.current_stage) {
    // If we removed outliers --> optimize, else --> next round
    camera_candidates.current_stage = num_total > 0
                                          ? CameraCandidates::Optimize
                                          : CameraCandidates::ComputeCandidates;
  }

  print_proceed_to(camera_candidates.current_stage);
}
