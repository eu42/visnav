#include <set>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

#include <visnav/calibration.h>
#include <visnav/common_types.h>

namespace visnav {

// keep track of next available unique id for detected keypoints
int last_keypoint_id = 0;

void filter_points(
    std::vector<std::map<int, Eigen::AffineCompact2f>>& transforms,
    Calibration& calib_cam, Eigen::Matrix3d& E,
    double epipolar_error_threshold) {
  std::vector<int> keypoints_to_filter;

  for (const auto& [id1, transform1] : transforms.at(1)) {
    auto it = transforms.at(0).find(id1);

    // keypoint is detected in both left and right images
    if (it != transforms.at(0).end()) {
      Eigen::Vector2d p_l = it->second.translation().cast<double>();
      Eigen::Vector2d p_r = transform1.translation().cast<double>();

      Eigen::Vector3d proj_l = calib_cam.intrinsics.at(0)->unproject(p_l);
      Eigen::Vector3d proj_r = calib_cam.intrinsics.at(1)->unproject(p_r);

      // check epipolar constraint
      if (abs(proj_l.transpose() * E * proj_r) > epipolar_error_threshold) {
        keypoints_to_filter.emplace_back(id1);
      }
    }
  }

  // remove keypoints from right camera if epipolar constraint is failed
  for (const auto& id : keypoints_to_filter) {
    transforms.at(1).erase(id);
  }
}

void track_points(const cv::Mat& image_0, const cv::Mat& image_1,
                  std::map<int, Eigen::AffineCompact2f>& transforms_0,
                  std::map<int, Eigen::AffineCompact2f>& transforms_1,
                  cv::TermCriteria termCrit, cv::Size win_size_cv,
                  int pyramid_level, double trackback_t) {
  if (transforms_0.size() == 0) return;

  std::vector<cv::Point2f> points_0_cv;
  std::vector<int> points_0_ids;
  for (const auto& [id, transform] : transforms_0) {
    Eigen::Vector2f point_v = transform.translation();
    points_0_cv.emplace_back(cv::Point2f(point_v[0], point_v[1]));
    points_0_ids.emplace_back(id);
  }

  std::vector<cv::Point2f> tracked;
  std::vector<uchar> status;
  std::vector<float> err;

  std::vector<cv::Mat> pyramid_0, pyramid_1;
  cv::buildOpticalFlowPyramid(image_0, pyramid_0, win_size_cv, pyramid_level);
  cv::buildOpticalFlowPyramid(image_1, pyramid_1, win_size_cv, pyramid_level);

  // track points from left image_0 to image_1
  cv::calcOpticalFlowPyrLK(pyramid_0, pyramid_1, points_0_cv, tracked, status,
                           err, win_size_cv, pyramid_level, termCrit);

  int count0 = 0;
  std::vector<int> tracked_indices;
  std::vector<cv::Point2f> tracked_points;
  // keep information for successfully tracked points
  for (size_t i = 0; i < points_0_cv.size(); ++i) {
    if (status[i] == 1) {
      cv::Point2f tracked_point = tracked[i];
      Eigen::Vector2f point_v(tracked_point.x, tracked_point.y);
      Eigen::Vector2f point_orig(points_0_cv[i].x, points_0_cv[i].y);

      tracked_indices.emplace_back(i);
      tracked_points.emplace_back(tracked_point);
    }
  }

  if (tracked_points.size() == 0) return;

  std::vector<cv::Point2f> tracked2;
  std::vector<uchar> status2;
  std::vector<float> err2;
  // track tracked points back to image_0 from image_1
  cv::calcOpticalFlowPyrLK(pyramid_1, pyramid_0, tracked_points, tracked2,
                           status2, err2, win_size_cv, pyramid_level, termCrit);

  for (size_t i = 0; i < tracked_points.size(); ++i) {
    if (status2[i] == 1) {
      cv::Point2f tracked_back = tracked2[i];
      Eigen::Vector2f point_v(tracked_back.x, tracked_back.y);

      int j = tracked_indices[i];
      Eigen::Vector2f point_orig(points_0_cv[j].x, points_0_cv[j].y);

      // skip if tracked back point is not close enough to the original point
      float dist = (point_orig - point_v).squaredNorm();
      if (dist > trackback_t) continue;

      Eigen::AffineCompact2f transform;
      transform.setIdentity();

      cv::Point2f tracked_point = tracked[j];
      Eigen::Vector2f tracked_point_v(tracked_point.x, tracked_point.y);
      transform.translation() = tracked_point_v;

      transforms_1[points_0_ids[j]] = transform;
    }
  }
}

void track_points_stereo(
    cv::Mat& image_0, cv::Mat& image_1,
    std::vector<std::map<int, Eigen::AffineCompact2f>>& transforms,
    Calibration& calib_cam, Eigen::Matrix3d& E, cv::TermCriteria termCrit,
    cv::Size win_size, int pyramid_level, double trackback_t,
    double epipolar_t) {
  track_points(image_0, image_1, transforms.at(0), transforms.at(1), termCrit,
               win_size, pyramid_level, trackback_t);
  filter_points(transforms, calib_cam, E, epipolar_t);
}

void detect_keypoints(const cv::Mat& image, KeypointsData& kd, int grid_x,
                      int grid_y, int points_per_cell, int fast_t, int edge_t,
                      int dist2_t,
                      const std::vector<Eigen::Vector2d>& tracked_points =
                          std::vector<Eigen::Vector2d>()) {
  kd.corners.clear();

  // Divide image into cells of patch_size
  // we will leave x_start and y_start pixels from beginning and end
  const int x_start = (image.cols % grid_x) / 2;
  const int x_end = x_start + image.cols - grid_x;
  const int y_start = (image.rows % grid_y) / 2;
  const int y_end = y_start + image.rows - grid_y;

  int cells_rows = image.rows / grid_y + 1;
  int cells_cols = image.cols / grid_x + 1;

  // mark cells of tracked points
  std::map<int, std::vector<int>> tracked_point_cells;
  for (int i = 0; i < tracked_points.size(); ++i) {
    const Eigen::Vector2d& p = tracked_points[i];
    if (x_start <= p[0] && y_start <= p[1]) {
      int x = (p[0] - x_start) / grid_x;
      int y = (p[1] - y_start) / grid_y;

      int cell_ind = y * cells_rows + x;
      tracked_point_cells[cell_ind].push_back(i);
    }
  }

  // traverse cells
  for (int x = x_start; x < x_end; x += grid_x) {
    for (int y = y_start; y < y_end; y += grid_y) {
      int cell_ind = ((int)(y - y_start) / grid_y) * cells_rows +
                     ((int)(x - x_start) / grid_x);

      cv::Mat cell_image =
          image(cv::Range(y, y + grid_y), cv::Range(x, x + grid_x));

      int points_added = 0;

      std::vector<cv::KeyPoint> keypoints, selected_keypoints;
      cv::FAST(cell_image, keypoints, fast_t);

      std::sort(keypoints.begin(), keypoints.end(),
                [](const cv::KeyPoint& i, const cv::KeyPoint& j) -> bool {
                  return i.response > j.response;
                });
      auto it = tracked_point_cells.find(cell_ind);

      if (it != tracked_point_cells.end()) {
        points_added = it->second.size();
      }

      for (size_t i = 0; i < keypoints.size() && points_added < points_per_cell;
           ++i) {
        int _x = x + keypoints[i].pt.x;
        int _y = y + keypoints[i].pt.y;

        // pass keypoints if they are too close to image edges
        if (edge_t > _x || edge_t > _y || _x > (image.cols - edge_t - 1) ||
            _y > (image.rows - edge_t - 1))
          continue;

        // check if keypoint is very close to any selected keypoint
        bool overlaps = false;
        if (it != tracked_point_cells.end()) {
          for (int j : it->second) {
            const Eigen::Vector2d& p = tracked_points[j];
            cv::Point2f _p = cv::Point(p[0] - x, p[1] - y);
            cv::Point2f diff = _p - keypoints[i].pt;
            float dist2 = diff.x * diff.x + diff.y * diff.y;
            if (dist2 < dist2_t) {
              overlaps = true;
              break;
            }
          }
        }
        if (overlaps) continue;
        for (const auto& k : selected_keypoints) {
          cv::Point2f diff = k.pt - keypoints[i].pt;
          float dist2 = diff.x * diff.x + diff.y * diff.y;
          if (dist2 < dist2_t) {
            overlaps = true;
            break;
          }
        }
        if (overlaps) continue;

        selected_keypoints.emplace_back(keypoints[i]);
        kd.corners.emplace_back(_x, _y);
        points_added++;
      }
    }
  }
}

void add_points(std::vector<std::map<int, Eigen::AffineCompact2f>>& transforms,
                cv::Mat& img_l, cv::Mat& img_r, int grid_x, int grid_y,
                int points_per_grid, int fast_t, int edge_t, int dist2_t,
                cv::TermCriteria termCrit, cv::Size win_size,
                bool track_new_points_to_right, int pyramid_level,
                double trackback_t) {
  std::vector<Eigen::Vector2d> points_l;

  // keep tracked points for left image
  for (const auto& kv : transforms.at(0)) {
    points_l.emplace_back(kv.second.translation().cast<double>());
  }

  KeypointsData new_points;

  detect_keypoints(img_l, new_points, grid_x, grid_y, points_per_grid, fast_t,
                   edge_t, dist2_t, points_l);

  std::map<int, Eigen::AffineCompact2f> new_poses_l, new_poses_r;

  for (size_t i = 0; i < new_points.corners.size(); ++i) {
    // it is newly found, there is no extra transformation than its position
    Eigen::AffineCompact2f transform;
    transform.setIdentity();
    transform.translation() = new_points.corners[i].cast<float>();

    transforms.at(0)[last_keypoint_id] = transform;
    new_poses_l[last_keypoint_id] = transform;

    last_keypoint_id++;
  }

  if (new_poses_l.size() > 0 && track_new_points_to_right) {
    // track points from left camera to right camera
    track_points(img_l, img_r, new_poses_l, new_poses_r, termCrit, win_size,
                 pyramid_level, trackback_t);
    for (const auto& kv : new_poses_r) {
      transforms.at(1).emplace(kv);
    }
  }
}

void localize_camera(const std::shared_ptr<AbstractCamera<double>>& cam,
                     const std::map<int, Eigen::AffineCompact2f>& transforms_l,
                     const Landmarks& landmarks,
                     const double reprojection_error_pnp_inlier_threshold_pixel,
                     MatchData& md, Sophus::SE3d& T_w_c,
                     std::vector<int>& inliers) {
  inliers.clear();

  T_w_c = Sophus::SE3d();

  opengv::bearingVectors_t bearing_vec;
  opengv::points_t points;

  // find keypoints that belong to a landmark
  for (const auto& [id0, transform0] : transforms_l) {
    auto it = landmarks.find(id0);

    if (it != landmarks.end()) {
      Eigen::Vector3d p_3d_unprojected =
          cam->unproject(transform0.translation().cast<double>());
      const Eigen::Vector3d& p_3d = it->second.p;

      bearing_vec.push_back(p_3d_unprojected);
      points.push_back(p_3d);

      // populate md to use later in add_new_landmarks.
      // `inliers` will contain indices from md.matches
      md.matches.emplace_back(id0, id0);
    }
  }

  if (points.size() == 0) return;

  // Find the pose (T_w_c) and the inliers using the landmark
  // to keypoints matches and PnP.

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
  // double focal_length = 500;
  double focal_length = (cam->data()[0] + cam->data()[1]) / 2;
  double threshold =
      1.0 - cos(atan(reprojection_error_pnp_inlier_threshold_pixel * 0.5 /
                     focal_length));
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

void add_new_landmarks(
    const TimeCamId tcidl, const TimeCamId tcidr,
    std::vector<std::map<int, Eigen::AffineCompact2f>> transforms,
    Calibration calib_cam, const Sophus::SE3d& T_w_c0, Landmarks& landmarks,
    std::vector<int>& inliers, MatchData md, MatchData md_stereo) {
  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // track stereo observations that are added to landmarks to find observations
  // not added in order to triangulate and add them
  std::set<int> added_inliers;
  // For all inliers add the observations to the existing landmarks.
  for (int inlier_id : inliers) {
    const int f_id = md.matches[inlier_id].first;

    Landmark& landmark = landmarks[f_id];

    landmark.obs.emplace(tcidl, f_id);

    // if the left point is in md_stereo.inliers then add both observations
    for (const auto& [corner_idl, corner_idr] : md_stereo.inliers) {
      int f_idl = md_stereo.corner0_transform_map[corner_idl];
      if (f_idl == f_id) {
        // f_idl = f_idr = f_id
        landmark.obs.emplace(tcidr, f_id);

        // track stereo observations that are added to existing landmarks
        added_inliers.emplace(f_id);

        break;
      }
    }
  }

  // create bearing vectors
  opengv::bearingVectors_t bearing_vec_0, bearing_vec_1;

  // For all stereo observations that were not added to the existing landmarks
  // triangulate and add new landmarks.
  for (const auto& [c_idl, c_idr] : md_stereo.inliers) {
    int f_idl = md_stereo.corner0_transform_map[c_idl];

    // inlier is not added to landmarks
    if (added_inliers.find(f_idl) == added_inliers.end()) {
      bearing_vec_0.push_back(calib_cam.intrinsics[0]->unproject(
          transforms[0].at(f_idl).translation().cast<double>()));
      bearing_vec_1.push_back(calib_cam.intrinsics[1]->unproject(
          transforms[1].at(f_idl).translation().cast<double>()));
    }
  }

  // create the central relative adapter
  opengv::relative_pose::CentralRelativeAdapter adapter(
      bearing_vec_0, bearing_vec_1, t_0_1, R_0_1);

  int landmarks_before = landmarks.size();

  int j = 0;

  for (const auto& [c_idl, c_idr] : md_stereo.inliers) {
    int f_idl = md_stereo.corner0_transform_map[c_idl];

    // inlier is not added to landmarks
    if (added_inliers.find(f_idl) == added_inliers.end()) {
      Landmark l;

      // triangulated point at world coordinate frame
      l.p = T_w_c0 * opengv::triangulation::triangulate(adapter, j++);

      // add observations to the new landmark
      l.obs.emplace(tcidl, f_idl);
      l.obs.emplace(tcidr, f_idl);

      // add new landmark to `landmarks`
      landmarks.emplace(f_idl, l);
    }
  }
}

}  // namespace visnav
