#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <iostream>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

// Parameters
const std::string CALIB_DATA_DIR = "./calibration_data";
const std::string LEFT_IMG_PATH = "./calibrationImages/test/imageLeft0.png";
const std::string RIGHT_IMG_PATH = "./calibrationImages/test/imageRight0.png";
const int DISPARITY_RANGE = 128;  // Must be divisible by 16
const int BLOCK_SIZE = 9;         // Odd number between 3-15
const float SCALE_FACTOR = 0.45f;  // Scale factor for output images

// Structure to hold calibration data
struct CalibrationData {
    cv::Mat left_map_x, left_map_y;
    cv::Mat right_map_x, right_map_y;
    cv::Mat Q;
    cv::Mat mtx_l, dist_l;
    cv::Mat mtx_r, dist_r;
    cv::Mat R, T;
    double baseline;
};

// Load calibration data
CalibrationData loadCalibration() {
    std::cout << "=== 🔄 Loading Calibration Data ===" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    CalibrationData data;

    // Load remap files
    cv::FileStorage fs_left_x(CALIB_DATA_DIR + "/left_map_x.npy", cv::FileStorage::READ);
    cv::FileStorage fs_left_y(CALIB_DATA_DIR + "/left_map_y.npy", cv::FileStorage::READ);
    cv::FileStorage fs_right_x(CALIB_DATA_DIR + "/right_map_x.npy", cv::FileStorage::READ);
    cv::FileStorage fs_right_y(CALIB_DATA_DIR + "/right_map_y.npy", cv::FileStorage::READ);
    
    fs_left_x["mat"] >> data.left_map_x;
    fs_left_y["mat"] >> data.left_map_y;
    fs_right_x["mat"] >> data.right_map_x;
    fs_right_y["mat"] >> data.right_map_y;
    
    fs_left_x.release();
    fs_left_y.release();
    fs_right_x.release();
    fs_right_y.release();

    // Load Q matrix
    cv::FileStorage fs_q(CALIB_DATA_DIR + "/Q.npy", cv::FileStorage::READ);
    fs_q["mat"] >> data.Q;
    fs_q.release();

    // Load intrinsics
    cv::FileStorage fs_intrinsics(CALIB_DATA_DIR + "/intrinsics.yml", cv::FileStorage::READ);
    fs_intrinsics["mtx_l"] >> data.mtx_l;
    fs_intrinsics["dist_l"] >> data.dist_l;
    fs_intrinsics["mtx_r"] >> data.mtx_r;
    fs_intrinsics["dist_r"] >> data.dist_r;
    fs_intrinsics["R"] >> data.R;
    fs_intrinsics["T"] >> data.T;
    data.baseline = fs_intrinsics["baseline_cm"];
    fs_intrinsics.release();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "• Loaded all calibration data in " << duration << "ms" << std::endl;
    std::cout << "• Baseline: " << data.baseline << " cm | Image size: " 
              << data.left_map_x.size[1] << "x" << data.left_map_x.size[0] << std::endl;
    
    return data;
}

// Stereo rectifier class
class StereoRectifier {
public:
    StereoRectifier(const cv::Mat& left_map_x, const cv::Mat& left_map_y,
                    const cv::Mat& right_map_x, const cv::Mat& right_map_y) 
        : left_map_x_(left_map_x), left_map_y_(left_map_y),
          right_map_x_(right_map_x), right_map_y_(right_map_y) {
        
        // Convert maps to float32 if needed
        if (left_map_x_.type() != CV_32FC1) {
            left_map_x_.convertTo(left_map_x_, CV_32FC1);
            left_map_y_.convertTo(left_map_y_, CV_32FC1);
            right_map_x_.convertTo(right_map_x_, CV_32FC1);
            right_map_y_.convertTo(right_map_y_, CV_32FC1);
        }
    }

    void rectify(const cv::Mat& left_img, const cv::Mat& right_img,
                 cv::Mat& rect_left, cv::Mat& rect_right) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // CPU implementation (GPU version would require CUDA)
        cv::remap(left_img, rect_left, left_map_x_, left_map_y_, cv::INTER_LINEAR);
        cv::remap(right_img, rect_right, right_map_x_, right_map_y_, cv::INTER_LINEAR);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::cout << "• Rectified via CPU in " << duration << "ms" << std::endl;
    }

private:
    cv::Mat left_map_x_, left_map_y_;
    cv::Mat right_map_x_, right_map_y_;
};

// Compute disparity with SGBM
void computeDisparity(const cv::Mat& rect_left, const cv::Mat& rect_right,
                      cv::Mat& disparity, cv::Mat& disparity_vis, bool use_wls = false) {
    std::cout << "\n=== 🎭 Computing Disparity ===" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat gray_left, gray_right;
    cv::cvtColor(rect_left, gray_left, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rect_right, gray_right, cv::COLOR_BGR2GRAY);

    if (use_wls) {
        // Left matcher
        auto left_matcher = cv::StereoSGBM::create(
            0,                              // minDisparity
            DISPARITY_RANGE,                // numDisparities
            BLOCK_SIZE,                     // blockSize
            8 * 3 * BLOCK_SIZE * BLOCK_SIZE, // P1
            32 * 3 * BLOCK_SIZE * BLOCK_SIZE, // P2
            1,                             // disp12MaxDiff
            10,                            // uniquenessRatio
            100,                           // speckleWindowSize
            2,                             // speckleRange
            63,                             // preFilterCap
            cv::StereoSGBM::MODE_HH        // mode
        );

        // Compute left disparity
        cv::Mat left_disp;
        left_matcher->compute(gray_left, gray_right, left_disp);
        left_disp.convertTo(left_disp, CV_32F, 1.0 / 16.0);

        // Right matcher for WLS filter
        auto right_matcher = cv::ximgproc::createRightMatcher(left_matcher);
        cv::Mat right_disp;
        right_matcher->compute(gray_right, gray_left, right_disp);
        right_disp.convertTo(right_disp, CV_32F, 1.0 / 16.0);

        // WLS filter parameters
        double lambda_wls = 8000.0;
        double sigma_wls = 1.5;

        // Create WLS filter
        auto wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
        wls_filter->setLambda(lambda_wls);
        wls_filter->setSigmaColor(sigma_wls);

        // Apply WLS filter
        wls_filter->filter(left_disp, gray_left, disparity, right_disp);

        std::cout << "• Computed via CPU SGBM with WLS filter" << std::endl;
    } else {
        // Regular SGBM
        auto stereo = cv::StereoSGBM::create(
            0,                              // minDisparity
            DISPARITY_RANGE,                // numDisparities
            BLOCK_SIZE,                     // blockSize
            8 * 3 * BLOCK_SIZE * BLOCK_SIZE, // P1
            32 * 3 * BLOCK_SIZE * BLOCK_SIZE, // P2
            1,                             // disp12MaxDiff
            10,                            // uniquenessRatio
            100,                           // speckleWindowSize
            2,                             // speckleRange
            63,                            // preFilterCap
            cv::StereoSGBM::MODE_SGBM_3WAY  // mode
        );

        stereo->compute(gray_left, gray_right, disparity);
        disparity.convertTo(disparity, CV_32F, 1.0 / 16.0);
        
        std::cout << "• Computed via CPU SGBM" << std::endl;
    }

    // Normalize for visualization
    cv::normalize(disparity, disparity_vis, 0, 255, cv::NORM_MINMAX, CV_8U);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "• Computed disparity in " << duration << "ms" << std::endl;
    std::cout << "• Disparity range: " << DISPARITY_RANGE 
              << " | Block size: " << BLOCK_SIZE << std::endl;
}

// Resize image helper function
cv::Mat resizeImage(const cv::Mat& img, float scale) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);
    return resized;
}

// Apply bilateral filter
cv::Mat applyBilateralFilter(const cv::Mat& img) {
    cv::Mat filtered;
    cv::bilateralFilter(img, filtered, 9, 75, 75);
    return filtered;
}

int main() {
    // Load calibration data
    CalibrationData calib_data = loadCalibration();

    // Initialize rectifier
    StereoRectifier rectifier(calib_data.left_map_x, calib_data.left_map_y,
                             calib_data.right_map_x, calib_data.right_map_y);

    // Load test images
    cv::Mat left_img = cv::imread(LEFT_IMG_PATH);
    cv::Mat right_img = cv::imread(RIGHT_IMG_PATH);
    
    if (left_img.empty() || right_img.empty()) {
        std::cerr << "Error loading images!" << std::endl;
        return -1;
    }

    // Rectify images
    cv::Mat rect_left, rect_right;
    rectifier.rectify(left_img, right_img, rect_left, rect_right);

    // Apply bilateral filter
    cv::Mat filtered_left = applyBilateralFilter(rect_left);
    cv::Mat filtered_right = applyBilateralFilter(rect_right);

    // Compute disparity (toggle between regular and WLS version)
    cv::Mat disparity, disparity_vis;
    // computeDisparity(filtered_left, filtered_right, disparity, disparity_vis, true); // With WLS
    computeDisparity(filtered_left, filtered_right, disparity, disparity_vis, false); // Without WLS

    // Create heatmap
    cv::Mat disparity_heatmap;
    cv::applyColorMap(disparity_vis, disparity_heatmap, cv::COLORMAP_JET);

    // Resize images for display
    cv::Mat resized_left = resizeImage(rect_left, SCALE_FACTOR);
    cv::Mat resized_right = resizeImage(rect_right, SCALE_FACTOR);
    cv::Mat resized_disp = resizeImage(disparity_vis, SCALE_FACTOR);
    cv::Mat resized_heatmap = resizeImage(disparity_heatmap, SCALE_FACTOR);

    // Convert grayscale disparity to 3-channel if needed
    if (resized_disp.channels() == 1) {
        cv::cvtColor(resized_disp, resized_disp, cv::COLOR_GRAY2BGR);
    }

    // Create the 2x2 grid
    cv::Mat top_row, bottom_row, grid_image;
    cv::hconcat(resized_left, resized_right, top_row);
    cv::hconcat(resized_disp, resized_heatmap, bottom_row);
    cv::vconcat(top_row, bottom_row, grid_image);

    // Save result
    std::string out_path = "./output/disparity.png";
    fs::create_directories(fs::path(out_path).parent_path());
    cv::imwrite(out_path, grid_image);
    std::cout << "✅ Saved disparity map to: " << out_path << std::endl;

    // Show result
    cv::imshow("Disparity Map", grid_image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
