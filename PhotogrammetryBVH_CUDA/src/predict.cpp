#include <opencv2/imgcodecs.hpp>
#include <string>
#include <glob.h>
#include <memory>
#include <iostream>
#include <filesystem>
#include <vector>
#include <omp.h>
#include "LiuHongSKINMASK.h" 
#include "LiuHongStereo.h"
#include "nlohmann/json.hpp"
#include <fstream>

// * helper function to help load images from a directory **
std::vector<std::string> _loadImages(const std::string & path) {
    std::vector<std::string> imagePaths;
    for(const auto & entry: std::filesystem::directory_iterator(path)) {
        if(entry.is_regular_file() && entry.path().extension() == ".tiff") {
            imagePaths.push_back(entry.path().string());
        }
    }
    std::sort(imagePaths.begin(), imagePaths.end());
    return imagePaths;
}
// * define the main function to prepare the data **
int main(int argc, char * argv[]){
    // * read the args and get json info from json file **
    if(argc != 2){
        std::cout << "Usage: ./prepData configPath" << std::endl;
        return 1;
    }
    auto start = std::chrono::high_resolution_clock::now();
    std::string configPath = argv[1];
    std::ifstream configFile(configPath);
    if(!configFile.is_open()){
        std::cerr << "Error: Cannot open file " << configPath << std::endl;
        std::exit(EXIT_FAILURE);
    }
    nlohmann::json configData;
    configFile >> configData;
    configFile.close();
    
    std::string rawPath = configData["inputImgPath"];
    std::string imgPath = configData["outputImgPath"];
    std::string maskPath = configData["outputMaskPath"];
    std::string snapPath = configData["firstViewPath"];
    std::string depthPath = configData["outputDepthPath"];
    std::string skinMaskModelPath = configData["skinMaskModelPath"];
    std::string debugFolder = configData["debugFolder"];
    bool debug = configData["debug"];
    // clean the output directory
    try{
        for (const auto& entry : std::filesystem::directory_iterator(imgPath)) {
            std::filesystem::remove_all(entry);
        }
        std::cout << "Removed all files in " << imgPath << std::endl;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    // clean the mask directory
    try{
        for (const auto& entry : std::filesystem::directory_iterator(maskPath)) {
            std::filesystem::remove_all(entry);
        }
        std::cout << "Removed all files in " << maskPath << std::endl;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    // * Load the Images to vector **
    std::vector<std::string> imagePaths = _loadImages(rawPath); // get the image path from the raw image folder path 
    std::cout << "Number of images: " << imagePaths.size() << std::endl;
    std::unique_ptr<SKINMASK> skinMask = std::make_unique<SKINMASK>(skinMaskModelPath); // create the skin mask object
    std::vector<cv::Mat> images(imagePaths.size()); // create a vector to store 8 bit image
    #pragma omp parallel for // * omp parallel to load raw image and normalize it to 8 bit ** 
    for(int i = 0; i < imagePaths.size(); i++) {
        cv::Mat tempImage = cv::imread(imagePaths[i], cv::IMREAD_UNCHANGED); 
        // TODO: Add pseudo color generation part
        cv::normalize(tempImage, images[i], 0, 255, cv::NORM_MINMAX, CV_8U);
        std::ostringstream oss;
        oss << imgPath << "/" << std::setw(5) << std::setfill('0') << i << ".jpg";
        std::string outputFilename = oss.str();
        cv::imwrite(outputFilename, images[i]);
    }
    // * Generate the skin mask for all inputs **
    std::vector<cv::Mat> masks(imagePaths.size());
    for(int i = 0; i < images.size(); i++){
        masks[i] = skinMask->predict(images[i]);
    }
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
    #pragma omp parallel for
    for(int i = 0; i < masks.size(); i++){
        cv::Mat temp;
        cv::erode(masks[i], temp, kernel, cv::Point(-1, -1), 1);
        cv::copyMakeBorder(temp, temp, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
        int h = temp.rows;
        int w = temp.cols;
        cv::Mat mask = cv::Mat::zeros(h+2, w+2, CV_8U);
        cv::Mat img_filled = temp.clone();
        cv::floodFill(img_filled, mask, cv::Point(0, 0), cv::Scalar(255));

        cv::bitwise_not(img_filled, img_filled);
        cv::bitwise_or(temp, img_filled, masks[i]);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(masks[i], masks[i], cv::MORPH_OPEN, kernel);
        cv::morphologyEx(masks[i], masks[i], cv::MORPH_CLOSE, kernel);    
        masks[i] = masks[i](cv::Rect(1, 1, w-2, h-2));
        std::ostringstream oss;
        oss << maskPath << "/" << std::setw(5) << std::setfill('0') << i << ".png";
        std::string outputFilename = oss.str();
        cv::imwrite(outputFilename, masks[i]);
    }
    std::cout << "Mask generated successfully" << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "skinMask for 30 images: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds" << std::endl;
    // ************** Start stereo reconstruction **************
    start = std::chrono::high_resolution_clock::now();
    std::string firstViewPath = configData["firstViewPath"];
    std::string leftImageName = configData["leftImageName"];
    std::string rightImageName = configData["rightImageName"];
    std::string firstStereoModelPath = configData["firstStereoModelPath"];
    std::string secondStereoModelPath = configData["secondStereoModelPath"];
    // Read the left and right images
    std::string firstImagePath = firstViewPath + "/" + leftImageName;
    std::string secondImagePath = firstViewPath + "/" + rightImageName;
    cv::Mat left = cv::imread(firstImagePath, cv::IMREAD_UNCHANGED);
    cv::Mat right = cv::imread(secondImagePath, cv::IMREAD_UNCHANGED);
    // Normalize the images to 8 bit
    cv::normalize(left, left, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(right, right, 0, 255, cv::NORM_MINMAX, CV_8U);
    left.convertTo(left, CV_8U);
    right.convertTo(right, CV_8U);
    // Convert the images to RGB to prepare depth generation
    cv::cvtColor(left, left, cv::COLOR_BGR2RGB);
    cv::cvtColor(right, right, cv::COLOR_BGR2RGB);
    // Get the stereo camera parameters from the json file, M1, d1, M2, d2, R1, T1, R2, T2
    auto M1_json = configData["BGDN_CameraMatrix"];
    std::vector<double> M1_data = M1_json.get<std::vector<double>>();
    cv::Mat M1 = cv::Mat(3, 3, CV_64F, M1_data.data()).clone();

    auto d1_json = configData["BGDN_distortionCoefficients"];
    std::vector<double> d1_data = d1_json.get<std::vector<double>>();
    cv::Mat d1 = cv::Mat(1, 5, CV_64F, d1_data.data()).clone();

    auto M2_json = configData["BGON_CameraMatrix"];
    std::vector<double> M2_data = M2_json.get<std::vector<double>>();
    cv::Mat M2 = cv::Mat(3, 3, CV_64F, M2_data.data()).clone();

    auto d2_json = configData["BGON_distortionCoefficients"];
    std::vector<double> d2_data = d2_json.get<std::vector<double>>();
    cv::Mat d2 = cv::Mat(1, 5, CV_64F, d2_data.data()).clone();

    auto R1_json = configData["BGDN_R"];
    std::vector<double> R1_data = R1_json.get<std::vector<double>>();
    cv::Mat R1 = cv::Mat(3, 3, CV_64F, R1_data.data()).clone();

    auto T1_json = configData["BGDN_T"];
    std::vector<double> T1_data = T1_json.get<std::vector<double>>();
    cv::Mat T1 = cv::Mat(3, 1, CV_64F, T1_data.data()).clone();

    auto R2_json = configData["BGON_R"];
    std::vector<double> R2_data = R2_json.get<std::vector<double>>();
    cv::Mat R2 = cv::Mat(3, 3, CV_64F, R2_data.data()).clone();

    auto T2_json = configData["BGON_T"];
    std::vector<double> T2_data = T2_json.get<std::vector<double>>();
    cv::Mat T2 = cv::Mat(3, 1, CV_64F, T2_data.data()).clone();

    // Stereo rectification
    cv::Size imageSize = left.size();
    cv::Mat R1_out, R2_out, P1, P2, Q;
    cv::Mat R_rel = R2 * R1.t();
    cv::Mat T_rel = T2 - R_rel * T1;
    cv::stereoRectify(M1, d1, M2, d2, imageSize, R_rel, T_rel, R1_out, R2_out, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0, imageSize);

    cv::Mat map1x, map1y, map2x, map2y;
    cv::initUndistortRectifyMap(M1, d1, R1_out, P1, imageSize, CV_32F, map1x, map1y);
    cv::initUndistortRectifyMap(M2, d2, R2_out, P2, imageSize, CV_32F, map2x, map2y);
    cv::Mat map1x_, map1y_; // reverse mapping
    cv::initInverseRectificationMap(M1, d1, R1_out, P1, imageSize, CV_32F, map1x_, map1y_);

    cv::Mat leftRectified, rightRectified;
    cv::remap(left, leftRectified, map1x, map1y, cv::INTER_LINEAR);
    cv::remap(right, rightRectified, map2x, map2y, cv::INTER_LINEAR);

    if(debug){
        cv::Mat combined;
        cv::hconcat(leftRectified, rightRectified, combined);
        for (int y = 0; y < combined.rows; y += 100) {
            cv::line(combined, cv::Point(0, y), cv::Point(combined.cols, y), cv::Scalar(0, 255, 0), 1);
        }
        cv::cvtColor(combined, combined, cv::COLOR_RGB2BGR);
        std::string combinedImagePath = debugFolder + "/stereo_debug.jpg";
        cv::imwrite(combinedImagePath, combined);
        std::cout << "Combined image saved to " << combinedImagePath << std::endl;
    }
    // CRESTEREO to get the disparity map
    STEREO stereo = STEREO(firstStereoModelPath, secondStereoModelPath);
    cv::Mat disparity = stereo.predict(leftRectified, rightRectified, debug, debugFolder);
    cv::Mat threeDpoints;
    cv::reprojectImageTo3D(disparity, threeDpoints, Q, true);

    cv::Mat distanceMap(threeDpoints.size(), CV_32FC1);
    // ! The distance map here is distance from camera center to 3D point.
    for(int i = 0; i < threeDpoints.rows; i++){
        for(int j = 0; j < threeDpoints.cols; j++){
            cv::Vec3f point = threeDpoints.at<cv::Vec3f>(i, j);
            distanceMap.at<float>(i, j) = cv::norm(point);
        }
    }
    // apply distortion back to the depth map
    cv::remap(distanceMap, distanceMap, map1x_, map1y_, cv::INTER_LINEAR);
    cv::imwrite(depthPath + "/distanceMap.tiff", distanceMap);
    
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Stereo reconstruction for first of view: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds" << std::endl;
}