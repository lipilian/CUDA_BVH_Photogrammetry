#include <opencv2/imgcodecs.hpp>
#include <string>
#include <glob.h>
#include <memory>
#include <iostream>
#include <filesystem>
#include <vector>
#include <omp.h>
#include "LiuHongSKINMASK.h"
#include "LiuHongAIMASK.h"
#include "nlohmann/json.hpp"
#include <fstream>

std::vector<std::string> _loadImages(const std::string & path) {
    std::vector<std::string> imagePaths;
    for(const auto & entry: std::filesystem::directory_iterator(path)) {
        if(entry.is_regular_file() && entry.path().extension() == ".png") {
            imagePaths.push_back(entry.path().string());
        }
    }
    std::sort(imagePaths.begin(), imagePaths.end());
    return imagePaths;
}
// * define the main function to generate the prediction mask from the data **
int main(int argc, char * argv[]){
    //omp_set_dynamic(1);
    if(argc != 2){
        std::cout << "Usage: ./AI configPath" << std::endl;
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

    std::string inputPath = configData["inputImgPath"];
    std::string outputPath = configData["outputImgPath"];
    std::string skinMaskModelPath = configData["skinMaskModelPath"];
    std::string skinMaskDebugPath = configData["skinMaskDebugPath"];
    bool debug = configData["debug"];
    try{
        for(const auto & entry : std::filesystem::directory_iterator(outputPath)) {
            std::filesystem::remove_all(entry);
        }
        std::cout << "Removed all files in " << outputPath << std::endl;
    } catch(const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    try{
        for(const auto & entry : std::filesystem::directory_iterator(skinMaskDebugPath)) {
            std::filesystem::remove_all(entry);
        }
        std::cout << "Removed all files in " << skinMaskDebugPath << std::endl;
    } catch(const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    // * Load the Images to vector **
    std::vector<std::string> imagePaths = _loadImages(inputPath);
    std::cout << "Number of images: " << imagePaths.size() << std::endl;
    std::unique_ptr<SKINMASK> skinMask = std::make_unique<SKINMASK>(skinMaskModelPath);
    std::vector<cv::Mat> images(imagePaths.size()); // create a vector to store 8 bit image
    #pragma omp parallel for
    for(int i = 0; i < imagePaths.size(); i++) {
        images[i] = cv::imread(imagePaths[i], cv::IMREAD_COLOR);
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
        std::filesystem::path fs_path(imagePaths[i]);
        std::string filename = fs_path.filename().string();
        if(debug){
            std::ostringstream oss;
            oss << skinMaskDebugPath << "/" << filename;
                cv::imwrite(oss.str(), masks[i]);
        }
        images[i].setTo(cv::Scalar(0, 0, 0), masks[i] == 0);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "SkinMask prediction time for " << imagePaths.size() << " images: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds" << std::endl;
    // * perform AI prediction on the images ** 
    start = std::chrono::high_resolution_clock::now();
    std::string aiModelPath = configData["burnNonBurnModelPath"];
    std::unique_ptr<AIMASK> aiMask = std::make_unique<AIMASK>(aiModelPath);
    std::vector<cv::Mat> predictions(imagePaths.size());
    for(int i = 0; i < images.size(); i++){
        predictions[i] = aiMask->predict(images[i]);
    }
    #pragma omp parallel for
    for(int i = 0; i < images.size(); i++){
        std::filesystem::path fs_path(imagePaths[i]);
        std::string filename = fs_path.filename().string();
        std::ostringstream oss;
        oss << outputPath << "/" << filename;
        cv::imwrite(oss.str(), predictions[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "AI prediction time for " << imagePaths.size() << " images: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds" << std::endl;

    
    return 0;
}