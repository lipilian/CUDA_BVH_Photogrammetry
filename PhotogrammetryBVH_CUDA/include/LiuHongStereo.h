# pragma once
#include <torch/script.h>
#include <torch/torch.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp> 
#include <iostream>
#include <vector>
#include <chrono>

class STEREO{
    private:
        std::string torchPath1;
        std::string torchPath2;
        std::unique_ptr<torch::jit::script::Module>  model1;
        std::unique_ptr<torch::jit::script::Module>  model2;
    public:
        STEREO(const std::string& modelPath1, const std::string& modelPath2);
        cv::Mat predict(cv::Mat left, cv::Mat right, bool debug, const std::string& debugFolder);
};