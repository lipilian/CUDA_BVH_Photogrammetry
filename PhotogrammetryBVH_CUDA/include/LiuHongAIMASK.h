#pragma once
#include <torch/script.h>
#include <torch/torch.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <chrono>

class AIMASK{
    private:
        std::string torchScriptPath;
        std::unique_ptr<torch::jit::script::Module>  model;
    public:
        AIMASK(const std::string& path);
        cv::Mat predict(const cv::Mat & image);
        cv::Mat predict(const std::string & path);
        ~AIMASK();
};
