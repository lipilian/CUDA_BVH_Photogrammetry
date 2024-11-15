#include "LiuHongSKINMASK.h"


SKINMASK::SKINMASK(const std::string& path): torchScriptPath(path) {
    // Constructor implementation
    std::cout << "torchScript path: " << torchScriptPath << std::endl;
    std::cout << "SKINMASK object created" << std::endl;
    try {
        model = std::make_unique<torch::jit::script::Module>(torch::jit::load(path));
        model->eval();  
        std::cout << "Model loaded successfully" << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
    }
    // Move model to CUDA device
    if (torch::cuda::is_available()) {
        model->to(torch::kCUDA);
        std::cout << "Model moved to CUDA device" << std::endl;
    } else {
        std::cout << "CUDA is not available, model will remain on CPU" << std::endl;
    }
}

cv::Mat SKINMASK::predict(const cv::Mat original_image) {
    if (original_image.empty()) {
        std::cerr << "Failed to load image: " << std::endl;
        return original_image;
    }
    cv::Mat image = original_image.clone();
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    int width = image.cols;
    int height = image.rows;

    cv::resize(image, image, cv::Size(512, 512), 0, 0, cv::INTER_NEAREST);
    cv::Scalar mean, stdDev;
    cv::meanStdDev(image, mean, stdDev);
    std::vector<cv::Mat> channels;
    image.convertTo(image, CV_32F);
    cv::split(image, channels);
    for (size_t i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean[i] + 1e-7) / (stdDev[i] + 1e-7);
    }
    cv::merge(channels, image);

    torch::Tensor image_tensor = torch::from_blob(image.data, {1, 512, 512, 3}, torch::kFloat32);
    image_tensor = image_tensor.permute({0, 3, 1, 2});
    image_tensor = image_tensor.to(torch::kCUDA);
    auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor output = model->forward({image_tensor}).toTensor().squeeze();
    auto end = std::chrono::high_resolution_clock::now();
    //std::cout << "Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    

    torch::Tensor predict = output[1].gt(0.5);
    predict = predict.cpu().detach();
    predict = predict.to(torch::kUInt8);
    cv::Mat result(predict.size(0), predict.size(1), CV_8UC1, predict.data_ptr());
    cv::resize(result, result, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    return result;
}

cv::Mat SKINMASK::predict(const std::string &path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << path << std::endl;
        return image;
    }
    cv::Mat rgb_image;
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    int width = image.cols;
    int height = image.rows;

    cv::resize(image, image, cv::Size(512, 512), 0, 0, cv::INTER_NEAREST);
    cv::Scalar mean, stdDev;
    cv::meanStdDev(image, mean, stdDev);
    std::vector<cv::Mat> channels;
    image.convertTo(image, CV_32F);
    cv::split(image, channels);
    for (size_t i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean[i] + 1e-7) / (stdDev[i] + 1e-7);
    }
    cv::merge(channels, image);

    torch::Tensor image_tensor = torch::from_blob(image.data, {1, 512, 512, 3}, torch::kFloat32);
    image_tensor = image_tensor.permute({0, 3, 1, 2});
    image_tensor = image_tensor.to(torch::kCUDA);
    auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor output = model->forward({image_tensor}).toTensor().squeeze();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;


    torch::Tensor predict = output[1].gt(0.5);
    predict = predict.cpu().detach();
    predict = predict.to(torch::kUInt8) * 255;
    cv::Mat result(predict.size(0), predict.size(1), CV_8UC1, predict.data_ptr());
    return result;
}

SKINMASK::~SKINMASK() {
    // Destructor implementation
    std::cout << "SKINMASK object destroyed" << std::endl;
}   