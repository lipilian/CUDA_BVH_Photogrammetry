#include "LiuHongAIMASK.h"

#define R_Mean 0.485f
#define G_Mean 0.456f
#define B_Mean 0.406f
#define R_Std 0.229f
#define G_Std 0.224f
#define B_Std 0.225f
#define IMG_SIZE 480

AIMASK::AIMASK(const std::string & path): torchScriptPath(path){
    // Constructor implementation
    std::cout << "torchScript path: " << torchScriptPath << std::endl;
    std::cout << "AIMASK object created" << std::endl;
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

AIMASK::~AIMASK() {
    // Destructor implementation
    std::cout << "AIMASK object destroyed" << std::endl;
}

cv::Mat AIMASK::predict(const cv::Mat & original_image) {
    if (original_image.empty()) {
        std::cerr << "Failed to load image: " << std::endl;
        return original_image;
    }
    cv::Mat image = original_image.clone();
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    int width = image.cols;
    int height = image.rows;
    // comment resize image to 480,480
    cv::resize(image, image, cv::Size(IMG_SIZE, IMG_SIZE), 0, 0, cv::INTER_NEAREST);
    // float image / 255
    image.convertTo(image, CV_32F, 1.0 / 255);
    // - mean / std
    cv::Scalar mean_values(R_Mean, G_Mean, B_Mean);
    cv::Scalar std_values(R_Std, G_Std, B_Std);
    cv::subtract(image, mean_values, image);
    cv::divide(image, std_values, image);
    // change to tensor
    torch::Tensor image_tensor = torch::from_blob(image.data, {1, IMG_SIZE, IMG_SIZE, 3}, torch::kFloat32);
    image_tensor = image_tensor.permute({0, 3, 1, 2});
    image_tensor = image_tensor.to(torch::kCUDA);
    torch::Tensor output = model->forward({image_tensor}).toTensor().squeeze();
    // heatmap output size 1,2,480,480
    output = torch::nn::functional::softmax(output, 0);
    torch::Tensor predict = output[1].cpu().detach().squeeze();
    predict = predict.gt(0.5).to(torch::kUInt8) * 255;

    cv::Mat result(predict.size(0), predict.size(1), CV_8UC1, predict.data_ptr());
    cv::resize(result, result, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);

    return result;
}