#include "LiuHongStereo.h"

STEREO::STEREO(const std::string & modelPath1, const std::string & modelPath2){
    std::cout << "torchScript path 1: " << modelPath1 << std::endl;
    std::cout << "torchScript path 2: " << modelPath2 << std::endl;
    std::cout << "STEREO object created" << std::endl;
    try {
        model1 = std::make_unique<torch::jit::script::Module>(torch::jit::load(modelPath1));
        model1->eval();
        std::cout << "Model 1 loaded successfully" << std::endl;

        model2 = std::make_unique<torch::jit::script::Module>(torch::jit::load(modelPath2));
        model2->eval();

        std::cout << "Model 2 loaded successfully" << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
    }
    // Move model to CUDA device 
    if(torch::cuda::is_available()){
        model1->to(torch::kCUDA);
        model2->to(torch::kCUDA);
        std::cout << "Model moved to CUDA device" << std::endl;
    } else {
        std::cout << "CUDA is not available" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

cv::Mat STEREO::predict(cv::Mat left, cv::Mat right, bool debug, const std::string & debugFolder){
    int width = left.cols;
    int height = left.rows;
    cv::Mat left_resized, right_resized, left_temp, right_temp;
    int new_width = width/2;
    int new_height = height/2;
    int temp_width = width/4;
    int temp_height = height/4;

    cv::resize(left, left_resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
    cv::resize(right, right_resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
    cv::resize(left, left_temp, cv::Size(temp_width, temp_height), 0, 0, cv::INTER_LINEAR);
    cv::resize(right, right_temp, cv::Size(temp_width, temp_height), 0, 0, cv::INTER_LINEAR);

    torch::Tensor input1 = torch::from_blob(left_temp.data, {temp_height, temp_width, 3}, torch::kByte);
    torch::Tensor input2 = torch::from_blob(right_temp.data, {temp_height, temp_width, 3}, torch::kByte);
    torch::Tensor input3 = torch::from_blob(left_resized.data, {new_height, new_width, 3}, torch::kByte);
    torch::Tensor input4 = torch::from_blob(right_resized.data, {new_height, new_width, 3}, torch::kByte);
    input1 = input1.to(torch::kFloat32);//.div(255.0);
    input2 = input2.to(torch::kFloat32);//.div(255.0);
    input3 = input3.to(torch::kFloat32);//.div(255.0);
    input4 = input4.to(torch::kFloat32);//.div(255.0);
    input1 = input1.unsqueeze(0);
    input2 = input2.unsqueeze(0);
    input3 = input3.unsqueeze(0);
    input4 = input4.unsqueeze(0);

    input1 = input1.permute({0, 3, 1, 2});
    input2 = input2.permute({0, 3, 1, 2});
    input3 = input3.permute({0, 3, 1, 2});
    input4 = input4.permute({0, 3, 1, 2});
    std::vector<torch::jit::IValue> inputs12;
    std::vector<torch::jit::IValue> inputs34;
    
    if (torch::cuda::is_available()) {
        inputs12.push_back(input1.to(torch::kCUDA));
        inputs12.push_back(input2.to(torch::kCUDA));
        std::cout << "Input12 moved to CUDA device" << std::endl;
    } else {
        std::cout << "CUDA is not available, input will remain on CPU" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    torch::Tensor midOutput = model1->forward(inputs12).toTensor().squeeze();
    midOutput = midOutput.unsqueeze(0);
    if (torch::cuda::is_available()) {
        inputs34.push_back(input3.to(torch::kCUDA));
        inputs34.push_back(input4.to(torch::kCUDA));
        inputs34.push_back(midOutput);
        std::cout << "Input34 moved to CUDA device" << std::endl;
    } else {
        std::cout << "CUDA is not available, input will remain on CPU" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    torch::Tensor output = model2->forward(inputs34).toTensor().cpu().detach().squeeze();
    torch::Tensor pred_disp = output[0];

    cv::Mat result = cv::Mat(pred_disp.size(0), pred_disp.size(1), CV_32F, pred_disp.data_ptr());
    result = result * 2;
    cv::resize(result, result, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

    double minVal, maxVal;
    cv::minMaxLoc(result, &minVal, &maxVal);
    std::cout << "Disparity : Min value: " << minVal << ", Max value: " << maxVal << std::endl;

    if(debug){
        cv::Mat disp_vis;
        cv::normalize(result, disp_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(disp_vis, disp_vis, cv::COLORMAP_INFERNO);
        std::string debugPath = debugFolder + "/disp_debug.jpg";
        cv::imwrite(debugPath, disp_vis);
        std::cout << "wrote debug image to " << debugPath << std::endl;

        cv::Size imageSize = left.size();
        cv::Mat xmap(imageSize, CV_32F);
        cv::Mat ymap(imageSize, CV_32F);
        for(int y = 0; y < height ; y++){
            for(int x = 0; x < width; x++){
                float d = result.at<float>(y, x);
                float new_x = x - d;
                new_x = std::max(0.0f, std::min(new_x, (float)(imageSize.width - 1)));
                xmap.at<float>(y, x) = new_x;
                ymap.at<float>(y, x) = y;
            }
        }
        cv::Mat aligned_right;
        cv::remap(right, aligned_right, xmap, ymap, cv::INTER_LINEAR);
        cv::imwrite(debugFolder + "/aligned_right.jpg", aligned_right);
        cv::imwrite(debugFolder + "/left.jpg", left);
    }


    return result;
}