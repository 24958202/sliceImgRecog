#include "authorinfo/author_info.h" 
#include <vector> 
#include <tuple> 
#include <queue>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <cmath> // For std::abs
#include <map>
#include <set>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <cmath>
#include <algorithm>//for std::max
#include <numeric>
#include <ranges> //std::views
#include <cstdint> 
#include <functional>
#include <cstdlib>
#include <unordered_set>
#include <iterator> 
#include <utility>        // For std::pair  
#include <execution> // for parallel execution policies (C++17)
#include <stdlib.h>
#include <jpeglib.h>
#include "cvLib_subclasses.h"
bool cvLib_subclasses::isValidImage(const std::string& img_path){
    if(img_path.empty()){
        return false;
    }
    std::vector<std::string> image_extensions{
        ".jpg",
        ".JPG",
        ".jpeg",
        ".JPEG"
        //".png",
        //".PNG"
    };
    for(const auto& item : image_extensions){
        if(img_path.find(item) != std::string::npos){
            return true;
        }
    }
    return false;
}
void cvLib_subclasses::compressJPEG(const std::string& inputFilename, const std::string& outputFilename, int quality) {
    if (inputFilename.empty() || outputFilename.empty()) {
        throw std::invalid_argument("Input and output filenames must not be empty.");
    }
    // Create a JPEG compression struct and error handler
    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    // Open the input file
    FILE* infile = fopen(inputFilename.c_str(), "rb");
    if (!infile) {
        jpeg_destroy_compress(&cinfo);
        throw std::runtime_error("Unable to open input file: " + inputFilename);
    }
    // Initialize JPEG decompression
    jpeg_decompress_struct dinfo;
    dinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&dinfo);
    jpeg_stdio_src(&dinfo, infile);
    if (jpeg_read_header(&dinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&dinfo);
        fclose(infile);
        throw std::runtime_error("Failed to read JPEG header from " + inputFilename);
    }
    jpeg_start_decompress(&dinfo);
    // Get image properties
    int width = dinfo.output_width;
    int height = dinfo.output_height;
    int numChannels = dinfo.num_components;
    // Allocate memory for the image data
    unsigned char* buffer = new unsigned char[width * height * numChannels];
    while (dinfo.output_scanline < height) {
        unsigned char* row_pointer = buffer + dinfo.output_scanline * width * numChannels;
        if (jpeg_read_scanlines(&dinfo, &row_pointer, 1) != 1) {
            delete[] buffer;
            jpeg_destroy_decompress(&dinfo);
            fclose(infile);
            throw std::runtime_error("Failed to read scanlines from " + inputFilename);
        }
    }
    // Finish decompression and close the input file
    jpeg_finish_decompress(&dinfo);
    fclose(infile);
    jpeg_destroy_decompress(&dinfo);
    // Set up compression
    FILE* outfile = fopen(outputFilename.c_str(), "wb");
    if (!outfile) {
        delete[] buffer;
        throw std::runtime_error("Unable to open output file: " + outputFilename);
    }
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = numChannels;
    // Comprehensive color space handling
    switch (numChannels) {
        case 1:
            cinfo.in_color_space = JCS_GRAYSCALE;
            break;
        case 3:
            cinfo.in_color_space = JCS_RGB; // Assuming RGB
            break;
        case 4:
            cinfo.in_color_space = JCS_CMYK; // Assuming CMYK, if supported
            break;
        default:
            delete[] buffer;
            fclose(outfile);
            throw std::invalid_argument("Unsupported number of channels: " + std::to_string(numChannels));
    }
    jpeg_set_defaults(&cinfo);
    // Check quality value
    if (quality < 0 || quality > 100) {
        delete[] buffer;
        fclose(outfile);
        throw std::invalid_argument("Quality must be between 0 and 100.");
    }
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_stdio_dest(&cinfo, outfile);
    // Start compression
    jpeg_start_compress(&cinfo, TRUE);
    while (cinfo.next_scanline < cinfo.image_height) {
        unsigned char* row_pointer = buffer + cinfo.next_scanline * width * numChannels;
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }
    // Finish compression and clean up
    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
    delete[] buffer;
    std::cout << "JPEG image compressed and saved to " << outputFilename << std::endl;
}
void cvLib_subclasses::preprocessImg(const std::string& img_path, const unsigned int ReSIZE_IMG_WIDTH, const unsigned int ReSIZE_IMG_HEIGHT, std::vector<cv::Mat>& outImg) {  
    if (img_path.empty()) {  
        std::cerr << "Error: img_path is empty." << std::endl;  
        return;  
    }  
	if (ReSIZE_IMG_WIDTH == 0 || ReSIZE_IMG_HEIGHT == 0) {  
        std::cerr << "Error: Invalid resize dimensions." << std::endl;  
        return;  
    }  
    try {  
        outImg.clear();  
        // Load the image  
        cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);  
        if (img.empty()) {  
            std::cerr << "Error: Image not loaded correctly from path: " << img_path << std::endl;  
            return; // Handle the error appropriately  
        }  
        // Resize the image  
        cv::Mat resizeImg;  
        cv::resize(img, resizeImg, cv::Size(ReSIZE_IMG_WIDTH, ReSIZE_IMG_HEIGHT));  
        // Check if resized image is smaller than slice size  
        if (resizeImg.cols < 64 || resizeImg.rows < 64) {  
            std::cerr << "Error: Resized image is smaller than slice size (64x64)." << std::endl;  
            return;  
        }  
        // Convert to grayscale  
        cv::Mat img_gray;  
        cv::cvtColor(resizeImg, img_gray, cv::COLOR_BGR2GRAY);  
        // Apply Gaussian Blur  
        cv::Mat blurredImg;  
        cv::GaussianBlur(img_gray, blurredImg, cv::Size(5, 5), 0);  
        // Define the size of slices  
        const int sliceWidth = 64;  
        const int sliceHeight = 64;  
        // Iterate and slice the image into 64x64 pieces  
        for (int y = 0; y <= blurredImg.rows - sliceHeight; y += sliceHeight) {  
            for (int x = 0; x <= blurredImg.cols - sliceWidth; x += sliceWidth) {  
                cv::Rect sliceArea(x, y, sliceWidth, sliceHeight);  
                // Ensure the slice area is valid  
                if (sliceArea.x + sliceArea.width <= blurredImg.cols && sliceArea.y + sliceArea.height <= blurredImg.rows) {  
                    cv::Mat slice = blurredImg(sliceArea); // Create a slice  
                    outImg.push_back(slice); // Store slice in output vector  
                }  
            }  
        }  
    } catch (const cv::Exception& ex) {  
        std::cerr << "OpenCV Error: " << ex.what() << std::endl;  
    } catch (const std::exception& ex) {  
        std::cerr << "Standard Error: " << ex.what() << std::endl;  
    } catch (...) {  
        std::cerr << "cvLib::subclasses::preprocessImg Unknown errors" << std::endl;  
    }  
}
void cvLib_subclasses::preprocessImg_gray(cv::Mat& img, const unsigned int ReSIZE_IMG_WIDTH, const unsigned int ReSIZE_IMG_HEIGHT) {
    if (img.empty()) {
        std::cerr << "preprocessImg_gray Error: img is empty." << std::endl;
        return; // No need to proceed if image is empty
    }
    try {
        // Resize the image
        cv::Mat resizedImg; // Renaming for clarity
        cv::resize(img, resizedImg, cv::Size(ReSIZE_IMG_WIDTH, ReSIZE_IMG_HEIGHT));
        // Convert to grayscale
        cv::Mat img_gray;
        cv::cvtColor(resizedImg, img_gray, cv::COLOR_BGR2GRAY);
        // Apply Gaussian Blur
        cv::GaussianBlur(img_gray, img, cv::Size(5, 5), 0); // Now applying to processed grayscale image
    } 
    catch (const cv::Exception& ex) {
        std::cerr << "OpenCV Error: " << ex.what() << std::endl;
    } 
    catch (const std::exception& ex) {
        std::cerr << "Standard Error: " << ex.what() << std::endl;
    } 
    catch (...) {
        std::cerr << "cvLib::subclasses::preprocessImg Unknown errors" << std::endl;
    }
}
std::vector<cv::KeyPoint> cvLib_subclasses::extractSIFTFeatures(const cv::Mat& img, cv::Mat& descriptors, const unsigned int MAX_FEATURES){
    std::vector<cv::KeyPoint> keypoints_input;
    if(img.empty()){
        return keypoints_input;
    }
    try{
        cv::Ptr<cv::SIFT> detector = cv::SIFT::create(MAX_FEATURES);
        detector->detectAndCompute(img, cv::noArray(), keypoints_input, descriptors);
    }
    catch(std::exception& ex){
        std::cerr << ex.what() << std::endl;
    }
    catch(...){
        std::cerr << "cvLib::subclasses::extractORBFeatures Unknown errors" << std::endl;
    }
    return keypoints_input;
}
void cvLib_subclasses::saveModel_keypoint(const std::unordered_map<std::string, std::vector<cv::Mat>>& featureMap, const std::string& filename) {  
    if (filename.empty()) {  
        return;  
    }  
    std::ofstream ofs(filename, std::ios::binary);  
    if (!ofs.is_open()) {  
        throw std::runtime_error("Unable to open file for writing.");  
    }  
    try {  
        size_t mapSize = featureMap.size();  
        ofs.write(reinterpret_cast<const char*>(&mapSize), sizeof(mapSize));  
        for (const auto& [className, images] : featureMap) {  
            size_t keySize = className.size();  
            ofs.write(reinterpret_cast<const char*>(&keySize), sizeof(keySize));  
            ofs.write(className.data(), keySize);  
            size_t imageCount = images.size();  
            ofs.write(reinterpret_cast<const char*>(&imageCount), sizeof(imageCount));  
            for (const auto& img : images) {  
                if (img.data) { // Check if image data is valid  
                    // Write image dimensions  
                    int rows = img.rows;  
                    int cols = img.cols;  
                    int type = img.type();  
                    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(rows));  
                    ofs.write(reinterpret_cast<const char*>(&cols), sizeof(cols));  
                    ofs.write(reinterpret_cast<const char*>(&type), sizeof(type));  
                    // Write image data  
                    ofs.write(reinterpret_cast<const char*>(img.data), img.total() * img.elemSize());  
                } else {  
                    std::cerr << "Invalid image data encountered for class: " << className << std::endl;  
                }  
            }  
        }  
    } catch (const std::exception& e) {  
        std::cerr << "Error writing to file: " << e.what() << std::endl;  
        throw; // Rethrow the exception after logging  
    }  
	ofs.close();
}
cv::Mat cvLib_subclasses::computeDescriptors(std::vector<cv::KeyPoint>& keypoints, const unsigned int MAX_FEATURES, const unsigned int ReSIZE_IMG_WIDTH, const unsigned int ReSIZE_IMG_HEIGHT) {
    // Check for empty keypoints
    if (keypoints.empty()) {
        std::cerr << "No keypoints provided." << std::endl;
        return cv::Mat(); // Handle the error appropriately
    }
    cv::Mat test_descriptors;
    try {
        cv::Ptr<cv::ORB> orb = cv::ORB::create(MAX_FEATURES);
        // Validate dummy image size
        if (ReSIZE_IMG_WIDTH <= 0 || ReSIZE_IMG_HEIGHT <= 0) {
            std::cerr << "Invalid dimensions for dummy_image: " 
                      << ReSIZE_IMG_WIDTH << "x" << ReSIZE_IMG_HEIGHT << std::endl;
            return cv::Mat();
        }
        // Create a dummy Mat to store the keypoints
        cv::Mat dummy_image = cv::Mat::zeros(cv::Size(ReSIZE_IMG_WIDTH, ReSIZE_IMG_HEIGHT), CV_8UC1);
        // Check validity of keypoints against dummy image
        for (const auto& kp : keypoints) {
            if (kp.pt.x < 0 || kp.pt.x >= dummy_image.cols || kp.pt.y < 0 || kp.pt.y >= dummy_image.rows) {
                std::cerr << "Keypoint out of bounds: " << kp.pt << std::endl;
                return cv::Mat(); // or handle this case more appropriately
            }
        }
        // Set the dummy image to a default value (e.g., 255)
        dummy_image.setTo(cv::Scalar(255));
        // Compute the descriptors
        orb->compute(dummy_image, keypoints, test_descriptors);
        if (test_descriptors.empty()) {
            std::cerr << "Failed to compute descriptors." << std::endl;
        }
    }
    catch (cv::Exception& ex) {
        std::cerr << "OpenCV Exception: " << ex.what() << std::endl;
    }
    catch (std::exception& ex) {
        std::cerr << "Standard Exception: " << ex.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown error in cvLib::subclasses::computeDescriptors" << std::endl;
    }
    return test_descriptors;
}
bool cvLib_subclasses::img1_img2_are_matched(const std::string& img1, const std::string& img2, const unsigned int DE_THRESHOLD, const unsigned int MAX_FEATURES, const unsigned int ReSIZE_IMG_WIDTH, const unsigned int ReSIZE_IMG_HEIGHT, const float RATIO_THRESH){
    if (img1.empty() || img2.empty()) {  
        std::cerr << "Image paths are empty." << std::endl;  
        return false;  
    }  
     // Load the image
    try{
        cv::Mat img_input1 = cv::imread(img1, cv::IMREAD_COLOR);
        if (img_input1.empty()) {
            std::cerr << "Error: img_input1 not loaded correctly from path: " << img1 << std::endl;
            return false; // Handle the error appropriately
        }
        cv::Mat img_input2 = cv::imread(img2, cv::IMREAD_COLOR);
        if (img_input2.empty()) {
            std::cerr << "Error: img_input2 not loaded correctly from path: " << img2 << std::endl;
            return false; // Handle the error appropriately
        }
        preprocessImg_gray(img_input1,ReSIZE_IMG_WIDTH,ReSIZE_IMG_HEIGHT);
        preprocessImg_gray(img_input2,ReSIZE_IMG_WIDTH,ReSIZE_IMG_HEIGHT);
        if (img_input1.empty() || img_input2.empty()) {  
            std::cerr << "Failed to read one or both images." << std::endl;  
            return false;  
        }  
        // Use ORB for keypoint detection and description
        std::vector<cv::KeyPoint> keypoints1, keypoints2;  
        cv::Mat descriptors1, descriptors2;  
        keypoints1 = extractSIFTFeatures(img_input1,descriptors1, MAX_FEATURES);
        keypoints2 = extractSIFTFeatures(img_input2,descriptors2, MAX_FEATURES);
        if (descriptors1.empty() || descriptors2.empty()) {
            std::cerr << "cvLib_subclasses::img1_img2_are_matched Existing m_img1 or m_img2 has no descriptors." << std::endl;
            return false;
        }
        /*
         *Start comparing
        */
        cv::BFMatcher matcher(cv::NORM_L2);
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);
        std::vector<cv::DMatch> goodMatches;
        for (const auto& match : knnMatches) {
            if (match.size() > 1 && match[0].distance < RATIO_THRESH * match[1].distance) {
                goodMatches.push_back(match[0]);
            }
        }
        if (goodMatches.size() > DE_THRESHOLD) {
            std::cout << "goodMatches.size() : " << goodMatches.size() << std::endl;
            return true;
        }
        std::cout << "Did not match, goodMatches.size() : " << goodMatches.size() << std::endl;
        return false;
    }
    catch (cv::Exception& ex) {
        std::cerr << "OpenCV Exception: " << ex.what() << std::endl;
    }
    catch (std::exception& ex) {
        std::cerr << "Standard Exception: " << ex.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown error in cvLib::subclasses::img1_img2_are_matched" << std::endl;
    }
}
bool cvLib_subclasses::img1_img2_are_matched_cvMat(const cv::Mat& img1, const cv::Mat& img2, const unsigned int DE_THRESHOLD, 
const unsigned int MAX_FEATURES, const unsigned int ReSIZE_IMG_WIDTH, const unsigned int ReSIZE_IMG_HEIGHT, const float RATIO_THRESH){
    if (img1.empty() || img2.empty()) {  
        std::cerr << "Images input are empty." << std::endl;  
        return false;  
    }  
	cv::Mat img1_resized;
	cv::Mat img2_resized;
	cv::resize(img1, img1_resized, cv::Size(ReSIZE_IMG_WIDTH,ReSIZE_IMG_HEIGHT));
	cv::resize(img2, img2_resized, cv::Size(ReSIZE_IMG_WIDTH,ReSIZE_IMG_HEIGHT));
    // Use ORB for keypoint detection and description
    std::vector<cv::KeyPoint> keypoints1, keypoints2;  
    cv::Mat descriptors1, descriptors2;  
    keypoints1 = extractSIFTFeatures(img1_resized,descriptors1, MAX_FEATURES);
    keypoints2 = extractSIFTFeatures(img2_resized,descriptors2, MAX_FEATURES);
    if (descriptors1.empty() || descriptors2.empty()) {
        //std::cerr << "img1_img2_are_matched_cvMat Existing m_img1 or m_img2 has no descriptors." << std::endl;
        return false;
    }
    /*
     *Start comparing
    */
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);
    std::vector<cv::DMatch> goodMatches;
    for (const auto& match : knnMatches) {
        if (match.size() > 1 && match[0].distance < RATIO_THRESH * match[1].distance) {
            goodMatches.push_back(match[0]);
        }
    }
    if (goodMatches.size() > DE_THRESHOLD) {
        return true;
    }
    return false;  
}
void cvLib_subclasses::sortAndFilterSlices(std::vector<cv::Mat>& outImg, const double similarityThreshold, const double emptyThreshold) {  
    if (outImg.empty()) {  
        return;  
    }  
    // Helper function to check if a slice is empty (background)  
    auto isEmptySlice = [&emptyThreshold](const cv::Mat&img) -> bool {  
        if (img.empty()) return true; // Check if the image is empty  
        cv::Mat gray;  
        if (img.channels() == 3) {  
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);  
        } else {  
            gray = img; // No need to clone if already grayscale  
        }  
        double meanValue = cv::mean(gray)[0];  
        return meanValue < emptyThreshold; // Background slice if intensity is below threshold  
    };  
    // Helper function to compute similarity metric (Histogram Comparison)  
    auto computeSimilarity = [](const cv::Mat& img1, const cv::Mat& img2) -> double {  
        if (img1.empty() || img2.empty()) return 0.0; // Check if images are valid  
        cv::Mat gray1, gray2;  
        if (img1.channels() == 3) {  
            cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);  
        } else {  
            gray1 = img1; // No need to clone if already grayscale  
        }  
        if (img2.channels() == 3) {  
            cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);  
        } else {  
            gray2 = img2; // No need to clone if already grayscale  
        }  
        int histSize = 256;  
        float range[] = { 0, 256 };  
        const float* histRange = { range };  
        cv::Mat hist1, hist2;  
        cv::calcHist(&gray1, 1, 0, cv::Mat(), hist1, 1, &histSize, &histRange);  
        cv::calcHist(&gray2, 1, 0, cv::Mat(), hist2, 1, &histSize, &histRange);  
        cv::normalize(hist1, hist1, 0, 1, cv::NORM_MINMAX);  
        cv::normalize(hist2, hist2, 0, 1, cv::NORM_MINMAX);  
        return cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL);  
    };  
    // Step 1: Filter out empty slices  
    outImg.erase(std::remove_if(outImg.begin(), outImg.end(), isEmptySlice), outImg.end());  
    // Step 2: Sort the remaining slices based on similarity  
    if (!outImg.empty()) {  
        const cv::Mat& referenceSlice = outImg[0]; // Use the first slice as the reference  
        std::sort(outImg.begin(), outImg.end(), [&computeSimilarity, &referenceSlice](const cv::Mat&a,const cv::Mat&b) {  
            return computeSimilarity(referenceSlice, a) > computeSimilarity(referenceSlice, b);  
        });  
    }  
}  
void cvLib_subclasses::sortByMeanIntensity(std::vector<cv::Mat>& outImg, const double emptyThreshold) {  
    if (outImg.empty()) {  
        return;  
    }  
    // Helper function to calculate the mean intensity of a slice  
    auto computeMeanIntensity = [](const cv::Mat& img) -> double {  
        if (img.empty()) return 0.0; // Check if the image is valid  
        cv::Mat gray;  
        if (img.channels() == 3) {  
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);  
        } else {  
            gray = img; // No need to clone if already grayscale  
        }  
        return cv::mean(gray)[0];  
    };  
    // Helper function to check if a slice is "empty" (background slice)  
    auto isEmptySlice = [&computeMeanIntensity, &emptyThreshold](const cv::Mat&img) -> bool {  
        double meanIntensity = computeMeanIntensity(img);  
        return meanIntensity < emptyThreshold;  
    };  
    // Step 1: Filter out empty slices  
    outImg.erase(std::remove_if(outImg.begin(), outImg.end(), isEmptySlice), outImg.end());  
    // Step 2: Sort remaining slices in descending order of mean intensity  
    std::sort(outImg.begin(), outImg.end(), [&computeMeanIntensity](const cv::Mat&a,const cv::Mat&b) {  
        return computeMeanIntensity(a) > computeMeanIntensity(b);  
    });  
}