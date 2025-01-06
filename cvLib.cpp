/*
    c++20 lib for using opencv
    Dependencies:
    opencv,tesseract,sdl2,sdl_image,boost
*/
#include <opencv2/opencv.hpp>  
#include <opencv2/features2d.hpp> 
#include <opencv2/video.hpp> 
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
//#include <openssl/sha.h> // For SHA256
#include <iomanip>
#include "cvLib_subclasses.h"
#include "cvLib.h"
const unsigned int MAX_FEATURES = 1000;   // Max number of features to detect
const float RATIO_THRESH = 0.65f;          // Ratio threshold for matching
const unsigned int DE_THRESHOLD = 10;      // Min matches to consider an object as existing 10
const unsigned int ReSIZE_IMG_WIDTH = 448;
const unsigned int ReSIZE_IMG_HEIGHT = 448;
const unsigned int COUNT_NUM = 12;
const float LEARNING_RATE = 0.3f;
unsigned int faceCount = 0; 
//class ImageCounter {
//public:
//    // Increment the count for the given image
//    void incrementCount(const cv::Mat& img) {
//        std::string img_hash = computeHash(img); // Hash the image to use as key
//        auto it = data.find(img_hash);
//        if (it != data.end()) {
//            it->second++;  // Increment count if image matches
//        } else {
//            data[img_hash] = 1; // If not found, add the hash and initialize count to 1
//        }
//    }
//    // Check if the data is empty
//    bool is_empty() const {
//        return data.empty(); // Return true if data is empty
//    }
//    // Retrieve the counts of images as a vector
//    std::unordered_map<std::string, unsigned int> getData() const {
//        return data; // Return the data map
//    }
//private:
//    std::string computeHash(const cv::Mat& img) const {
//        // Resize or convert image if necessary to ensure consistent representation
//        cv::Mat img_resized;
//        cv::resize(img, img_resized, cv::Size(256, 256)); // Example of resizing
//        // Perform SHA-256 hashing on the raw data of the image
//        unsigned char hash[SHA256_DIGEST_LENGTH];
//        SHA256(img_resized.data, img_resized.total() * img_resized.elemSize(), hash);
//        // Convert the hash to a string representation
//        std::ostringstream oss;
//        for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
//            oss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
//        }
//        return oss.str();
//    }
//private:
//    std::unordered_map<std::string, unsigned int> data; // Hash-map to hold image hashes and their counts
//};
// A utility function to generate a unique hash for a cv::Mat image
std::string cvLib::matToString(const cv::Mat& img) {
    std::vector<uchar> buf;
    cv::imencode(".jpg", img, buf);  // Encode to jpg format
    return std::string(buf.begin(), buf.end());
}
// Function to remove duplicates from vector<cv::Mat>
//std::vector<cv::Mat> cvLib::removeDuplicates(const std::vector<cv::Mat>& images) {
//    std::unordered_set<std::string> unique_set;
//    std::vector<cv::Mat> unique_images;
//    for (const auto& img : images) {
//        std::string img_hash = matToString(img); // Get the hash of the image
//        if (unique_set.insert(img_hash).second) { // Try to insert the hash
//            unique_images.push_back(img); // If successful, it's unique
//        }
//    }
//    return unique_images; // Return the vector of unique images
//}
void cvLib::img_compress(const std::string& input_folder,int quality){
    if(input_folder.empty()){
        return;
    }
    if (!std::filesystem::exists(input_folder)) {
        std::cerr << "The folder does not exist" << std::endl;
        return;
    }
    cvLib_subclasses cvlib_sub;
    try {
        for (const auto& entryMainFolder : std::filesystem::directory_iterator(input_folder)) {  
            if (entryMainFolder.is_directory()) {  
                std::string sub_folder_path = entryMainFolder.path().string();
                std::cout << "Start working on folder: " << sub_folder_path << std::endl;
                for (const auto& entrySubFolder : std::filesystem::directory_iterator(sub_folder_path)) {  
                    if (entrySubFolder.is_regular_file()) {   
                        std::string imgFilePath = entrySubFolder.path().string(); 
                        if(imgFilePath.empty()){
                            continue;
                        }
                        if(cvlib_sub.isValidImage(imgFilePath)){
                            try{
                                cvlib_sub.compressJPEG(imgFilePath,imgFilePath,quality);
                            }
                            catch(const std::exception& ex){
                                std::cerr << ex.what() << std::endl;
                            }
                            catch(...){
                                std::cerr << "cvlib_sub.compressJPEG(imgFilePath,imgFilePath,quality); Unknown error!" << std::endl;
                            }
                            std::cout << "Successfully compressed the image: " << imgFilePath << std::endl;
                        }
                    }
                }
            }
        }
    }
    catch(std::exception& ex){
        std::cerr << ex.what() << std::endl;
    }
    catch(...){
        std::cerr << "Unknown errors." << std::endl;
    }
    std::cout << "All jobs are done!" << std::endl;
}
void train_process(cvLib_subclasses& cvlib_sub, const std::vector<std::string> img_process, const std::string& sub_folder_name, std::vector<cv::Mat>& sub_folder_imgs, std::mutex& outputMutex, unsigned int thread_id){
	if(img_process.empty() || sub_folder_name.empty()){
        return;
    }
    for(unsigned int i = 0; i < img_process.size(); ++i){
        std::cout << "Thread: " << thread_id << " Reading: " << img_process[i] << " , please wait..." << std::endl;
        std::vector<cv::Mat> trained_img1;
        cvlib_sub.preprocessImg(img_process[i],ReSIZE_IMG_WIDTH,ReSIZE_IMG_HEIGHT,trained_img1);
        if(!trained_img1.empty()){
            std::unordered_map<unsigned int,unsigned int> train_img_index_count;
            /*
             * initialize train_img_index_count 
             */
            for(unsigned int ii = 0; ii < trained_img1.size(); ++ii){
                train_img_index_count[ii] = 0;
            }
            for(unsigned int ii = 0; ii < trained_img1.size(); ++ii){
                /*
                 *Preprocess image 
                 */
                for(unsigned int j = 0; j < img_process.size(); ++j){
                    if(j!=i){//if(j!=i && already_checked[img_process[j]] == 0)
                        std::vector<cv::Mat> trained_others;
                        cvlib_sub.preprocessImg(img_process[j],ReSIZE_IMG_WIDTH,ReSIZE_IMG_HEIGHT,trained_others);
                        if(!trained_others.empty()){
                            for(unsigned int k = 0; k < trained_others.size(); ++k){
                                if(cvlib_sub.img1_img2_are_matched_cvMat(trained_img1[ii],trained_others[k],DE_THRESHOLD,MAX_FEATURES,ReSIZE_IMG_WIDTH,ReSIZE_IMG_HEIGHT,RATIO_THRESH)){
                                    train_img_index_count[ii]++;
                                    std::cout << "Matched found for index: " << ii << std::endl;
                                }
                            }
                        }
                        else{
                            std::cerr << "train_img_occurrences: trained_others is empty!" << std::endl;
                        }
                    }
                }
            }
            if(!train_img_index_count.empty()){
                std::vector<std::pair<unsigned int, unsigned int>> sortedFreq(train_img_index_count.begin(), train_img_index_count.end());
                // Sort the vector of pairs
                std::sort(sortedFreq.begin(), sortedFreq.end(), [](const auto& a, const auto& b) {
                    return a.second > b.second;
                });
                if(sortedFreq.empty()){
                    std::cerr << "Sorted result is empty!" << std::endl;
                    continue;
                }
                /*
                 * Evaluate images, remove those can not be recognized
                 * */
                if(!sortedFreq.empty()){
                     unsigned int check_num = 0;
                     std::cout << "Evaluating image collections..." << std::endl;
                     for(auto it = sortedFreq.begin(); it != sortedFreq.end();){
                        cv::Mat eval_descriptors;
                        std::vector<cv::KeyPoint> trained_key = cvlib_sub.extractSIFTFeatures(trained_img1[it->first],eval_descriptors,MAX_FEATURES);
                        if (eval_descriptors.empty()) {
                            std::cerr << "Removing bad image at index: " << std::to_string(check_num) << std::endl;
                            it = sortedFreq.erase(it);
                        }
                        else{
                            ++it;
                        }
                        check_num++;
                     }
                }
                else{
                    std::cerr << "cvLib::train_img_occurrences sub_folder_imgs is empty!" << std::endl;
                    continue;
                }
                if(sortedFreq.empty()){
                    std::cerr << "Sorted result is empty! Nothing left after evaluation." << std::endl;
                    continue;
                }
                unsigned int fetched_number_count = 0;
                unsigned int total_num = sortedFreq.size();
                unsigned int stop_fetching_number = static_cast<unsigned int>(LEARNING_RATE*total_num);
                for (const auto& trainItem : sortedFreq) {
                    if(trainItem.first < trained_img1.size()){
                        std::lock_guard<std::mutex> lock(outputMutex);
                        sub_folder_imgs.push_back(trained_img1[trainItem.first]); 
                        fetched_number_count++;
                        if(fetched_number_count > stop_fetching_number){
                            break;
                        }
                    }
                }
                
            }
            else{
                std::cerr << "cvLib::train_img_occurrences trained_img1_counter is empty!" << std::endl;
            }
        }//!trained_img1.empty()
		else{
			std::cout << "preprocessImg returned an empty value!" << std::endl;
		}
    }//for
}
void cvLib::train_img_occurrences(const std::string& images_folder_path, const std::string& output_model_path){
    if(images_folder_path.empty() || output_model_path.empty()){
        return;
    }
    cvLib_subclasses cvlib_sub;//img1_img2_are_matched_return
    std::unordered_map<std::string, std::vector<cv::Mat>> dataset_keypoint;
    try {  
        std::cout << "Start working..." << std::endl;
        const int numThreads = std::thread::hardware_concurrency();// Use the number of available cores
		std::cout << "Threads number: " << numThreads << std::endl;
        std::mutex outputMutex;
        for (const auto& entryMainFolder : std::filesystem::directory_iterator(images_folder_path)) {  
            if (entryMainFolder.is_directory()) { // Check if the entry is a directory  
                std::string sub_folder_name = entryMainFolder.path().filename().string();  
                std::string sub_folder_path = entryMainFolder.path().string(); 
				std::cout << "sub_folder_name: " << sub_folder_name << std::endl;
				std::cout << "sub_folder_path: " << sub_folder_path << std::endl;
                std::vector<std::string> sub_folder_file_list;
                std::vector<cv::Mat> sub_folder_imgs;
                std::cout << "Reading images in folder: " <<  sub_folder_path << std::endl;
                for (const auto& entrySubFolder : std::filesystem::directory_iterator(sub_folder_path)) {  
                    if (entrySubFolder.is_regular_file()) {  
                        std::string imgFilePath = entrySubFolder.path().string(); 
						if(imgFilePath.find("._") != std::string::npos){//file format from Mac
							continue;
						}
                        std::cout << "Adding image: " << imgFilePath << " to the list." << std::endl;
                        if(cvlib_sub.isValidImage(imgFilePath)){
                           sub_folder_file_list.push_back(imgFilePath);
                        }
                    }
                }
                /*
                 * Start processing sub_folder_file_list
                 */
                if(!sub_folder_file_list.empty()){
                    /*Only has one image*/
                    if(sub_folder_file_list.size() == 1){
                        std::vector<cv::Mat> trained_img;
                        cvlib_sub.preprocessImg(sub_folder_file_list[0],ReSIZE_IMG_WIDTH,ReSIZE_IMG_HEIGHT,trained_img);
                        if(!trained_img.empty()){
							cvlib_sub.sortByMeanIntensity(trained_img,10);//sortByMeanIntensity
                            sub_folder_imgs.insert(sub_folder_imgs.end(),trained_img.begin(),trained_img.end());
                        }
                        else{
                            std::cerr << "train_img_occurrences: sub_folder_file_list[0] is empty!" << std::endl;
                        }
                    }
                    else{ // more than one image-read with multiple-threads
                        int total_file_number = sub_folder_file_list.size();
                        std::vector<std::thread> threads;
						std::vector<cv::Mat> trained_final_result;
                        int linesPerThread = total_file_number / numThreads;
                        std::vector<std::string> thread_images_to_process;
                        for (int i = 0; i < numThreads; ++i) {
                            int startLine = i * linesPerThread;
                            int endLine = (i == numThreads - 1) ? total_file_number : (i + 1) * linesPerThread; 
							threads.emplace_back([&, startLine, endLine, i]() {  
								std::vector<cv::Mat> local_trained_result;  
								train_process(cvlib_sub, std::vector<std::string>(sub_folder_file_list.begin() + startLine, sub_folder_file_list.begin() + endLine), sub_folder_name, local_trained_result, outputMutex, i);  
								if(!local_trained_result.empty()){
									trained_final_result.insert(trained_final_result.end(),local_trained_result.begin(),local_trained_result.end()); 
								}
								else{
									std::cout << images_folder_path << ": local_trained_result is empty!" << std::endl;
								}
                            });  
                        }
                        for (auto& thread : threads) {
                            thread.join();
                        }
						// Merge results from all threads  
                        sub_folder_imgs.insert(sub_folder_imgs.end(), trained_final_result.begin(), trained_final_result.end());  

                    }//else{ // more than one image
                }
                dataset_keypoint[sub_folder_name] = sub_folder_imgs;
                std::cout << sub_folder_name << " training is done!" << std::endl;
            }
        }
    }
    catch (const std::filesystem::filesystem_error& e) {  
        std::cerr << "Filesystem error: " << e.what() << std::endl;  
        return; // Return an empty dataset in case of filesystem error  
    }  
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }
    if(!dataset_keypoint.empty()){
        cvlib_sub.saveModel_keypoint(dataset_keypoint,output_model_path);
        std::cout << "Successfully saved the model file to " << output_model_path << std::endl;
    }
    else{
        std::cerr << "cvLib::train_img_occurrences dataset_keypoint is empty!" << std::endl;
    }
}
void cvLib::loadModel_keypoint(std::unordered_map<std::string, std::vector<cv::Mat>>& featureMap, const std::string& filename) {  
    if (filename.empty()) {  
        return;  
    }  
    std::ifstream ifs(filename, std::ios::binary);  
    if (!ifs.is_open()) {  
        std::cerr << "Error: Unable to open file for reading." << std::endl;  
        return;  
    }  
    try {  
        size_t mapSize;  
        ifs.read(reinterpret_cast<char*>(&mapSize), sizeof(mapSize));  
        for (size_t i = 0; i < mapSize; ++i) {  
            size_t keySize;  
            ifs.read(reinterpret_cast<char*>(&keySize), sizeof(keySize));  
            std::string className(keySize, '\0');  
            ifs.read(&className[0], keySize);  
            size_t imageCount;  
            ifs.read(reinterpret_cast<char*>(&imageCount), sizeof(imageCount));  
            std::vector<cv::Mat> images(imageCount);  
            for (size_t j = 0; j < imageCount; ++j) {  
                // Read image dimensions  
                int rows, cols, type;  
                ifs.read(reinterpret_cast<char*>(&rows), sizeof(rows));  
                ifs.read(reinterpret_cast<char*>(&cols), sizeof(cols));  
                ifs.read(reinterpret_cast<char*>(&type), sizeof(type));  
                // Create an empty Mat to hold the image  
                cv::Mat img(rows, cols, type);  
                // Read image data  
                ifs.read(reinterpret_cast<char*>(img.data), img.total() * img.elemSize());  
                images[j] = img; // Store the image in the vector  
            }  
            featureMap[className] = images; // Store the vector of images in the map  
        }  
    } catch (const std::exception& e) {  
        std::cerr << "Error reading from file: " << e.what() << std::endl;  
    }  
    ifs.close();  
}
void cvLib::ini_trained_data(const std::string& model_path){
    if(model_path.empty()){
        std::cerr << "Model file path is empty!" << std::endl;
        return;
    }
    loadModel_keypoint(trained_dataset, model_path);
}
void cvLib::img_recognition(const std::vector<std::string>& input_images_path, std::unordered_map<std::string, return_img_info>& return_imgs, const float read_rate, const float bad_rate) {  
    // Validate input parameters and trained dataset  
    if (input_images_path.empty()) {  
        std::cerr << "Error: input_images_path is empty!" << std::endl;  
        return;  
    }  
    if (trained_dataset.empty()) {  
        std::cerr << "Error: trained_dataset is empty!" << std::endl;  
        return;  
    }  
    std::cout << "Starting image recognition on " << input_images_path.size() << " input images." << std::endl;  
    // Initialize utility class  
    cvLib_subclasses cvlib_sub;  
    // Loop through each test image path  
    for (const auto& test_item : input_images_path) {  
        return_img_info rii;  
        try {  
            auto t_count_start = std::chrono::high_resolution_clock::now(); // Start timing  
            std::vector<cv::Mat> test_img_slices;  
            cvlib_sub.preprocessImg(test_item, ReSIZE_IMG_WIDTH, ReSIZE_IMG_HEIGHT, test_img_slices);  
            if (test_img_slices.empty()) {  
                std::cerr << "Error: preprocessImg produced no slices for " << test_item << std::endl;  
                continue;  
            }  
            const double emptyThreshold = 10.0; // Below this intensity means it's background  
            cvlib_sub.sortByMeanIntensity(test_img_slices, emptyThreshold);  
            /*  
             * Start recognizing   
             */  
            unsigned int first_num = static_cast<unsigned int>(read_rate * test_img_slices.size());  
            unsigned int bad_num = static_cast<unsigned int>(bad_rate * test_img_slices.size());  
            unsigned int slice_count = 0;  
            std::unordered_map<std::string, unsigned int> test_slices_totally_count;  
            for (int kk = 0; kk < test_img_slices.size(); ++kk) {  
                std::unordered_map<std::string, unsigned int> score_count;  
                // Use ORB for keypoint detection and description  
                cv::Mat descriptors1;  
                auto keypoints1 = cvlib_sub.extractSIFTFeatures(test_img_slices[kk], descriptors1, MAX_FEATURES);  
                if (descriptors1.empty()) {  
                    continue;  
                }  
                slice_count++;  
                if (slice_count > first_num) {  
                    slice_count = 0;  
                    break;  
                }  
				unsigned int bad_fish_split_out = 0;  
                for (const auto& item : trained_dataset) {  
                    auto item_data = item.second;  
                    for (const auto& item_data_item : item_data) {  
                        // Use ORB for keypoint detection and description  
                        cv::Mat descriptors2;  
                        auto keypoints2 = cvlib_sub.extractSIFTFeatures(item_data_item, descriptors2, MAX_FEATURES);  
                        if (descriptors2.empty()) {  
                            continue;  
                        }  
                        /*  
                         * Start comparing  
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
                            score_count[item.first] += goodMatches.size();  
                        } else {  
                            bad_fish_split_out++;  
                            if (bad_fish_split_out > bad_num) {  
                                bad_fish_split_out = 0;  
                                break; // Break out of the inner loop  
                            }  
                        }  
                    }  
                }  
                /*  
                 * Count and sort the result   
                 */  
                std::vector<std::pair<std::string, unsigned int>> sorted_score_counting(score_count.begin(), score_count.end());  
                // Sort the vector of pairs  
                std::sort(sorted_score_counting.begin(), sorted_score_counting.end(), [](const auto& a, const auto& b) {  
                    return a.second > b.second;  
                });  
                if (!sorted_score_counting.empty()) {  
                    auto it = sorted_score_counting.begin();  
                    test_slices_totally_count[it->first]++;  
                }  
            }  
            std::vector<std::pair<std::string, unsigned int>> sorted_final_test_score_counting(test_slices_totally_count.begin(), test_slices_totally_count.end());  
            // Sort the vector of pairs  
            std::sort(sorted_final_test_score_counting.begin(), sorted_final_test_score_counting.end(), [](const auto& a, const auto& b) {  
                return a.second > b.second;  
            });  
            if (!sorted_final_test_score_counting.empty()) {  
                auto it = sorted_final_test_score_counting.begin();  
                rii.objName = it->first;  
            } else {  
                std::cerr << "Error: No matches found for test image " << test_item << std::endl;  
                rii.objName = "Unknown";  
            }  
            auto t_count_end = std::chrono::high_resolution_clock::now();  
            std::chrono::duration<double> timespent = t_count_end - t_count_start;  
            rii.timespent = timespent;  
            return_imgs[test_item] = rii;  
        } catch (const std::exception& e) {  
            std::cerr << "Exception occurred while processing " << test_item << ": " << e.what() << std::endl;  
            continue; // Skip to the next image  
        }  
    }  
    std::cout << "Image recognition completed!" << std::endl;  
}
void cvLib::checkExistingGestures(const cv::Mat& frame_input, std::string& gesture_catched){
    if(frame_input.empty()){
        std::cerr << "cvLib::checkExistingGestures frame input is empty!" << std::endl;
        return;
    }
    if(trained_dataset.empty()){
        std::cerr << "cvLib::checkExistingGestures trained_dataset is empty!" << std::endl;
        return;
    }
    cv::Mat test_img;
    std::vector<std::pair<std::string, unsigned int>> score_count;
    try {
        cv::Mat resizeImg;
        cv::resize(frame_input, resizeImg, cv::Size(ReSIZE_IMG_WIDTH, ReSIZE_IMG_HEIGHT));//$$$$$$$$$$$$
        cv::Mat img_gray;
        cv::cvtColor(resizeImg, img_gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(img_gray, test_img, cv::Size(5, 5), 0);
    } catch(const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return;
    } catch(...) {
        std::cerr << "cvLib::checkExistingGestures try preprocess image unknown errors" << std::endl;
        return;
    }
    if (test_img.empty()) {
        std::cerr << "cvLib::img_recognition: preprocessImg, output test_img is empty!" << std::endl;
        return; // Skip empty images
    }
    cvLib_subclasses cvlib_sub;
    cv::Mat test_descriptors;
    std::vector<cv::KeyPoint> sub_key = cvlib_sub.extractSIFTFeatures(test_img, test_descriptors,MAX_FEATURES);
    if (test_descriptors.empty()) {
        std::cerr << "cvLib::img_recognition: test_img->descriptors is empty!" << std::endl;
        return; // Skip if no descriptors are found
    }
    for (const auto& item : trained_dataset) { // Process trained corpus
        auto item_collections = item.second;
        for (const auto& trained_item : item_collections) {
            cv::Mat trained_descriptors;
            std::vector<cv::KeyPoint> trained_key = cvlib_sub.extractSIFTFeatures(trained_item, trained_descriptors,MAX_FEATURES);
            if (trained_descriptors.empty()) {
                std::cerr << "Warning: trained descriptors are empty for trained_item." << std::endl;
                continue; // Skip this trained item if no descriptors found
            }
            cv::BFMatcher matcher(cv::NORM_L2);
            std::vector<std::vector<cv::DMatch>> knnMatches;
            matcher.knnMatch(test_descriptors, trained_descriptors, knnMatches, 2);
            std::vector<cv::DMatch> goodMatches;
            for (const auto& match : knnMatches) {
                if (match.size() > 1 && match[0].distance < RATIO_THRESH * match[1].distance) {
                    goodMatches.push_back(match[0]);
                }
            }
            if (goodMatches.size() > DE_THRESHOLD) { // DE_THRESHOLD
                score_count.push_back(std::make_pair(item.first, goodMatches.size()));
                // Assuming you have the coordinates of the recognized object
                // For demonstration, let's use dummy coordinates for the rectangle
                cv::Point topLeft(50, 50);    // Top-left corner of the rectangle
                cv::Point bottomRight(150, 150); // Bottom-right corner of the rectangle
                cv::rectangle(frame_input, topLeft, bottomRight, cv::Scalar(0, 255, 0), 2); // Draw rectangle
            }
        }
    }
    if (!score_count.empty()) {
        std::unordered_map<std::string, unsigned int> max_scores;
        // Iterate through the score_count vector
        for (const auto& [key, value] : score_count) {
            // Update the maximum score for the current key
            max_scores[key] = std::max(max_scores[key], value);
        }
        std::vector<std::pair<std::string, double>> sorted_score_counting(max_scores.begin(), max_scores.end());
        // Sort the vector of pairs
        std::sort(sorted_score_counting.begin(), sorted_score_counting.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });
        auto it = sorted_score_counting.begin();
        gesture_catched = it->first;
    }
}
bool cvLib::checkExistingFace(const std::string& faces_folder_path, const cv::Mat& img_input) {
    if (!std::filesystem::exists(faces_folder_path) || !std::filesystem::is_directory(faces_folder_path)) {
        std::cerr << "The folder does not exist or cannot be accessed." << std::endl;
        return false;
    }
    cv::Mat img_gray;
    cv::cvtColor(img_input, img_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(img_gray, img_gray, cv::Size(5, 5), 0);
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create(MAX_FEATURES);
    std::vector<cv::KeyPoint> keypoints_input;
    cv::Mat descriptors_input;
    detector->detectAndCompute(img_gray, cv::noArray(), keypoints_input, descriptors_input);
    if (descriptors_input.empty()) {
        std::cerr << "Input image has no descriptors." << std::endl;
        return false;
    }
    for (const auto& entry : std::filesystem::directory_iterator(faces_folder_path)) {
        if (entry.is_regular_file()) {
            cv::Mat existing_face = cv::imread(entry.path().string());
            if (existing_face.empty()) continue;
            cv::Mat existing_face_gray;
            cv::cvtColor(existing_face, existing_face_gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::KeyPoint> keypoints_existing;
            cv::Mat descriptors_existing;
            detector->detectAndCompute(existing_face_gray, cv::noArray(), keypoints_existing, descriptors_existing);
            if (descriptors_existing.empty()) {
                std::cerr << "Existing face has no descriptors." << std::endl;
                continue;
            }
            cv::BFMatcher matcher(cv::NORM_L2);
            std::vector<std::vector<cv::DMatch>> knnMatches;
            matcher.knnMatch(descriptors_existing, descriptors_input, knnMatches, 2);
            std::vector<cv::DMatch> goodMatches;
            for (const auto& match : knnMatches) {
                if (match.size() > 1 && match[0].distance < RATIO_THRESH * match[1].distance) {
                    goodMatches.push_back(match[0]);
                }
            }
            if (goodMatches.size() > DE_THRESHOLD) {
                //std::cout << "The face image already exists." << std::endl;
                return true;
            }
        }
    }
    return false;
}
void cvLib::onFacesDetected(const std::vector<cv::Rect>& faces, cv::Mat& frame, const std::string& face_folder) {
    if (faces.empty()) return;
    for (size_t i = 0; i < faces.size(); ++i) {
        cv::rectangle(frame, faces[i], cv::Scalar(0, 255, 0), 2);
        std::string text = (i < 9) ? "0" + std::to_string(i + 1) : std::to_string(i + 1);
        cv::putText(frame, text, cv::Point(faces[i].x, faces[i].y - 10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    }
    cv::Rect faceROI = faces[0]; // Save the first detected face
    cv::Mat faceImage = frame(faceROI).clone();
    if (checkExistingFace(face_folder, faceImage)) {
        return;
    }
    std::string fileName = face_folder + "/face_" + std::to_string(faceCount++) + ".jpg";
    cv::imwrite(fileName, faceImage);
    std::cout << "Saved snapshot: " << fileName << std::endl;
    /*
     Detect gestures 
     */
    
}
void cvLib::start_recording(unsigned int webcamIndex){
    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load("/Users/dengfengji/ronnieji/MLCpplib-main/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error: Could not load Haar Cascade model." << std::endl;
        return;
    }
    cv::VideoCapture cap(webcamIndex);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video stream." << std::endl;
        return;
    }
    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Empty frame." << std::endl;
            continue;
        }
        cv::Mat gestrueFrame = frame.clone();
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 10, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
        if (!faces.empty()) {
            onFacesDetected(faces, frame, "/Users/dengfengji/ronnieji/MLCpplib-main/faces");
        }
        std::string catched_gesture;
        checkExistingGestures(gestrueFrame,catched_gesture);
        if(!catched_gesture.empty()){
            std::cout << "Gesture catched: " << catched_gesture << std::endl;
        }
        cv::imshow("Face Detection", frame);
        char key = cv::waitKey(30);
        if (key == 'q') {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
}