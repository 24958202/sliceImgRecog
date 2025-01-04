#ifndef CVLIB_SUBCLASSES_H
#define CVLIB_SUBCLASSES_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C" {
#endif
class cvLib_subclasses{
    public:
        /*
            check if a file extension contains image extension
        */
        bool isValidImage(const std::string&);
        /*
            Function to compress images 
            para1: input image file path
            para2: output image file path
            para3: image quality 0-100
         */
        void compressJPEG(const std::string&, const std::string&, int);
        /*
         Function to resize and blur input image
         para1: image path 
         para2: ReSIZE_IMG_WIDTH (process image width: 448 default)
         para3: ReSIZE_IMG_HEIGHT(process image height: 448 default)
         para4: output image slices std::vector<cv::Mat>
         */
        void preprocessImg(const std::string&,const unsigned int, const unsigned int,std::vector<cv::Mat>&);
        /*
         *Preprocess image , output gray image
            para1: cv::Mat image input/output 
            para2: ReSIZE_IMG_WIDTH (process image width: 448 default)
            para3: ReSIZE_IMG_HEIGHT(process image height: 448 default)
         */
        void preprocessImg_gray(cv::Mat&,const unsigned int, const unsigned int);
        /*
         para1: input image cv::Mat
         para2: output descriptors
         para3: MAX_FEATURES(default: 1000)
         */
        std::vector<cv::KeyPoint> extractSIFTFeatures(const cv::Mat&, cv::Mat&, const unsigned int);
        void saveModel_keypoint(const std::unordered_map<std::string, std::vector<cv::Mat>>& featureMap, const std::string& filename);
        /*
          Function to convert cv::KeyPoint to descriptors
          para1: input keypoints
          para2: MAX_FEATURES(default: 1000)
          para3: ReSIZE_IMG_WIDTH (process image width: 448 default)
          para4: ReSIZE_IMG_HEIGHT(process image height: 448 default)
        */
        cv::Mat computeDescriptors(std::vector<cv::KeyPoint>&, const unsigned int, const unsigned int, const unsigned int);
        /*
            Function to compare two images 
            para1: img1 file path
            para2: img2 file path
            para3: DE_THRESHOLD
            para4: MAX_FEATURES(default: 1000)
            para5: ReSIZE_IMG_WIDTH (process image width: 448 default)
            para6: ReSIZE_IMG_HEIGHT(process image height: 448 default)
            para7: RATIO_THRESH (default 0.95f);          // Ratio threshold for matching
        */
        bool img1_img2_are_matched(const std::string&, const std::string&, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const float);
        /*
         *  Function to compare two images 
            para1: img1 cv::Mat input
            para2: img2 cv::Mat input
            para3: DE_THRESHOLD
            para4: MAX_FEATURES(default: 1000)
            para5: ReSIZE_IMG_WIDTH (process image width: 448 default)
            para6: ReSIZE_IMG_HEIGHT(process image height: 448 default)
            para7: RATIO_THRESH (default 0.95f);          // Ratio threshold for matching
         */
        bool img1_img2_are_matched_cvMat(const cv::Mat&, const cv::Mat&, const unsigned int, const unsigned int, const unsigned int, const unsigned int, const float);
		 /*
         * Function to sort slices of an image by their similities 
         * para1: slices of an image
         * para2: similarityThreshold
         * para3: emptyThreshold
         *      // Define thresholds
                const double emptyThreshold = 10.0;     // Below this intensity means it's background
                const double similarityThreshold = 0.5; // (Optional for filtering similar slices)
                // Sort and filter slices
                sortAndFilterSlices(slices, similarityThreshold, emptyThreshold);
                // Print and display results
                for (size_t i = 0; i < slices.size(); ++i) {
                    std::cout << "Slice " << i << ": Mean intensity = " << cv::mean(slices[i])[0] << std::endl;
                }
         */
        void sortAndFilterSlices(std::vector<cv::Mat>&, const double, const double);
		/*
		 * Instead of sorting by similarty,sort the slices in descending order of their mean intensity
		   para1: slices of an image
         * para2: emptyThreshold
		 */
		void sortByMeanIntensity(std::vector<cv::Mat>&, const double);
        
};

#ifdef __cplusplus
}
#endif
#endif
