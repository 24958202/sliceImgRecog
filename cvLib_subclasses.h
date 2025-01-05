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
		/*
		 * Function to detect object's position according to the const std::vector<int>& selectedSlices
		 * para1: const std::vector<int>& selectedSlices, 
		 * if an image was divided by 100 pieces (10x10),  gridSize = 10, matrix_size = 100;
		 * para2: gridSize 
		 * para3: matrix_size
		 * 
		 */
		std::vector<cv::Rect> detectClusters(const std::vector<int>&, int, int);\
		/*
		 * Function to mark objects according to the slices index
			para1: original image, much be resize to ReSIZE_IMG_WIDTH,ReSIZE_IMG_HEIGHT
			para2: gridSize
			para3: the slices of the object detected 
			para4: the slices of the object's name (first value= object's name, second value = slice index)
		 */
		void markClusters(cv::Mat&, int, const std::vector<cv::Rect>&, const std::vector<std::pair<std::string, unsigned int>>&);
		/*
		 * Function to pass slices of image, objects in the image to above functions to mark.
		   para1: original image , much be resize to ReSIZE_IMG_WIDTH,ReSIZE_IMG_HEIGHT
		   para2: image slices with objects detection mark, first value = image slice, second value = object mark
		   para3: object names with its related image slice index
		   if an image was divided by 100 pieces (10x10),  gridSize = 10, matrix_size = 100;
		   para4: gridSize
		   para4: matrix_size
		 */
		void detect_obj_and_draw(cv::Mat&, const std::vector<std::pair<cv::Mat, unsigned int>>&,  
                          const std::vector<std::pair<std::string, unsigned int>>&, int, int);
        
};

#ifdef __cplusplus
}
#endif
#endif
/*
	std::vector<cv::Mat> slice_the_image;  
    preprocessImg("/home/ronnieji/ronnieji/Kaggle/train/panda/496bd52415.jpg", 448, 448, slice_the_image);  
    if (slice_the_image.empty()) {  
        std::cerr << "Slices dataset is empty!" << std::endl;  
        return -1;  
    }  
    // Example usage  
    std::vector<std::pair<cv::Mat, unsigned int>> imgSlices(slice_the_image.size());  
    // Load your slices into imgSlices...  
    for (size_t i = 0; i < slice_the_image.size(); ++i) {  
        unsigned int score = (i == 2 || i == 3 || i == 12 || i == 13 || i == 5 || i == 6 || i == 15 || i == 16) ? 1 : 0; // Set score based on condition  
        imgSlices[i] = {slice_the_image[i], score}; // Assign the slice and its score  
    }  
    // Create object names based on slice indices  
    std::vector<std::pair<std::string, unsigned int>> obj_names = {  
        {"Apple", 2}, {"Apple", 3}, {"Banana", 5}, {"Banana", 6},  
        {"Apple", 12}, {"Apple", 13}, {"Banana", 15}, {"Banana", 16}  
    };  
    cv::Mat img = cv::imread("/home/ronnieji/ronnieji/Kaggle/train/panda/496bd52415.jpg", cv::IMREAD_COLOR);  
    if (img.empty()) {  
        std::cerr << "Error: Image not loaded correctly from path!" << std::endl;  
        return -1; // Handle the error appropriately  
    }  
    // Resize the image  
    cv::Mat resizeImg;  
    cv::resize(img, resizeImg, cv::Size(448, 448));    
    // Call the function  
    detect_obj_and_draw(resizeImg, imgSlices, obj_names, 64, 448); // Pass the slice size (64 for 64x64 slices)  
    return 0;  
 * */