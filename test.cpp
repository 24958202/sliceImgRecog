/*
 * Program for Facial recognition / Gesture recognition / object recognition
 * Put training image in the folder like:
 
  main folder
       |
  subfolder1, subfolder2,subfolder3...
  
 g++ /Users/dengfengji/ronnieji/lib/new_cvLib/main/test.cpp -o /Users/dengfengji/ronnieji/lib/new_cvLib/main/test -I/Users/dengfengji/ronnieji/lib/new_cvLib/include -I/Users/dengfengji/ronnieji/lib/new_cvLib/src /Users/dengfengji/ronnieji/lib/new_cvLib/src/*.cpp -I/opt/homebrew/Cellar/opencv/4.10.0_12/include/opencv4 -L/opt/homebrew/Cellar/opencv/4.10.0_12/lib -Wl,-rpath,/opt/homebrew/Cellar/opencv/4.10.0_12/lib -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_features2d -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_video -DOPENCV_VERSION=4.10.0_12 -std=c++20
 * */
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include "cvLib.h"
#include "cvLib_subclasses.h"
bool isValidImage(const std::string& img_path){
    if(img_path.empty()){
        return false;
    }
    std::vector<std::string> image_extensions{
        ".jpg",
        ".JPG",
        ".jpeg",
        ".JPEG",
        ".png",
        ".PNG"
    };
    for(const auto& item : image_extensions){
        if(img_path.find(item) != std::string::npos){
            return true;
        }
    }
    return false;
}
void test_image_recognition(){
    std::vector<std::string> testimgs;
    std::string sub_folder_path = "/home/ronnieji/ronnieji/Kaggle/test"; //"/Users/dengfengji/ronnieji/Kaggle/test";
    for (const auto& entrySubFolder : std::filesystem::directory_iterator(sub_folder_path)) {  
        if (entrySubFolder.is_regular_file()) {  
            std::string imgFilePath = entrySubFolder.path().string();  
            if(isValidImage(imgFilePath)){
                testimgs.push_back(imgFilePath);
            }
        }
    }
    std::unordered_map<std::string,return_img_info> results;
    cvLib cvl_j;
    cvl_j.ini_trained_data("/home/ronnieji/ronnieji/16_16_visual/main/model.dat");//load model.dat before img_recognition
    cvl_j.img_recognition(
        testimgs,
        results,
		0.55f,
		0.2f
        );
    if(!results.empty()){
        for(const auto& item : results){
            auto it = item.second;
			std::chrono::duration<double> time_spent = it.timespent;
            std::cout << item.first << " is a/an: " << it.objName << '\n';
            std::cout << "Time spent: " << time_spent << std::endl;
        }
    }
}
void compareTwoImages(){
    cvLib_subclasses cv_sub_j;
    if(cv_sub_j.img1_img2_are_matched(
        "/Users/dengfengji/ronnieji/Kaggle/test/sample_64_64.jpg",
        "/Users/dengfengji/ronnieji/Kaggle/test/img_0601_jpeg.rf.49476aa25fb1745b71df5401bebaee89.jpg",
        10,
        1000,
        480,
        480,
        0.95f
    )){
        std::cout << "Matched!" << std::endl;
    }
    else{
        std::cout << "Did not match!" << std::endl;
    }
}
int main(){
    /*
            compress all images
    */
//     cvLib cvl_j;
//     cvl_j.img_compress("/Users/dengfengji/ronnieji/Kaggle/archive-2/train",18);
     
     cvLib cvl_j;
     cvl_j.train_img_occurrences(
         "/home/ronnieji/ronnieji/Kaggle/train",
         "/home/ronnieji/ronnieji/16_16_visual/main/model.dat"
     );     
     test_image_recognition();
	
     //cvl_j.ini_trained_data("/Users/dengfengji/ronnieji/lib/new_cvLib/main/model.dat");
     //cvl_j.start_recording(0);
     
     //compareTwoImages();
}