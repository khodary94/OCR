//
//  main.cpp
//  Vision_project
//
//  Created by Mahmoud Khodary on 12/1/15.
//  Copyright (c) 2015 Mahmoud Khodary. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

#define FORE_THRESH 160

void printImg(cv::Mat in, std::string title){
    cv::namedWindow(title, CV_WINDOW_AUTOSIZE);
    cv::imshow(title, in);
}

/*void doBinary(cv::Mat &image){
    std::cout << "=>Converting Image to binary...\n";
    for(int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.cols; j++){
            uchar *pixel = &image.at<uchar>(i, j);
            if(*pixel > FORE_THRESH) *pixel = 255;
            else *pixel = 0;
        }
    }
}*/

void updateAccum(int a, int b, int x, int y, std::vector<std::vector<int>>& vec){
    for(int i = 0; i <= x; i++){
        for(int j = 0; j <= y; j++){
            if(vec[i][j] == b) vec[i][j] = a;
        }
    }
}

void CCL2(const cv::Mat& image, std::vector<std::vector<int>>& accum){

    std::cout << "=>Doing CCL...";

    int blobCount = 0;

    accum.resize(image.rows);
    for(int i = 0; i < accum.size(); i++){
        accum[i].resize(image.cols);
    }

    for(int i = 0; i < accum.size(); i++){
        for(int j = 0; j < accum[i].size(); j++)
            accum[i][j] = 0;
    }

    for(int i = 1; i < image.rows; i++){
        for(int j = 1; j < image.cols; j++){
            uchar pixel_current = image.at<uchar>(i, j);
            if(pixel_current == 0) continue;

            uchar pixel_left = image.at<uchar>(i - 1,j), pixel_up = image.at<uchar>(i, j - 1);

            //case no surrounding foreground
            if(pixel_left != 0)
                accum[i][j] = accum[i - 1][j];
            if(pixel_up != 0){
                if(accum[i][j] != 0){
                    int min = std::min(accum[i-1][j], accum[i][j-1]);
                    int max = std::max(accum[i-1][j], accum[i][j-1]);
                    accum[i][j] = min;
                    if(min != max){
                        //updateEquiTable(min, max, equiTable);
                        updateAccum(min, max, i, j, accum);
                    }
                }else{
                    accum[i][j] = accum[i][j - 1];
                }
            }
            if(accum[i][j] == 0){ //still not assigned, so create new blob.
                blobCount++;
                /*std::vector<int> blob_number;
                blob_number.push_back(blobCount);
                equiTable.push_back(blob_number);*/
                accum[i][j] = blobCount;
            }
        }
    }
    std::cout << "Done!\n";
}

int main(int argc, const char * argv[]) {
    // insert code here...
    std::cout << "Hello, World!\n";

    cv::Mat imColor, imGrayscale, imBinary;
    imColor = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    imGrayscale = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

    if(!imColor.data) {         // Check for invalid input
        std::cout <<  "Could not open or find the image\n";
        return -1;
    }
    //cv::Mat imResize(imColor.rows/2, imColor.cols/2, imColor.type());
    //cv::Mat imResize2(imGrayscale.rows/2, imGrayscale.cols/2, imGrayscale.type());
    //cv::resize(imGrayscale, imResize2, imResize2.size(), 0, 0);
    //cv::resize(imColor, imResize, imResize.size(), 0, 0);
    cv::threshold(imGrayscale, imBinary, FORE_THRESH, 255, 0);
    //doBinary(imGrayscale);

    printImg(imGrayscale, "Grayscale Image");
    printImg(imBinary, "Binary");
    cv::waitKey();
    return 0;
}
