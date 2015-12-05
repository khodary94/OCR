//
//  main.cpp
//  Vision_project
//
//  Created by Mahmoud Khodary on 12/1/15.
//  Copyright (c) 2015 Mahmoud Khodary. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

#define FORE_THRESH 100

struct Blob{
    int minY, maxY,
    minX, maxX;
    int blob_number;
    int size;
};

void printImg(cv::Mat in, std::string title){
    cv::namedWindow(title, CV_WINDOW_AUTOSIZE);
    cv::imshow(title, in);
}

int mod(int x, int k){
    int temp = x%k;
    if(temp < 0) temp = x + k;

    return temp;
}

void smoothTran(const cv::Mat &image, cv::Mat& result){

    std::cout << "=>Smoothing Image...";
    int x = image.rows;
    int y = image.cols;
    float pixel = 0;

    result = image;

    const int kx = 3, ky = 3;
    float kernel[kx][ky] = {
        {1/9.0, 1/9.0, 1/9.0},
        {1/9.0, 1/9.0, 1/9.0},
        {1/9.0, 1/9.0, 1/9.0},
    }; //blur image using average filter 3x3

    //the use of mod is to not go out of bound
    for(int i = 0; i < x; i++){
        for(int j = 0; j < y; j++){
            for(int i2 = - 1; i2 <= kx/2; i2++){
                for(int j2 = - 1; j2 <= ky/2; j2++){
                    pixel += float(image.at<uchar>(mod(i + i2, x - 1), mod(j + j2, y - 1))) * kernel[i2 + 1][j2 + 1];
                }
            }
            result.at<uchar>(i, j) = pixel;
            pixel = 0;
        }
    }
    std::cout << "Done!\n";
}

void doBinary(cv::Mat &image){
    std::cout << "=>Converting Image to binary...\n";
    for(int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.cols; j++){
            uchar *pixel = &image.at<uchar>(i, j);
            if(*pixel < FORE_THRESH) *pixel = 255;
            else *pixel = 0;
        }
    }
}

void updateAccum(int a, int b, int x, int y, std::vector<std::vector<int>>& vec){
    for(int i = 0; i < vec.size(); i++){
        for(int j = 0; j < vec[i].size(); j++){
            if(vec[i][j] == b) vec[i][j] = a;
        }
    }
}

void CCL(const cv::Mat& image, std::vector<std::vector<int>>& accum){

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

int checkBlob(int a, const std::vector<Blob>& vec){
    for(int i = 0; i < vec.size(); i++){
        if(a == vec[i].blob_number) return i;
    }
    return - 1;
}

int createBlobs(const std::vector<std::vector<int>>& accum, std::vector<Blob>& vecBlobs){
    //bool flag = false;
    std::cout << "=>Creating Blobs...";
    int blobCount = 0;
    Blob blob;
    for(int i = 0; i < accum.size(); i++){
        for(int j = 0; j < accum[i].size(); j++){
            if(accum[i][j] == 0) continue;

            int exist = checkBlob(accum[i][j], vecBlobs);
            if(exist == - 1){
                blob.blob_number = accum[i][j];

                blob.minY = i;
                blob.maxY = i;
                blob.minX = j;
                blob.maxX = j;

                //std::cout << "First Point: " << cv::Point(j, i) << '\t';

                blob.size = 1;
                vecBlobs.push_back(blob);
                blobCount++;
            }else{
                if(vecBlobs[exist].minX > j){
                    vecBlobs[exist].minX = j;
                    //std::cout << "y_minX: " << cv::Point(j, i) << '\t';
                }
                if(vecBlobs[exist].maxX < j){
                    vecBlobs[exist].maxX = j;
                    //std::cout << "y_maxX: " << cv::Point(j, i) << '\t';
                }
                if(vecBlobs[exist].minY > i){
                    vecBlobs[exist].minY = i;
                    //std::cout << "x_minY: " << cv::Point(j, i) << '\t';
                }
                if(vecBlobs[exist].maxY < i){
                    vecBlobs[exist].maxY = i;
                    //std::cout << "x_maxY: " << cv::Point(j, i) << '\t';
                }
                //vecBlobs[exist].endPoint = cv::Point(j, i);
                vecBlobs[exist].size += 1;
            }
        }
    }
    std::cout << " Blob Count: " << blobCount << " Done!\n";
    return blobCount;
}

void createLetters(const std::vector<std::vector<int>>& accum, const std::vector<Blob>& vecBlobs, std::vector<cv::Mat>& letters){
    for(int i = 0; i < vecBlobs.size(); i++){
        int w_min, l_min, w_max, l_max;
        int l, w;

        l_min = vecBlobs[i].minY;
        l_max = vecBlobs[i].maxY;
        w_min = vecBlobs[i].minX;
        w_max = vecBlobs[i].maxX;
        l = l_max - l_min + 1;
        w = w_max - w_min + 1;

        cv::Mat temp(l, w, CV_LOAD_IMAGE_GRAYSCALE);
        for(int j = 0; j < l; j++){
            for(int k = 0; k < w; k++){

                int z = accum[j + l_min][k + w_min];
                if( z > 0)
                    temp.at<uchar>(j, k) = 0;
                else
                    temp.at<uchar>(j, k) = 255;
            }
        }
        letters.push_back(temp);
    }
}

bool findCorners(cv::Mat& image, std::vector<Blob>& b){
    bool flag = false;
    std::cout << "=>Finding Corners...";
    for(int i = 0; i < b.size(); i++){
        //sflag = checkPoints(b[i]);
        cv::circle(image, cv::Point(b[i].minX, b[i].minY), 2, cv::Scalar(0, 255, 0));
        cv::circle(image, cv::Point(b[i].minX, b[i].maxY), 2, cv::Scalar(0, 255, 0));
        cv::circle(image, cv::Point(b[i].maxX, b[i].minY), 2, cv::Scalar(0, 255, 0));
        cv::circle(image, cv::Point(b[i].maxX, b[i].maxY), 2, cv::Scalar(0, 255, 0));

        cv::rectangle(image, cv::Point(b[i].minX, b[i].minY), cv::Point(b[i].maxX, b[i].maxY), cv::Scalar(0, 0, 255));
    }

    std::cout << "Done!\n";
    return flag;
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
    /*cv::Mat imResize(imColor.rows/2, imColor.cols/2, imColor.type());
    cv::Mat imResize2(imGrayscale.rows/2, imGrayscale.cols/2, imGrayscale.type());
    cv::resize(imGrayscale, imResize2, imResize2.size(), 0, 0);
    cv::resize(imColor, imResize, imResize.size(), 0, 0);
    cv::threshold(imGrayscale, imBinary, FORE_THRESH, 255, 0);*/

    std::vector<std::vector<int>> accum;
    std::vector<cv::Mat> letters;
    std::vector<Blob> vecBlobs;

    smoothTran(imGrayscale, imBinary);
    doBinary(imBinary);
    CCL(imBinary, accum);

    createBlobs(accum, vecBlobs);
    findCorners(imColor, vecBlobs);
    createLetters(accum, vecBlobs, letters);

    for(int i = 0; i < letters.size(); i++){
        printImg(letters[i], "Letter" + std::to_string(i));
    }
    cv::imwrite("/Users/macbookpro/Desktop/out.jpg", imBinary);
    cv::imwrite("/Users/macbookpro/Desktop/outcol.jpg", imColor);
    //printImg(imColor, "Colored Image");
    //printImg(imGrayscale, "Grayscale Image");
    //printImg(imBinary, "Binary");
    cv::waitKey();
    return 0;
}
