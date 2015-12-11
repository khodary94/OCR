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
#include "char_map.h"

#define FORE_THRESH 100

using namespace std;


struct Font{
	string name;
	int ch_count;
};

vector<cv::Mat> visionize(cv::Mat, bool, string, int);
int matchWindows(cv::Mat&, cv::Mat&);
void matchLetters(vector<cv::Mat>&, vector<cv::Mat>&, string&, int);
int minIndex(int[], int);
void init_templates(vector<cv::Mat>&, string, int);
void read_from_file(string, vector<Font>&);
void write_to_file(string, vector<Font>&);

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
    cout << "Done!\n";
}

void doBinary(cv::Mat &image){
    std::cout << "=>Converting Image to binary...";
    for(int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.cols; j++){
            uchar *pixel = &image.at<uchar>(i, j);
            if(*pixel < FORE_THRESH) *pixel = 255;
            else *pixel = 0;
        }
    }
	cout << "Done!\n";
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
	cout << "=>Filing the letters vector...";
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
	cout << "Done!\n";
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
	
	vector<cv::Mat> letters;
	vector<cv::Mat> templates;
	vector<Font> fonts;
	cv::Mat imColor;
	string result;

	if( argc != 2)
    {
		cout <<" No image argument is given to the program" << endl;
		system("pause");
		return -1;
	}
    cout << "Program has started!\n";

	imColor = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

	if(! imColor.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
		system("pause");
        return -1;
    }

	cout << "Please choose one of the following options:\n"<<
			"1. Carry out OCR using one of the existing fonts\n"<<
			"2. Import a new font for OCR\n";

	int choice;
	cin >> choice;
	bool import = choice-1;
	
	read_from_file("templates/Fonts.txt", fonts);
	string templates_dir;

	if(import){
		cout << "Please enter the name of the new font:\n";
		string input;
		cin >> input;
		templates_dir = "templates/" + input;

		Font temp;
		temp.name = input;
		temp.ch_count = 0;

		for(int i=0; i<4; i++){
			string samples[4] = {"templates/uppercase.png", "templates/lowercase.png", "templates/numbers.png", "templates/symbols.png"};
			int offsets[4] = {0, 26, 54, 64};
			imColor = cv::imread(samples[i], CV_LOAD_IMAGE_COLOR);
			
			if(! imColor.data )                              // Check for invalid input
			{
				cout <<  "Could not open or find the image" << std::endl ;
				system("pause");
				return -1;
			}
			letters = visionize(imColor, import, templates_dir, offsets[i]);				//save image templates in their designated folder
			temp.ch_count += letters.size();
		}
		
		fonts.push_back(temp);
		write_to_file("templates/Fonts.txt", fonts);
	}
	else{
		if(fonts.size() == 0)
			cout << "no fonts are imported for carrying our OCR!\n";
		else{
			cout << "Please select one of the existing fonts templates for performing OCR\n";
			for(int i=0; i<fonts.size(); i++)
				cout << i+1 << ". " << fonts[i].name << endl;
			int sel;
			cin >> sel;
			int font_chars = fonts[sel-1].ch_count;

			//Initialize templates
			templates_dir = "templates/" + fonts[sel-1].name;
			init_templates(templates, templates_dir, font_chars);
		
			//Read letters
			letters = visionize(imColor, import, "", 0);							//load letters and save their images in their folder
		
			//Match letters to templates
			matchLetters(letters, templates, result, font_chars);
			cout << "\nThe program reads:\n" << result << endl;
		}
	}

	system("pause");
    return 0;
}

vector<cv::Mat> visionize(cv::Mat imColor, bool import, string outloc, int offset){

	cv::Mat imGrayscale, imBinary;
	vector<std::vector<int>> accum;
    vector<cv::Mat> letters;
    vector<Blob> vecBlobs;

	//converting to grayscale and smoothing
	cv::cvtColor( imColor, imGrayscale, CV_BGR2GRAY );
	cv::Mat imTester = imGrayscale.clone();
    smoothTran(imGrayscale, imBinary);
    doBinary(imBinary);

	//--------------------------------------------------------------------------------------------------------------------
	//CCL algorithm
    CCL(imBinary, accum);
	createBlobs(accum, vecBlobs);
    findCorners(imColor, vecBlobs);
	createLetters(accum, vecBlobs, letters);
	string dir;
	if(import)
		dir = outloc;
	else
		dir = "letters";
	for(int i = 0; i < letters.size(); i++){				//write the found letters to files for testing and debugging
		cv::imwrite(dir+"/_" + std::to_string(offset+i) + ".png", letters[i]);
		//printImg(templates[30], "_" + std::to_string(1)); 
    }
	cv::imwrite("out.png", imBinary);
    cv::imwrite("outletters.png", imColor);
	return letters;
}

void read_from_file(string fname, vector<Font>& ar){
	ifstream infile;
	infile.open(fname);

	if(!infile.fail()){
		while(!infile.eof()){
			Font temp;
			infile >> temp.name;
			infile >> temp.ch_count;
			if(temp.name != "")
				ar.push_back(temp);
		}
	}
}

void write_to_file(string fname, vector<Font>& ar){
	ofstream outfile;
	outfile.open(fname);

	if(!outfile.fail()){
		for(int i=0; i<ar.size(); i++){
			if(i!=0)
				outfile << endl;
			outfile << ar[i].name << '\t' << ar[i].ch_count;
		}
	}
}

int matchWindows(cv::Mat& src, cv::Mat& temp){
	//float diff_acc=0, src_acc=0, temp_acc=0;
	int coeff = 0;
	cv::resize(temp, temp, cv::Size(src.cols, src.rows));
	for(int i=0; i<src.rows; i++){
		for(int j=0; j<src.cols; j++){
			int src_val = src.at<uchar>(i, j);
			int temp_val = temp.at<uchar>(i, j);
			coeff += std::pow(src_val - temp_val, 2.0);
			//diff_acc += std::pow(src_val - temp_val, 2.0);
			//src_acc += std::pow(src_val, 2.0);
			//temp_acc += std::pow(temp_val, 2.0);
		}
	}
	//return diff_acc / (src_acc * temp_acc);
	return coeff;
}

int minIndex(int ar[], int n){
	int min = ar[0], mini = 0;
	for(int i=1; i<n; i++){
		if(ar[i] < min){
			min = ar[i];
			mini = i;
		}
	}
	return mini;
}

void matchLetters(vector<cv::Mat>& letters, vector<cv::Mat>& template_letters, string& output, int letter_count){
	cout << "matching characers...";
	for(int i=0; i<letters.size(); i++){
		cv::Mat curLetter = letters[i];
		int* coeffs;
		coeffs = new int[letter_count];
		for(int j=0; j<template_letters.size(); j++){
			cv::Mat tempLetter = template_letters[j].clone();
			coeffs[j] = matchWindows(curLetter, tempLetter);
			//cout << coeffs[j] << " ";
			//printImg(curLetter, "letter");
			//printImg(tempLetter, ""+lmap[j]);
			//cv::waitKey(0);
		}
		int ind = minIndex(coeffs, letter_count);
		output += lmap[ind];
	}
	cout << "Done!\n";
}

void init_templates(vector<cv::Mat>& template_letters, string dir, int letter_count){
	cout << "=>Loading letter templates...";
	for(int i=0; i<letter_count; i++){
		cv::Mat temp;
		temp = cv::imread(dir+"/_"+std::to_string(i)+".png", CV_LOAD_IMAGE_GRAYSCALE);
		template_letters.push_back(temp);
	}
	cout << "Done!\n";
}