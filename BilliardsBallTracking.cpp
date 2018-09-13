// BilliardsBallTracking.cpp : 731 Project.

/*
Author: Dhaval Chauhan dmc8383@rit.edu
*/

#include "stdafx.h"

//opencv
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

//C
#include <stdio.h>

//C++
#include <functional>
#include <iostream>
#include <sstream>

//Include namespaces
using namespace cv;
using namespace std;

// Global variables
Mat frame; //current frame
Mat frame_gray; //gray frame
Mat frame_blur; //blur frame
Mat frame_sharp; //sharp frame
Mat table_mask; //Pool table
Vec4i table_minmax_XY; //Table coordinates

char keyboard; //input from keyboard


void segmentPoolTable(Mat frameBGR)
{
	Mat frame_pool(frame.rows, frame.cols, CV_8UC1, Scalar(0));

	/*for (int y = 0; y<frame.rows; y++)
	{
	for (int x = 0; x<frame.cols; x++)
	{

	if (frameBGR.at<Vec3b>(y, x)[0] >= 150 && frameBGR.at<Vec3b>(y, x)[0] <= 170 &&
	frameBGR.at<Vec3b>(y, x)[1] >= 70 && frameBGR.at<Vec3b>(y, x)[1] <= 90 &&
	frameBGR.at<Vec3b>(y, x)[2] <= 20)
	{
	frame_pool.at<uchar>(y, x) = 255;
	}
	else
	frame_pool.at<uchar>(y, x) = 0;
	}
	}*/

	for (int y = 0; y<frame.rows; y++)
	{
		for (int x = 0; x<frame.cols; x++)
		{

			if (frameBGR.at<Vec3b>(y, x)[0] >= 170 && frameBGR.at<Vec3b>(y, x)[0] <= 230 &&
				frameBGR.at<Vec3b>(y, x)[1] >= 100 && frameBGR.at<Vec3b>(y, x)[1] <= 140 &&
				frameBGR.at<Vec3b>(y, x)[2] <= 20)
			{
				frame_pool.at<uchar>(y, x) = 255;
			}
			else
				frame_pool.at<uchar>(y, x) = 0;
		}
	}

	//Erode the foreground in the mask
	int morph_size = 15;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	morphologyEx(frame_pool, frame_pool, MORPH_ERODE, element, Point(-1, -1));


	//Dilate the foreground in the mask
	morph_size = 15;
	element = getStructuringElement(MORPH_ELLIPSE, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	morphologyEx(frame_pool, frame_pool, MORPH_DILATE, element, Point(-1, -1));

	table_minmax_XY = Vec4i(9999, -1, 9999, -1);

	for (int y = 0; y < frame_pool.rows; y++)
	{
		for (int x = 0; x < frame_pool.cols; x++)
		{
			if (frame_pool.at<uchar>(y, x) == 255)
			{
				if (x < table_minmax_XY[0])
					table_minmax_XY[0] = x;
				if (x > table_minmax_XY[1])
					table_minmax_XY[1] = x;
				if (y < table_minmax_XY[2])
					table_minmax_XY[2] = y;
				if (y > table_minmax_XY[3])
					table_minmax_XY[3] = y;
			}
		}
	}

	table_minmax_XY[0] = table_minmax_XY[0] - 30;
	table_minmax_XY[1] = table_minmax_XY[1] + 30;
	table_minmax_XY[2] = table_minmax_XY[2] - 30;
	table_minmax_XY[3] = table_minmax_XY[3] + 30;
	//imshow("New", frame_pool);
	table_mask = frame_pool;
}


Vec3f subtractTableColor(Mat3b ball_frame)
{
	Vec3f avg_BGR(0.0, 0.0, 0.0);
	int count = 0;

	for (int y = 0; y < ball_frame.rows; y++)
	{
		for (int x = 0; x < ball_frame.cols; x++)
		{
			/*if (!(ball_frame.at<Vec3b>(y, x)[0] >= 150 && ball_frame.at<Vec3b>(y, x)[0] <= 170 &&
			ball_frame.at<Vec3b>(y, x)[1] >= 70 && ball_frame.at<Vec3b>(y, x)[1] <= 90 &&
			ball_frame.at<Vec3b>(y, x)[2] <= 20))*/
			if (!(ball_frame.at<Vec3b>(y, x)[0] >= 170 && ball_frame.at<Vec3b>(y, x)[0] <= 230 &&
				ball_frame.at<Vec3b>(y, x)[1] >= 100 && ball_frame.at<Vec3b>(y, x)[1] <= 140 &&
				ball_frame.at<Vec3b>(y, x)[2] <= 20))
			{
				avg_BGR[0] = avg_BGR[0] + ball_frame.at<Vec3b>(y, x)[0];
				avg_BGR[1] = avg_BGR[1] + ball_frame.at<Vec3b>(y, x)[1];
				avg_BGR[2] = avg_BGR[2] + ball_frame.at<Vec3b>(y, x)[2];
				count++;
			}
		}
	}
	avg_BGR[0] = avg_BGR[0] / count;
	avg_BGR[1] = avg_BGR[1] / count;
	avg_BGR[2] = avg_BGR[2] / count;

	//cout << avg_BGR << endl;
	return avg_BGR;
}


int identifyBallNumber(Mat3b ball_frame)
{
	Vec3f avg_BGR = subtractTableColor(ball_frame);

	////White
	//if (avg_BGR[0] > 160 && avg_BGR[1] > 160 && avg_BGR[2] > 160)
	//	return 16;		
	//
	////Black
	//if (avg_BGR[0] < 85 && avg_BGR[1] < 50 && avg_BGR[2] < 50)
	//	return 8;

	////Yellow
	//if (avg_BGR[0] < 80 && avg_BGR[1] < 140 && avg_BGR[2] < 255 &&
	//	avg_BGR[0] > 50 && avg_BGR[1] > 105 && avg_BGR[2] > 120)
	//	return 1;

	////Blue
	//if (avg_BGR[0] < 255 && avg_BGR[1] < 50 && avg_BGR[2] < 40 &&
	//	avg_BGR[0] > 100 && avg_BGR[1] > 0 && avg_BGR[2] > 0)
	//	return 2;

	////Red
	//if (avg_BGR[0] < 90 && avg_BGR[1] < 50 && avg_BGR[2] < 255 &&
	//	avg_BGR[0] > 70 && avg_BGR[1] > 30 && avg_BGR[2] > 100)
	//	return 3;

	////Pink
	//if (avg_BGR[0] < 140 && avg_BGR[1] < 80 && avg_BGR[2] < 255 &&
	//	avg_BGR[0] > 100 && avg_BGR[1] > 0 && avg_BGR[2] > 130)
	//	return 4;

	////Orange
	//if (avg_BGR[0] < 80 && avg_BGR[1] < 90 && avg_BGR[2] < 255 &&
	//	avg_BGR[0] > 0 && avg_BGR[1] > 50 && avg_BGR[2] > 140)
	//	return 5;

	////Green
	//if (avg_BGR[0] < 110 && avg_BGR[1] < 100 && avg_BGR[2] < 50 &&
	//	avg_BGR[0] > 40 && avg_BGR[1] > 40 && avg_BGR[2] > 0)
	//	return 6;

	////Brown/Caramel
	//if (avg_BGR[0] < 90 && avg_BGR[1] < 90 && avg_BGR[2] < 140 &&
	//	avg_BGR[0] > 60 && avg_BGR[1] > 60 && avg_BGR[2] > 80)
	//	return 7;


	//White
	if (avg_BGR[0] > 170 && avg_BGR[1] > 170 && avg_BGR[2] > 170)
		return 16;

	//Black
	if (avg_BGR[0] < 90 && avg_BGR[1] < 50 && avg_BGR[2] < 50)
		return 8;

	//Yellow
	if (avg_BGR[0] < 100 && avg_BGR[1] < 255 && avg_BGR[2] < 255 &&
		avg_BGR[0] > 0 && avg_BGR[1] > 150 && avg_BGR[2] > 150)
		return 1;

	//Blue
	if (avg_BGR[0] < 255 && avg_BGR[1] < 80 && avg_BGR[2] < 40 &&
		avg_BGR[0] > 120 && avg_BGR[1] > 0 && avg_BGR[2] > 0)
		return 2;

	//Red
	if (avg_BGR[0] < 120 && avg_BGR[1] < 90 && avg_BGR[2] < 255 &&
		avg_BGR[0] > 70 && avg_BGR[1] > 30 && avg_BGR[2] > 130)
		return 3;

	//Pink
	if (avg_BGR[0] < 160 && avg_BGR[1] < 110 && avg_BGR[2] < 255 &&
		avg_BGR[0] > 110 && avg_BGR[1] > 60 && avg_BGR[2] > 170)
		return 4;

	//Orange
	if (avg_BGR[0] < 120 && avg_BGR[1] < 120 && avg_BGR[2] < 255 &&
		avg_BGR[0] > 0 && avg_BGR[1] > 60 && avg_BGR[2] > 140)
		return 5;

	//Green
	if (avg_BGR[0] < 110 && avg_BGR[1] < 100 && avg_BGR[2] < 50 &&
		avg_BGR[0] > 40 && avg_BGR[1] > 40 && avg_BGR[2] > 0)
		return 6;

	//Brown/Caramel
	if (avg_BGR[0] < 110 && avg_BGR[1] < 110 && avg_BGR[2] < 140 &&
		avg_BGR[0] > 80 && avg_BGR[1] > 80 && avg_BGR[2] > 80)
		return 7;

	return 0;
}


/*
Reads in a video file
*/
void processVideo(char* videoFilename)
{
	//create the capture object
	VideoCapture capture(videoFilename);

	if (!capture.isOpened())
	{
		//error in opening the video input
		cerr << "Unable to open video file: " << videoFilename << endl;
		exit(EXIT_FAILURE);
	}

	int frame_width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	VideoWriter video("out30.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(frame_width, frame_height), true);


	//read input data. ESC or 'q' for quitting
	keyboard = 0;
	while (keyboard != 'q' && keyboard != 27)
	{
		//read the current frame
		if (!capture.read(frame))
		{
			cerr << "Unable to read next frame." << endl;
			cerr << "Exiting..." << endl;
			break;
			//exit(EXIT_FAILURE);
		}

		//get the frame number and write it on the current frame
		stringstream ss;

		//create a rectangular frame to display the frame number
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
			cv::Scalar(255, 255, 255), -1);
		ss << capture.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();

		if (stoi(frameNumberString) == 1)
		{
			segmentPoolTable(frame);
		}

		//for frames
		if (stoi(frameNumberString) > 1)
		{
			//Display the frame number
			putText(frame, frameNumberString.c_str(), cv::Point(35, 15),
				FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);

			// split BGR
			Mat frame_bgr[3];   //destination array
			split(frame, frame_bgr);//split source

									// Convert it to gray
			cvtColor(frame, frame_gray, CV_BGR2GRAY);

			// Create vector array for storing circle information
			vector<Vec3f> circles;

			// Apply the Hough Transform to find the circles
			HoughCircles(frame_bgr[1], circles, CV_HOUGH_GRADIENT, 1, 14, 50, 15, 8, 12);
			//cout << circles.size() << endl;

			//Remove out of table circles
			int erase_count = 0;

			if (!circles.empty())
			{
				for (int i = circles.size() - 1; i >= 0; i--)
				{
					if (!(circles[i][0] > table_minmax_XY[0] &&
						circles[i][0] < table_minmax_XY[1] &&
						circles[i][1] > table_minmax_XY[2] &&
						circles[i][1] < table_minmax_XY[3]))
					{
						circles.erase(circles.begin() + i);
						//cout << circles.size() << endl;
					}
				}
			}

			vector<Mat3b> balls_bbs(circles.size());

			// Draw the circles detected
			for (size_t i = 0; i < circles.size(); i++)
			{

				Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
				int radius = cvRound(circles[i][2]);

				// Compute the bounding box
				Rect bbox(circles[i][0] - circles[i][2] + 4, circles[i][1] - circles[i][2] + 4, 2 * circles[i][2] - 4, 2 * circles[i][2] - 4);

				// Create a black image
				balls_bbs.insert(balls_bbs.begin() + i, frame(bbox));

				int n = identifyBallNumber(balls_bbs.at(i));
				putText(frame, std::to_string(n), cvPoint(cvRound(circles[i][0] - 20), cvRound(circles[i][1]) - 10),
					FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(200, 200, 250), 1, CV_AA);

				/*putText(frame, std::to_string(i), cvPoint(cvRound(circles[i][0] + 20), cvRound(circles[i][1]) + 10),
					FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(0, 0, 250), 1, CV_AA);*/

				// Show your result
				//imshow("Result", balls_bbs.at(i));

				/*cv::rectangle(
					frame,
					cv::Point(table_minmax_XY[0], table_minmax_XY[2]),
					cv::Point(table_minmax_XY[1], table_minmax_XY[3]),
					cv::Scalar(0, 255, 255), 3
				);*/

				// circle outline
				circle(frame, center, 11, Scalar(0, 255, 0), 2, 8, 0);
			}

			//show the current frame, the fg masks, and the change map
			//cout << "=========================" << endl;
			// Write the frame into the file 'out.avi'
			video.write(frame);
			imshow("Frame", frame);
			//imshow("Gray", frame_bgr[1]);
			//imshow("FG Mask MOG 2", frame);

			//get the input from the keyboard
			keyboard = (char)waitKey(30);
		}
	}
	waitKey(0);

	//delete capture object
	capture.release();
}


/*
Main function
*/
int main(int argc, char* argv[])
{
	//create GUI windows
	namedWindow("Frame");
	//namedWindow("FG Mask MOG 2");

	//Process the video
	processVideo("top15clip04.mp4");

	//destroy GUI windows
	destroyAllWindows();
	return EXIT_SUCCESS;
}

