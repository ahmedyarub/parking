#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/text.hpp"
#include "opencv2/video/background_segm.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>
#include <ctime>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::text;
using namespace std;

int parking_time;

void ocr(Mat frame);

int main(int argc, const char** argv)
{
	bool car_fined = false;

	time_t parked_at;
	time(&parked_at);

	namedWindow("frame", 1);
	parking_time = 5;
	createTrackbar("Parking time", "frame", &parking_time, 20, NULL);

	// Init background substractor
	Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();

	// Create empy input img, foreground and background image and foreground mask.
	Mat img, frame, foregroundMask, backgroundImage, foregroundImg;

	// capture video from source 0, which is web camera, If you want capture video from file just replace //by  VideoCapture cap("videoFile.mov")
	VideoCapture cap("parking4.mp4");
	//VideoCapture cap(0);

	// main loop to grab sequence of input files
	for (;;) {
		cap.grab();
		cap.grab();
		bool ok = cap.grab();

		if (ok == false) {
			std::cout << "Video Capture Fail" << std::endl;
		}
		else {
			// obtain input image from source
			cap.retrieve(frame, CV_CAP_OPENNI_BGR_IMAGE);
			img = frame(Rect(frame.cols / 3, frame.rows / 2, frame.cols / 3, frame.rows / 2));
			// Just resize input image if you want
			//resize(img, img, Size(640, 480));

			// create foreground mask of proper size
			if (foregroundMask.empty()) {
				foregroundMask.create(img.size(), img.type());
			}

			// compute foreground mask 8 bit image
			// -1 is parameter that chose automatically your learning rate

			bg_model->apply(img, foregroundMask, false ? -1 : 0);

			// smooth the mask to reduce noise in image
			GaussianBlur(foregroundMask, foregroundMask, Size(11, 11), 3.5, 3.5);

			// threshold mask to saturate at black and white values
			threshold(foregroundMask, foregroundMask, 10, 255, THRESH_BINARY);

			float count_white = 0;
			for (int y = 0; y < foregroundMask.rows; y++) {
				for (int x = 0; x < foregroundMask.cols; x++) {
					if (foregroundMask.at<uchar>(y, x) != 0) {
						if (foregroundMask.at<uchar>(y, x) == 255) {
							count_white++;
						}
					}
				}
			}

			float mask_percent = count_white / (foregroundMask.rows * foregroundMask.cols);

			system("CLS");
			if (mask_percent < 0.75)
			{
				car_fined = false;

				time(&parked_at);
				cout << "No car parked";
			}
			else {
				if (car_fined)
				{
					cout << "Car fined!";
				}
				else {
					time_t cur_time;
					time(&cur_time);

					int parked_for = difftime(cur_time, parked_at);

					if (parked_for >= parking_time)
					{
						car_fined = true;

						imshow("Fined car", foregroundImg);
						//ocr(foregroundImg);
					}

					cout << "Car parked for: " << parked_for << " seconds";
				}
			}

			// create black foreground image
			foregroundImg = Scalar::all(0);
			// Copy source image to foreground image only in area with white mask
			img.copyTo(foregroundImg, foregroundMask);

			//Get background image
			bg_model->getBackgroundImage(backgroundImage);

			// Show the results
			//imshow("foreground mask", foregroundMask);
			imshow("frame", frame);
			//imshow("foreground image", foregroundImg);

			//int key6 = waitKey(40);

			if (!backgroundImage.empty()) {

				//imshow("mean background image", backgroundImage);
				int key5 = waitKey(40);

			}


		}

	}

	return EXIT_SUCCESS;
}


void ocr(Mat frame)
{
	// For testing
	Mat img_template = imread("c:\\template.png", IMREAD_GRAYSCALE);
	Mat img_input;
	cvtColor(frame, img_input, cv::COLOR_RGB2GRAY);

	if (!img_template.data || !img_input.data)
	{
		cout << "No data in image!";
	}

	// Compute features
	int minHessian = 400;
	Ptr<SURF> detector = SURF::create(minHessian);
	std::vector<KeyPoint> keypoints_object, keypoints_scene;
	Mat descriptors_object, descriptors_scene;
	detector->detectAndCompute(img_template, Mat(), keypoints_object,
		descriptors_object);
	detector->detectAndCompute(img_input, Mat(), keypoints_scene,
		descriptors_scene);

	// Match features
	FlannBasedMatcher matcher;
	std::vector<DMatch> matches;
	matcher.match(descriptors_object, descriptors_scene, matches);

	double max_dist = 0.01;
	double min_dist = 100;

	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	//Filter features
	std::vector<DMatch> good_matches;
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matches[i].distance  < 3 * min_dist) //3*min_dist )
		{
			good_matches.push_back(matches[i]);
			obj.push_back(keypoints_object[matches[i].queryIdx].pt);
			scene.push_back(keypoints_scene[matches[i].trainIdx].pt);
		}
	}

	Mat img_matches;
	drawMatches(img_input, keypoints_object, img_template, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	Mat H = findHomography(obj, scene, RANSAC);

	if (H.empty())
	{
		imshow("Fined car",frame);

		return;
	}

	std::vector<Point2f> obj_corners(4);

	cv::Mat matched;
	warpPerspective(img_input, matched, H, img_template.size(), WARP_INVERSE_MAP);

	imshow("Extracted", matched);

	return;
}
