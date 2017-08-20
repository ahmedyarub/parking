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
string parked_from;
string parked_to;
string img_file;

string get_date_time() {
	time_t rawtime;
	struct tm * timeinfo = new tm();
	char buffer[80];

	time(&rawtime);
	localtime_s(timeinfo, &rawtime);

	strftime(buffer, 80, "%G%m%d%I%M%S", timeinfo);

	return buffer;
}

string get_time() {
	time_t rawtime;
	struct tm * timeinfo = new tm();
	char buffer[80];

	time(&rawtime);
	localtime_s(timeinfo, &rawtime);

	strftime(buffer, 80, "%I:%M:%S", timeinfo);

	return buffer;
}

int main(int argc, const char** argv)
{


	ofstream outputFile(get_date_time() + ".txt");

	outputFile << "Image File\tParking Time\tLeaving Time\tSeconds Parked" << endl;
	outputFile.flush();

	bool car_fined = false;

	time_t parked_at;
	time(&parked_at);

	namedWindow("frame", 1);
	parking_time = 5;
	createTrackbar("Parking time", "frame", &parking_time, 60, NULL);

	// Init background substractor
	Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();

	// Create empy input img, foreground and background image and foreground mask.
	Mat img, frame, foregroundMask, backgroundImage, foregroundImg;

	// capture video from source 0, which is web camera, If you want capture video from file just replace //by  VideoCapture cap("videoFile.mov")
	//VideoCapture cap("parking4.mp4");
	VideoCapture cap(0);
	cap.set(CAP_PROP_AUTO_EXPOSURE, 0);
	cap.set(CAP_PROP_AUTOFOCUS, 0);

	// main loop to grab sequence of input files
	for (;;) {

		if (cap.grab())
		{
			cap.retrieve(frame, CV_CAP_OPENNI_GRAY_IMAGE);
			imshow("frame", frame);
		}

		if (cap.grab())
		{
			cap.retrieve(frame, CV_CAP_OPENNI_GRAY_IMAGE);
			imshow("frame", frame);
		}

		bool ok = cap.grab();
		if (ok)
		{
			cap.retrieve(frame, CV_CAP_OPENNI_GRAY_IMAGE);
			imshow("frame", frame);
		}

		if (ok == false) {
			std::cout << "Video Capture Fail" << std::endl;
		}
		else {
			// obtain input image from source
			cap.retrieve(frame, CV_CAP_OPENNI_GRAY_IMAGE);
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
			threshold(foregroundMask, foregroundMask, 200, 255, THRESH_BINARY);

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

#ifdef WIN32
			std::system("cls");
#else
			//std::system ("clear");
#endif

			if (mask_percent < 0.75)
			{
				if (car_fined)
				{
					outputFile << img_file << '\t' << parked_from << '\t' << get_time() << '\t' << parking_time << endl;
					outputFile.flush();
				}

				car_fined = false;

				time(&parked_at);
				parked_from = get_time();
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

					int parked_for = (int)difftime(cur_time, parked_at);

					if (parked_for >= parking_time)
					{
						car_fined = true;

						//imshow("Fined car", frame);

						img_file = get_date_time() + ".jpg", frame;
						imwrite(img_file, frame);
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
			//imshow("frame", frame);
			imshow("foreground image", foregroundImg);

			//int key6 = waitKey(40);

			if (!backgroundImage.empty()) {

				//imshow("mean background image", backgroundImage);
				int key5 = waitKey(40);

			}


		}

	}

	return EXIT_SUCCESS;
}
