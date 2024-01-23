
/*! \file imgDisplay.cpp
    \brief Display an image from a file.
    \author Manushi

    This program reads an image from a specified file path and displays it in a window.
    The program enters a loop, waiting for the user to press 'q' to quit the program.
    The window is destroyed upon exiting.
*/

#include <opencv2/opencv.hpp>
#include <iostream>

#include "filter.hpp" 

using namespace cv;
using namespace std;

/*!
 *  \brief Main function.
 *
 *  \param argc Count of command-line arguments.
 *  \param argv Array of command-line arguments.
 *  \return int Returns 0 on successful execution, -1 on failure.
 *
 *  The main function checks if an image file path is provided as an argument.
 *  It then reads the image, displays it in a window, performs sepia filtering on image and siaplys in new window and waits for the user to press 'q' to exit. 
 * 
 */
int main(int argc, char* argv[]) {
    // Check if the image file name is provided
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <Image_Path>" << endl;
        return -1;
    }

    // Read the image file
    Mat image = imread(argv[1], IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Could not read the image: " << argv[1] << endl;
        return -1;
    }

    Mat sepiaFil;

    // Create a window for display
    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", image);

    sepia(image, sepiaFil);
    namedWindow("Sepia", WINDOW_AUTOSIZE);
    imshow("Sepia", sepiaFil);

    Mat sobelY; // (image.rows, image.cols, CV_16SC1);
    sobelY.create(image.size(), CV_8UC3);

    sobelY3x3(image, sobelY);
    namedWindow("sobel y", WINDOW_AUTOSIZE);
    imshow("sobel y", sobelY);

    Mat sobelX; // (image.rows, image.cols, CV_16SC1);
    sobelX.create(image.size(), CV_8UC3);

    sobelX3x3(image, sobelX);
    namedWindow("sobel x", WINDOW_AUTOSIZE);
    imshow("sobel x", sobelX);

    Mat MedianFilterFrame;
    medianFilterPtrColor(image, MedianFilterFrame, 3);
    imshow("Median Filtering", MedianFilterFrame);

    //Mat blurr_gauss;
    //cv::GaussianBlur(image, blurr_gauss, cv::Size(5, 5), 0);
    //imshow("GaussianBlur", blurr_gauss);

    //Mat blurr_gauss_own;
    //blur5x5_2(image, blurr_gauss_own);
    //imshow("blur5x5_2", blurr_gauss_own);

    // Wait for a keystroke in the window
    cout << "Press 'q' to exit" << endl;
    for (;;) {
        char key = (char)waitKey(10);
        if (key == 'q') {
            break;
        }
        // Add more keypress functionalities here if needed
    }
    
    // Destroy the created window
    destroyWindow("Display window");

    return 0;
}
