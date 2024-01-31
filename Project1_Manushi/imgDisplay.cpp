/*! \file imgDisplay.cpp
    \brief Display an image from a file.
    \author Manushi
    \date January 24, 2024

    This program reads an image from a specified file path and displays it in a window.
    The program enters a loop, waiting for the user to press 'q' to quit the program.
    Additional functionality includes applying sepia, Sobel X, Sobel Y, and median filtering
    to the image and displaying them in separate windows.
    The windows are destroyed upon exiting.
*/

#include <opencv2/opencv.hpp>
#include <iostream>

#include "filters.hpp" 

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
 *  It then reads the image, displays it in a window, and applies various filters.
 *  The program waits for the user to press 'q' to exit.
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
    Mat sobelY;
    Mat sobelX;
    Mat MedianFilterFrame;

    // Apply sepia filter
    sepia(image, sepiaFil);
    namedWindow("Sepia", WINDOW_AUTOSIZE);
    imshow("Sepia", sepiaFil);

    // Apply Sobel X filter
    sobelX.create(image.size(), CV_8UC3);
    sobelX3x3(image, sobelX);
    namedWindow("Sobel X", WINDOW_AUTOSIZE);
    imshow("Sobel X", sobelX);

    // Apply Sobel Y filter
    sobelY.create(image.size(), CV_8UC3);
    sobelY3x3(image, sobelY);
    namedWindow("Sobel Y", WINDOW_AUTOSIZE);
    imshow("Sobel Y", sobelY);

    // Apply Median Filter
    medianFilterPtrColor(image, MedianFilterFrame, 3);
    namedWindow("Median Filtering", WINDOW_AUTOSIZE);
    imshow("Median Filtering", MedianFilterFrame);

    // Display original image
    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", image);

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
