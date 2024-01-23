/*

/*! \file Guassian.cpp
    \brief Display an image from a file, and perform blurring and using seperable filters
    \author Manushi

    This program reads an image from a specified file path and displays it in a window.
    Example of how to time an image processing task.

    Program takes a path to an image on the command line
*/

#include <cstdio> // a bunch of standard C/C++ functions like printf, scanf
#include <cstring> // C/C++ functions for working with strings
#include <cmath>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <chrono>
#include <sys/utime.h>

#include <filter.hpp>

// argc is # of command line parameters (including program name), argv is the array of strings
// This executable is expecting the name of an image on the command line.

int main(int argc, char* argv[]) {
    cv::Mat src; // define a Mat data type (matrix/image), allocates a header, image data is null
    cv::Mat dst; // cv::Mat to hold the output of the process
    cv::Mat dst1; // cv::Mat to hold the output of the process

    // usage: checking if the user provided a filename
    if (argc < 2) {
        printf("Usage %s <image filename>\n", argv[0]);
        exit(-1);
    }

    // read the image
    src = cv::imread(argv[1], IMREAD_COLOR); // allocating the image data
    // test if the read was successful
    if (src.data == NULL) {  // src.data is the reference to the image data
        printf("Unable to read image %s\n", argv[1]);
        exit(-1);
    }


    namedWindow("Original Color", WINDOW_AUTOSIZE); // Create a window for display.
    imshow("Original Color", src); // Show our image inside it.

    const int Ntimes = 10;
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    //////////////////////////////
    // set up the timing for version 1
    auto t1 = high_resolution_clock::now();

    // execute the file on the original image a couple of times
    for (int i = 0; i < Ntimes; i++) {
        blur5x5_1(src, dst);

        //namedWindow("Guassian 1 ", 2); // Create a window for display.
        //imshow("Guassian 1", dst); // Show our image inside it.
    }

    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1)/ Ntimes;

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = (t2 - t1)/ Ntimes;

    std::cout << "at() method blurring took " << ms_int.count() << "ms\n";
    std::cout << "at() method blurring took " << ms_double.count() << "ms\n";

    //////////////////////////////
    // set up the timing for version 2
    t1 = high_resolution_clock::now();

    //// execute the file on the original image a couple of times
    for (int i = 0; i < Ntimes; i++) {
        blur5x5_2(src, dst1);
        //namedWindow("Guassian 2 ", 3); // Create a window for display.
        //imshow("Guassian 2", dst1); // Show our image inside it.
    }

    t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    ms_int = duration_cast<milliseconds>(t2 - t1) / Ntimes;

    /* Getting number of milliseconds as a double. */
    ms_double = (t2 - t1) / Ntimes;

    std::cout << "seperable filters with pointer method blurring took " << ms_int.count() << "ms\n";
    std::cout << "seperable filters with pointer method blurring took " << ms_double.count() << "ms\n";

    // terminate the program
    printf("Terminating\n");
    waitKey(0); // Wait for a keystroke in the window
    destroyWindow("Original Color");
    /*destroyWindow("Guassian 1");
    destroyWindow("Guassian 2");*/

    return(0);
}
