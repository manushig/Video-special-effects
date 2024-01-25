/*! \file vidDisplay.cpp
    \brief Live Video Display from Camera with Grayscale Option.
    \author Manushi

    This program opens a video channel, captures and displays each frame. 
    Pressing 'g' toggles grayscale display mode on/off. 
    Pressing 'q' quits the program. 
    Pressing 's' saves the current frame with a timestamp.
    Pressing 'h' toggles alternative gray display mode on/off
    Pressing 'p' toggles sepia filter display mode on/off
    Pressing 'b' toggles blurr filter display mode on/off

*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <conio.h>
#include <direct.h> // For _getcwd on Windows

#include "filter.hpp" 
#include "faceDetect/faceDetect.h"

using namespace cv;
using namespace std;

/*!
 *  \brief Get current timestamp as a string.
 *  \return string A string representing the current date and time in the format YYYYMMDD_HHMMSS.
 *
 *  This function fetches the current system time and formats it into a string that can be used in filenames.
 */
string getCurrentTimestamp() {
    auto now = chrono::system_clock::now();
    auto now_c = chrono::system_clock::to_time_t(now);
    tm now_tm;
    localtime_s(&now_tm, &now_c);
    stringstream ss;
    ss << put_time(&now_tm, "%Y%m%d_%H%M%S");
    return ss.str();
}

string getCurrentPath() {
    char buffer[FILENAME_MAX];
    _getcwd(buffer, FILENAME_MAX);
    return string(buffer);
}

bool showChangeBrightness = false;
bool showChangeContrast = false;
int brightness = 0;
float contrast = 0.0;

// Callback function for mouse events
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_MOUSEWHEEL) {
        if (flags > 0) { // Scroll up
            if (showChangeBrightness) {
                brightness += 10; // Increase brightness
            }
            else if (showChangeContrast) {
                contrast += 10.0;  // Increase Contrast
            }
        }
        else if (flags < 0) { // Scroll down
            if (showChangeBrightness) {
                brightness -= 10; // Decrease brightness
            }
            else if (showChangeContrast) {
                contrast -= 10.0;  // Decrease Contrast
            }
            
        }
    }
}



/*!
 *  \brief Main function for capturing and displaying video from a camera.
 *
 *  \param argc Count of command-line arguments.
 *  \param argv Array of command-line arguments.
 *  \return int Returns 0 on successful execution, -1 on failure.
 *
 *  The main function initializes a video capture device, creates a window,
 *  and continuously captures frames from the device. It displays each frame
 *  and allows the user to quit the loop with 'q' or save a frame with 's'.
 */
int main(int argc, char* argv[]) {
    VideoCapture* capdev;

    // Open the video device
    capdev = new VideoCapture(0);
    if (!capdev->isOpened()) {
        cerr << "Unable to open video device" << endl;
        return -1;
    }

    // Get some properties of the image
    Size refS((int)capdev->get(CAP_PROP_FRAME_WIDTH),
        (int)capdev->get(CAP_PROP_FRAME_HEIGHT));
    std::cout << "Expected size: " << refS.width << " " << refS.height << endl;

    // Create a window
    namedWindow("Video Capture", 1);
    Mat frame, gray, alternateGray, sepiaFil, BlurrFil;
    bool showGray = false;
    bool showalternateGray = false;
    bool showsepia = false;
    bool showblurr = false;
    bool showsobelX = false;
    bool showsobelY = false;
    Mat SobelY;
    Mat SobelX;
    Mat grad_x;
    Mat grad_y;
    Mat SobelMagnitude;
    Mat grad_Magnitude;
    bool showmagnitude = false;
    Mat blurredQuantizedImage;
    bool showquantizing = false;
    Mat grey;
    vector<Rect> faces;
    Rect last(0, 0, 0, 0);
    bool showfaces = false;
    bool showblurrfaces = false;
    Mat blurredFrame;
    Mat frameWithSharpFaces;
    bool showembossing = false;
    Mat embossing;
    bool showcolorFace = false;
    Mat ColorFace;
    bool showHalo = false;
    Mat HaloFrame;
    //bool showMedianFilter = false;
    //Mat MedianFilterFrame;
    bool showRetainCertainColor = false;
    Mat RetainCertainColorFrame;
    //bool showChangeBrightness = false;
    Mat ChangeBrightnessFrame;
    //bool showChangeContrast = false;
    Mat ChangeContrastFrame;
    bool showCartooning = false;
    Mat CartooningFrame;
    bool showswapfaces = false;
    Mat swapFacesFrame;

    // Set the mouse callback function to capture scroll events
    cv::setMouseCallback("Video Capture", onMouse, nullptr);

    while (true) {
        *capdev >> frame; // Get a new frame from the camera
        if (frame.empty()) {
            cerr << "Frame is empty" << endl;
            break;
        }

        if (showalternateGray) {
            greyscale(frame, alternateGray);
            imshow("Video Capture", alternateGray);
        }
        else if (showsepia) {
            sepia(frame, sepiaFil);
            imshow("Video Capture", sepiaFil);
        }
        else if (showblurr) {
            blur5x5_2(frame, BlurrFil);
            imshow("Video Capture", BlurrFil);
        }
        else if (showblurr) {
            blur5x5_2(frame, BlurrFil);
            imshow("Video Capture", BlurrFil);
        }
        else if (showsobelX) {
            //Mat SobelX(frame.rows, frame.cols, CV_16SC3);
            SobelX.create(frame.size(), CV_16SC3);
            //SobelX.create(frame.size(), CV_8UC3);
            
            sobelX3x3(frame, SobelX);
            //imshow("Video Capture", SobelX);
            
            convertScaleAbs(SobelX, grad_x);
            imshow("Video Capture", grad_x);
        }
        else if (showsobelY) {
            //Mat SobelY(frame.rows, frame.cols, CV_16SC3);
            SobelY.create(frame.size(), CV_16SC3);
            //SobelY.create(frame.size(), CV_8UC3);
            
            sobelY3x3(frame, SobelY);
            //imshow("Video Capture", SobelY);

            convertScaleAbs(SobelY, grad_y);
            imshow("Video Capture", grad_y);
        }
        else if (showmagnitude) {
            SobelX.create(frame.size(), CV_16SC3);
            sobelX3x3(frame, SobelX);
            //convertScaleAbs(SobelX, grad_x);

            SobelY.create(frame.size(), CV_16SC3);
            sobelY3x3(frame, SobelY);
            //convertScaleAbs(SobelY, grad_y);

            SobelMagnitude.create(frame.size(), CV_8UC3);
            //magnitude(grad_x, grad_y, SobelMagnitude);
            magnitude(SobelX, SobelY, SobelMagnitude);

            convertScaleAbs(SobelMagnitude, grad_Magnitude);
            imshow("Video Capture", grad_Magnitude);
            //imshow("Video Capture", SobelMagnitude);
        }
        else if (showquantizing) {

            //Quantizing.create(frame.size(), CV_8UC3);
            blurQuantize(frame, blurredQuantizedImage, 10);
            imshow("Video Capture", blurredQuantizedImage);
        }
        else if (showfaces) {

            // convert the image to greyscale
            cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY, 0);

            // detect faces
            detectFaces(grey, faces);

            // draw boxes around the faces
            drawBoxes(frame, faces);

            // add a little smoothing by averaging the last two detections
            if (faces.size() > 0) {
                last.x = (faces[0].x + last.x) / 2;
                last.y = (faces[0].y + last.y) / 2;
                last.width = (faces[0].width + last.width) / 2;
                last.height = (faces[0].height + last.height) / 2;
            }

            imshow("Video Capture", frame);
        }
        else if (showblurrfaces) {
            // Convert the image to greyscale and detect faces
            cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY, 0);
            detectFaces(grey, faces);

            // Blur the entire frame
            cv::GaussianBlur(frame, blurredFrame, cv::Size(21, 21), 0);

            // Create a copy of the blurred frame to overlay sharp faces
            frameWithSharpFaces = blurredFrame.clone();

            // Overlay the sharp face areas from the original frame onto the blurred frame
            for (const auto& face : faces) {
                frame(face).copyTo(frameWithSharpFaces(face));
            }

            // display the frame with the sharp faces and blurred background
            cv::imshow("Video Capture", frameWithSharpFaces);
        }
        else if (showembossing) {
            SobelX.create(frame.size(), CV_16SC3);
            sobelX3x3(frame, SobelX);
            convertScaleAbs(SobelX, grad_x);

            SobelY.create(frame.size(), CV_16SC3);
            sobelY3x3(frame, SobelY);
            convertScaleAbs(SobelY, grad_y);

            embossing.create(frame.size(), CV_8UC3);
            createEmbossEffect(grad_x, grad_y, embossing);
            imshow("Video Capture", embossing);
        }
        else if (showcolorFace) {
            // convert the image to greyscale
            cvtColor(frame, grey, cv::COLOR_BGR2GRAY, 0);

            // detect faces
            detectFaces(grey, faces);

            ColorFace.create(frame.size(), CV_8UC3);
            colorFacesOnGray(frame, ColorFace, faces);
            imshow("Video Capture", ColorFace);
        }
        else if (showHalo) {
            // convert the image to greyscale
            cvtColor(frame, grey, cv::COLOR_BGR2GRAY, 0);

            // detect faces
            detectFaces(grey, faces);

            //embossing.create(frame.size(), CV_8UC3);
            addSparkleHalo(frame, faces);
            imshow("Video Capture", frame);
        }
        /*else if (showMedianFilter) {
            medianFilterPtrColor(frame, MedianFilterFrame, 3);
            imshow("Video Capture", MedianFilterFrame);
        }*/
        else if (showRetainCertainColor) {
            Scalar targetColor(76, 34, 135);
            colorPreserveGray(frame, RetainCertainColorFrame, targetColor, 50);
            imshow("Video Capture", RetainCertainColorFrame);
        }
        else if (showChangeBrightness) {
            
            adjustBrightness(frame, ChangeBrightnessFrame, brightness);
            imshow("Video Capture", ChangeBrightnessFrame);
        }
        else if (showChangeContrast) {

            adjustContrast(frame, ChangeContrastFrame, contrast);
            imshow("Video Capture", ChangeContrastFrame);
        }
        else if (showCartooning) {

            cartoonify(frame, CartooningFrame);
            imshow("Video Capture", CartooningFrame);
        }
        else if (showswapfaces) {
            // convert the image to greyscale
            cvtColor(frame, grey, cv::COLOR_BGR2GRAY, 0);

            // detect faces
            detectFaces(grey, faces);

            //swapFacesFrame.create(frame.size(), CV_8UC3);
            swapFaces(frame, swapFacesFrame, faces);
            imshow("Video Capture", swapFacesFrame);
        }
        else if (showGray) {
            /*
            This function converts an image from one color space to another. In this case, you'll convert the captured frame from BGR to greyscale.
             The conversion weights for each color channel (in BGR to grayscale conversion) are based on the perception of colors by the human eye.
             The OpenCV function cvtColor with COLOR_BGR2GRAY uses the following formula: Y = 0.299 * R + 0.587 * G + 0.114 * B
             Here, Y is the luminance component, and R, G, and B are the red, green, and blue color components of the image, respectively. 
             This formula reflects the fact that human eyes are more sensitive to green light, hence the higher weight for the green component.
             */
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            imshow("Video Capture", gray);
        }
        else {
            imshow("Video Capture", frame);
        }

        // Check for a waiting keystroke
        char key = (char)waitKey(10);
        switch (key) {
        case 'q':    // quit
            return 0;
        case 'g':    // go into grayscale
        {
            showGray = !showGray;
            showalternateGray = false;   // doesn't matter if previously alternate gray mode was enabled or not, disable alternate gray mode if gray mode is enabled
            showsepia = false;
            showblurr = false;
            showsobelY = false;
            showsobelX = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;
            //showMedianFilter = false;
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'h':    // alternative gray custom version
        {
            showalternateGray = !showalternateGray;
            showGray = false;           // doesn't matter if previously gray mode was enabled or not, disable gray mode if alternate gray mode is enabled
            showsepia = false;
            showblurr = false;
            showsobelY = false;
            showsobelX = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;
            //showMedianFilter = false;
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'p':    // sepia filter applied
        {
            showsepia = !showsepia;
            showGray = false;           // doesn't matter if previously gray mode was enabled or not, disable gray mode if alternate gray mode is enabled
            showalternateGray = false;           // doesn't matter if previously gray mode was enabled or not, disable gray mode if alternate gray mode is enabled
            showblurr = false;
            showsobelY = false;
            showsobelX = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;
            //showMedianFilter = false;
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'b':    // go into blurr mode
        {
            showblurr = !showblurr;
            showalternateGray = false;   // doesn't matter if previously alternate gray mode was enabled or not, disable alternate gray mode if gray mode is enabled
            showsepia = false;
            showGray = false;
            showsobelY = false;
            showsobelX = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;
            //showMedianFilter = false;
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'x':    // go into sobelX mode
        {
            showsobelX = !showsobelX;
            showalternateGray = false;   // doesn't matter if previously alternate gray mode was enabled or not, disable alternate gray mode if gray mode is enabled
            showsepia = false;
            showGray = false;
            showblurr = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;
            //showMedianFilter = false;
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'y':    // go into sobelY mode
        {
            showsobelY = !showsobelY;
            showalternateGray = false;
            showsepia = false;
            showGray = false;
            showblurr = false;
            showsobelX = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;
            //showMedianFilter = false;
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'm':    // go into sobel magnitude mode
        {
            showmagnitude = !showmagnitude;
            showalternateGray = false;
            showsepia = false;
            showGray = false;
            showblurr = false;
            showsobelX = false;
            showsobelY = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;
            //showMedianFilter = false;
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'i':    // go into blurr and quantizing mode
        {
            showquantizing = !showquantizing;
            showalternateGray = false;
            showsepia = false;
            showGray = false;
            showblurr = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;
            //showMedianFilter = false;
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'f':    // go into face detect mode
        {
            showfaces = !showfaces;
            showalternateGray = false;
            showsepia = false;
            showGray = false;
            showblurr = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;
            //showMedianFilter = false;
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'r':    // go to blurr face around the rectangle mode
        {
            showblurrfaces = !showblurrfaces;
            showalternateGray = false;
            showsepia = false;
            showGray = false;
            showblurr = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;
            //showMedianFilter = false;
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 't':    // go to embossing mode
        {
            showembossing = !showembossing;
            showalternateGray = false;
            showsepia = false;
            showGray = false;
            showblurr = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showcolorFace = false;
            showHalo = false;
            //showMedianFilter = false;
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'c':    // go to color face mode
        {
            showcolorFace = !showcolorFace;
            showalternateGray = false;
            showsepia = false;
            showGray = false;
            showblurr = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showHalo = false;
            //showMedianFilter = false;
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'u':    // go to halo effect mode
        {
            showHalo = !showHalo;
            showalternateGray = false;
            showsepia = false;
            showGray = false;
            showblurr = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            //showMedianFilter = false;
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        //case 'o':    // go to median filtering mode based on different kernel size
        //{
        //    showMedianFilter = !showMedianFilter;
        //    showalternateGray = false;
        //    showsepia = false;
        //    showGray = false;
        //    showblurr = false;
        //    showsobelX = false;
        //    showsobelY = false;
        //    showmagnitude = false;
        //    showquantizing = false;
        //    showfaces = false;
        //    showblurrfaces = false;
        //    showembossing = false;
        //    showcolorFace = false;
        //    showHalo = false;
        //    showRetainCertainColor = false;
        //    showChangeContrast = false;
        //    showChangeBrightness = false;
        //    break;
        //}
        case 'a':    // go to retain certain color mode
        {
            showRetainCertainColor = !showRetainCertainColor;
            showalternateGray = false;
            showsepia = false;
            showGray = false;
            showblurr = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;
            //showMedianFilter = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'd':    // go to change brightness mode
        {
            showChangeBrightness = !showChangeBrightness;
            showalternateGray = false;
            showsepia = false;
            showGray = false;
            showblurr = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;
            //showMedianFilter = false;
            showRetainCertainColor = false;
            showChangeContrast = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'e':    // go to change contrast mode
        {
            showChangeContrast = !showChangeContrast;
            showalternateGray = false;
            showsepia = false;
            showGray = false;
            showblurr = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;
            //showMedianFilter = false;
            showRetainCertainColor = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'j':    // go to cartooning mode
        {
            showCartooning = !showCartooning;
            showalternateGray = false;
            showsepia = false;
            showGray = false;
            showblurr = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;
            //showMedianFilter = false;
            showRetainCertainColor = false;
            showChangeBrightness = false;
            showChangeContrast = false;
            showswapfaces = false;
            break;
        }
        case 'k':    // go to swap faces mode
        {
            showswapfaces = !showswapfaces;
            showalternateGray = false;
            showsepia = false;
            showGray = false;
            showblurr = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;
            //showMedianFilter = false;
            showRetainCertainColor = false;
            showChangeBrightness = false;
            showChangeContrast = false;
            showCartooning = false;
            break;
        }
        case 's':     // save the screen capture
        {
            string timestamp = getCurrentTimestamp();
            string filename;
            if (showGray) {
                filename = "saved_frame_gray_" + timestamp + ".jpg";
                imwrite(filename, gray);
            }
            else if (showalternateGray) {
                filename = "saved_frame_alternative_gray_" + timestamp + ".jpg";
                imwrite(filename, alternateGray);
            }
            else if (showsepia) {
                filename = "saved_frame_sepia_" + timestamp + ".jpg";
                imwrite(filename, sepiaFil);
            }
            else if (showblurr) {
                filename = "saved_frame_blurr_" + timestamp + ".jpg";
                imwrite(filename, BlurrFil);
            }
            else if (showsobelX) {
                filename = "saved_frame_sobelX_" + timestamp + ".jpg";
                imwrite(filename, grad_x);
            }
            else if (showsobelY) {
                filename = "saved_frame_sobelY_" + timestamp + ".jpg";
                imwrite(filename, grad_y);
            }
            else if (showmagnitude) {
                filename = "saved_frame_sobel_magnitude_" + timestamp + ".jpg";
                imwrite(filename, SobelMagnitude);
            }
            else if (showquantizing) {
                filename = "saved_frame_blurr_quantizing_" + timestamp + ".jpg";
                imwrite(filename, blurredQuantizedImage);
            }
            else if (showfaces) {
                filename = "saved_frame_facedetect_" + timestamp + ".jpg";
                imwrite(filename, frame);
            }
            else if (showblurrfaces) {
                filename = "saved_frame_blurr_facedetect_" + timestamp + ".jpg";
                imwrite(filename, frameWithSharpFaces);
            }
            else if (showembossing) {
                filename = "saved_frame_embossing_" + timestamp + ".jpg";
                imwrite(filename, embossing);
            }
            else if (showcolorFace) {
                filename = "saved_frame_colorFace_" + timestamp + ".jpg";
                imwrite(filename, ColorFace);
            }
            else if (showHalo) {
                filename = "saved_frame_Haloreffect_" + timestamp + ".jpg";
                imwrite(filename, frame);
            }
            /*else if (showMedianFilter) {
                filename = "saved_frame_MedianFilteringEffect_" + timestamp + ".jpg";
                imwrite(filename, MedianFilterFrame);
            }*/
            else if (showRetainCertainColor) {
                filename = "saved_frame_RetainColor_" + timestamp + ".jpg";
                imwrite(filename, RetainCertainColorFrame);
            }
            else if (showChangeBrightness) {
                filename = "saved_frame_changeBrightness_" + timestamp + ".jpg";
                imwrite(filename, ChangeBrightnessFrame);
            }
            else if (showChangeContrast) {
                filename = "saved_frame_changeContrast_" + timestamp + ".jpg";
                imwrite(filename, ChangeContrastFrame);
            }
            else if (showCartooning) {
                filename = "saved_frame_cartooning_" + timestamp + ".jpg";
                imwrite(filename, CartooningFrame);
            }
            else if (showswapfaces) {
                filename = "saved_frame_swapFaces_" + timestamp + ".jpg";
                imwrite(filename, swapFacesFrame);
            }
            else {
                filename = "saved_frame_color_" + timestamp + ".jpg";
                imwrite(filename, frame);
            }
            string fullPath = getCurrentPath() + "\\" + filename;
            cout << "Frame saved as: " << fullPath << endl;
            break;
        }

        // no default since we want it to keep looping and waiting for key strokes
        }
    }

    delete capdev;
    return 0;
}


