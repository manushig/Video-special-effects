/*! \file vidDisplay.cpp
    \brief Advanced Video Display and Processing from Camera.
    \author Manushi
    \date January 24, 2024

    This program opens a video channel to capture and display live video from a camera.
    It includes multiple functionalities, each toggled by specific keypresses:
    - 'g': Toggles grayscale display mode on/off.
    - 'h': Toggles alternative grayscale display mode on/off.
    - 'p': Toggles sepia filter display mode on/off.
    - 'v': Toggles sepia filter with vignett display mode on/off.
    - 'b': Toggles blur filter using seperable matrix, display mode on/off.
    - 'z': Toggles blur filter using guassian filter, display mode on/off.
    - 'x' and 'y': Toggle Sobel filter in X and Y directions respectively.
    - 'm': Toggles gradient magnitude mode (Sobel) on/off.
    - 'i': Toggles blur and quantize mode on/off.
    - 'f': Toggles face detection mode on/off.
    - 'r': Toggles blurred faces mode on/off.
    - 't': Toggles embossing effect mode on/off.
    - 'c': Toggles color face mode on/off.
    - 'u': Toggles halo effect mode on/off.
    - 'a': Toggles retain certain color mode on/off.
    - 'd': Toggles brightness change mode on/off.
    - 'e': Toggles contrast change mode on/off.
    - 'j': Toggles cartooning mode on/off.
    - 'k': Toggles swap faces mode on/off.
    - 's': Saves the current frame with a timestamp.
    - 'q': Quits the program.

    These functionalities allow for a wide range of video processing and display options,
    making the program versatile for various image processing demonstrations.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <conio.h>
#include <direct.h>

#include "filters.hpp" 
#include "faceDetect/faceDetect.h"

using namespace cv;
using namespace std;

/*!
 * \brief Fetches the current system time and formats it into a string.
 * \return A string representing the current date and time in the format YYYYMMDD_HHMMSS.
 *
 * This function is useful for creating timestamped filenames.
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


/*!
 * \brief Retrieves the current working directory.
 * \return A string representing the current working directory.
 *
 * This function is useful for obtaining the directory path for file operations.
 */
string getCurrentPath() {
    char buffer[FILENAME_MAX];
    _getcwd(buffer, FILENAME_MAX);
    return string(buffer);
}

// Global variables to track the state of brightness and contrast adjustments
bool showChangeBrightness = false;
bool showChangeContrast = false;
int brightness = 0;
float contrast = 0.0;
bool time_blur1_next_frame = false;
bool time_blur2_next_frame = false;

/*!
 * \brief Callback function for handling mouse events.
 * \param event The type of mouse event.
 * \param x The x-coordinate of the mouse event.
 * \param y The y-coordinate of the mouse event.
 * \param flags Additional flags for the event.
 * \param userdata Pointer to user data (unused).
 *
 * This function responds to mouse wheel movements to adjust brightness and contrast.
 */
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
  * \brief Main function for the video display application.
  *
  * This function captures live video from a camera, processes it based on user input,
  * and displays the result. It allows toggling between various display modes like grayscale,
  * sepia, blur, Sobel filters, etc., and handles keypresses for different functionalities.
  *
  * \param argc Argument count.
  * \param argv Argument vector.
  * \return int Returns 0 on successful execution, -1 on failure.
  */
int main(int argc, char* argv[]) {
    VideoCapture* capdev;

    // Initialize the video capture device
    capdev = new VideoCapture(0);
    if (!capdev->isOpened()) {
        cerr << "Unable to open video device" << endl;
        return -1;
    }

    // Retrieve and display the frame dimensions
    Size refS((int)capdev->get(CAP_PROP_FRAME_WIDTH),
        (int)capdev->get(CAP_PROP_FRAME_HEIGHT));
    std::cout << "Expected size: " << refS.width << " " << refS.height << endl;

    // Create a window for video capture
    namedWindow("Video Capture", 1);

    // Variables for different modes and filters
    Mat frame, gray, alternateGray, sepiaFil, sepiaVignettFill, Blurr1Fil, Blurr1Fil1, Blurr1Fil2, Blurr2Fil, Blurr2Fil1, Blurr2Fil2;
    bool showGray = false;
    bool showalternateGray = false;
    bool showsepia = false;
    bool showsepiaVignett = false;
    bool showblurr1 = false;
    bool showblurr2 = false;
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
    bool showRetainCertainColor = false;
    Mat RetainCertainColorFrame;
    Mat ChangeBrightnessFrame;
    Mat ChangeContrastFrame;
    bool showCartooning = false;
    Mat CartooningFrame;
    bool showswapfaces = false;
    Mat swapFacesFrame;

    // Set callback for mouse events
    cv::setMouseCallback("Video Capture", onMouse, nullptr);

    // Main loop for capturing and processing frames
    while (true) {
        *capdev >> frame; // Capture a new frame
        if (frame.empty()) {
            cerr << "Frame is empty" << endl;
            break; // Exit the loop if no frame is captured
        }

        // Process the frame based on the current mode
        if (showalternateGray) {
            // Apply custom grayscale processing to the frame
            greyscale(frame, alternateGray);
            imshow("Video Capture", alternateGray);
        }
        else if (showsepia) {
            // Apply sepia filter to the frame
            sepia(frame, sepiaFil);
            imshow("Video Capture", sepiaFil);
        }
        else if (showsepiaVignett) {
            // Apply sepia filter with vignett to the frame
            sepiawithvignett(frame, sepiaVignettFill);
            imshow("Video Capture", sepiaVignettFill);
        }
        else if (showblurr1) {
            if (time_blur1_next_frame)
            {
                auto start = std::chrono::high_resolution_clock::now();
                blur5x5_1(frame, Blurr1Fil); // Perform the blur operation
                blur5x5_1(Blurr1Fil, Blurr1Fil1);
                blur5x5_1(Blurr1Fil1, Blurr1Fil2);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> blur1_time = end - start;
                std::cout << "Blur using Guassian Filter operation took " << blur1_time.count() / 3 << " milliseconds.\n";
                imshow("Video Capture", Blurr1Fil);
                time_blur1_next_frame = false; // Reset the flag
            }
            else
            {
                blur5x5_1(frame, Blurr1Fil); // Perform the blur operation
                blur5x5_1(Blurr1Fil, Blurr1Fil1);
                blur5x5_1(Blurr1Fil1, Blurr1Fil2);
                imshow("Video Capture", Blurr1Fil2);
            }
        }
        else if (showblurr2) {
            if (time_blur2_next_frame)
            {
                auto start = std::chrono::high_resolution_clock::now();
                blur5x5_2(frame, Blurr2Fil); // Perform the blur operation
                blur5x5_2(Blurr2Fil, Blurr2Fil1);
                blur5x5_2(Blurr2Fil1, Blurr2Fil2);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> blur2_time = end - start;
                std::cout << "Blur using Seperable Matrix operation took " << blur2_time.count() / 3 << " milliseconds.\n";
                imshow("Video Capture", Blurr2Fil2);
                time_blur2_next_frame = false;
            }
            else
            {
                blur5x5_2(frame, Blurr2Fil); // Perform the blur operation
                blur5x5_2(Blurr2Fil, Blurr2Fil1);
                blur5x5_2(Blurr2Fil1, Blurr2Fil2);
                imshow("Video Capture", Blurr2Fil2);
            }
        }
        else if (showsobelX) {
            // Apply Sobel filter in X direction
            SobelX.create(frame.size(), CV_16SC3);      
            sobelX3x3(frame, SobelX);
            
            convertScaleAbs(SobelX, grad_x); // Convert to absolute for display
            imshow("Video Capture", grad_x);
        }
        else if (showsobelY) {
            // Apply Sobel filter in Y direction
            SobelY.create(frame.size(), CV_16SC3);                        
            sobelY3x3(frame, SobelY);            
            
            convertScaleAbs(SobelY, grad_y); // Convert to absolute for display
            imshow("Video Capture", grad_y);
        }
        else if (showmagnitude) {
            // Calculate and display gradient magnitude using Sobel filters
            SobelX.create(frame.size(), CV_16SC3);
            sobelX3x3(frame, SobelX);            
           
            SobelY.create(frame.size(), CV_16SC3);
            sobelY3x3(frame, SobelY);
            
            SobelMagnitude.create(frame.size(), CV_8UC3);            
            magnitude(SobelX, SobelY, SobelMagnitude);
            
            convertScaleAbs(SobelMagnitude, grad_Magnitude);
            imshow("Video Capture", grad_Magnitude);            
        }
        else if (showquantizing) {
            // Apply blur and quantize effect
            blurQuantize(frame, blurredQuantizedImage, 10);
            imshow("Video Capture", blurredQuantizedImage);
        }
        else if (showfaces) {
            // Detect and display faces with bounding boxes
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
            // Blur the background while keeping faces sharp
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

            // Display the frame with the sharp faces and blurred background
            cv::imshow("Video Capture", frameWithSharpFaces);
        }
        else if (showembossing) {
            // Applying embossing effect            
            SobelX.create(frame.size(), CV_16SC3);
            sobelX3x3(frame, SobelX); // Apply Sobel filter in X direction
            convertScaleAbs(SobelX, grad_x);

            SobelY.create(frame.size(), CV_16SC3);
            sobelY3x3(frame, SobelY); // Apply Sobel filter in Y direction
            convertScaleAbs(SobelY, grad_y);

            embossing.create(frame.size(), CV_8UC3);
            createEmbossEffect(grad_x, grad_y, embossing); // Create emboss effect
            imshow("Video Capture", embossing);
        }
        else if (showcolorFace) {
            // Coloring detected faces while keeping the background grayscale
            cvtColor(frame, grey, cv::COLOR_BGR2GRAY, 0);

            detectFaces(grey, faces); // Detect faces in the grayscale image

            ColorFace.create(frame.size(), CV_8UC3);
            colorFacesOnGray(frame, ColorFace, faces); // Apply color to detected faces
            imshow("Video Capture", ColorFace);
        }
        else if (showHalo) {
            // Adding halo effect around detected faces
            cvtColor(frame, grey, cv::COLOR_BGR2GRAY, 0); // Convert frame to grayscale
            detectFaces(grey, faces); // Detect faces in the grayscale image
            addSparkleHalo(frame, faces); // Add halo effect around detected faces
            imshow("Video Capture", frame); // Display the resulting image
        }
        else if (showRetainCertainColor) {
            // Retain a specific color in the frame, making the rest grayscale
            //Scalar targetColor(28, 36, 36);
            Scalar targetColor(18, 17, 74);
            colorPreserveGray(frame, RetainCertainColorFrame, targetColor, 50);
            imshow("Video Capture", RetainCertainColorFrame);
        }
        else if (showChangeBrightness) {
            // Adjust the brightness of the frame
            adjustBrightness(frame, ChangeBrightnessFrame, brightness);
            imshow("Video Capture", ChangeBrightnessFrame);
        }
        else if (showChangeContrast) {
            // Adjust the contrast of the frame
            adjustContrast(frame, ChangeContrastFrame, contrast);
            imshow("Video Capture", ChangeContrastFrame);
        }
        else if (showCartooning) {
            // Apply cartooning effect to the frame
            cartoonify(frame, CartooningFrame);
            imshow("Video Capture", CartooningFrame);
        }
        else if (showswapfaces) {
            // Swap faces detected in the frame            
            cvtColor(frame, grey, cv::COLOR_BGR2GRAY, 0); // Convert frame to grayscale
      
            detectFaces(grey, faces); // Detect faces in the grayscale image

            //swapFacesFrame.create(frame.size(), CV_8UC3);
            swapFaces(frame, swapFacesFrame, faces); // Swap detected faces
            imshow("Video Capture", swapFacesFrame); // Display the image with swapped faces
        }
        else if (showGray) {
            // Convert the frame to grayscale and display it
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            imshow("Video Capture", gray);
        }
        else {
            // Default display of the captured frame
            imshow("Video Capture", frame);
        }

        // Handling keystrokes for toggling modes
        char key = (char)waitKey(10);
        
        switch (key) {
        case 'q':    // Quit the program
            return 0;
        case 'g':    // Toggle grayscale mode
        {
            showGray = !showGray;
            // Reset other modes to ensure only grayscale mode is active
            showalternateGray = false;
            showsepia = false; showsepiaVignett = false;
            showblurr1 = false; showblurr2 = false;
            showsobelY = false;
            showsobelX = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;            
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'h':    // Toggle alternative grayscale mode
        {
            showalternateGray = !showalternateGray;
            // Ensuring that no other modes are active
            showGray = false;
            showsepia = false; 
            showsepiaVignett = false;
            showblurr1 = false; 
            showblurr2 = false;
            showsobelY = false;
            showsobelX = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;            
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'p':    // Toggle sepia filter mode
        {
            showsepia = !showsepia;
            // Ensuring that no other modes are active
            showsepiaVignett = false;
            showGray = false;           
            showalternateGray = false;  
            showblurr1 = false; showblurr2 = false;
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
        case 'v':    // Toggle sepia filter with vignett mode
        {
            showsepiaVignett = !showsepiaVignett;
            // Ensuring that no other modes are active
            showsepia = false;
            showGray = false;
            showalternateGray = false;
            showblurr1 = false; showblurr2 = false;
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
        case 'z':    // Toggle blur 1 mode
        {
            showblurr1 = !showblurr1;
            if (showblurr1) {
                time_blur1_next_frame = true; // Set the flag to time next frame
            }
            // Ensuring that no other modes are active
            showblurr2 = false;
            showalternateGray = false;
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
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'b':    // Toggle blur 2 mode
        {
            showblurr2 = !showblurr2;
            if (showblurr2) {
                time_blur2_next_frame = true; // Set the flag to time next frame
            }
            // Ensuring that no other modes are active
            showblurr1 = false;
            showalternateGray = false;
            showsepia = false; showsepiaVignett = false;
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
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'x':    // Toggle Sobel filter in X direction
        {
            showsobelX = !showsobelX;
            // Ensuring that no other modes are active
            showalternateGray = false;
            showsepia = false; showsepiaVignett = false;
            showGray = false;
            showblurr1 = false; showblurr2 = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;            
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'y':    // Toggle Sobel filter in Y direction     
        {
            showsobelY = !showsobelY;
            // Ensuring that no other modes are active
            showalternateGray = false;
            showsepia = false; showsepiaVignett = false;
            showGray = false;
            showblurr1 = false; showblurr2 = false;
            showsobelX = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;            
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'm':    // Toggle gradient magnitude mode (Sobel)
        {
            showmagnitude = !showmagnitude;
            // Ensuring that no other modes are active
            showalternateGray = false;
            showsepia = false; showsepiaVignett = false;
            showGray = false;
            showblurr1 = false; showblurr2 = false;
            showsobelX = false;
            showsobelY = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;            
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'i':    // Toggle blur and quantizing mode
        {
            showquantizing = !showquantizing;
            // Ensuring that no other modes are active
            showalternateGray = false;
            showsepia = false; showsepiaVignett = false;
            showGray = false;
            showblurr1 = false; showblurr2 = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;            
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'f':    // Toggle face detection mode
        {
            showfaces = !showfaces;
            // Ensuring that no other modes are active
            showalternateGray = false;
            showsepia = false; showsepiaVignett = false;
            showGray = false;
            showblurr1 = false; showblurr2 = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;            
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'r':    //Toggle blurred faces mode
        {
            showblurrfaces = !showblurrfaces;
            // Ensuring that no other modes are active
            showalternateGray = false;
            showsepia = false; showsepiaVignett = false;
            showGray = false;
            showblurr1 = false; showblurr2 = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;            
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 't':    // Toggle embossing mode
        {
            showembossing = !showembossing;
            // Ensuring that no other modes are active
            showalternateGray = false;
            showsepia = false; showsepiaVignett = false;
            showGray = false;
            showblurr1 = false; showblurr2 = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showcolorFace = false;
            showHalo = false;            
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'c':    // Toggle color face mode
        {
            showcolorFace = !showcolorFace;
            // Ensuring that no other modes are active
            showalternateGray = false;
            showsepia = false; showsepiaVignett = false;
            showGray = false;
            showblurr1 = false; showblurr2 = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showHalo = false;            
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'u':    // Toggle halo effect mode
        {
            showHalo = !showHalo;
            // Ensuring that no other modes are active
            showalternateGray = false;
            showsepia = false; showsepiaVignett = false;
            showGray = false;
            showblurr1 = false; showblurr2 = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;            
            showRetainCertainColor = false;
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'a':    // Toggle retain certain color mode
        {
            showRetainCertainColor = !showRetainCertainColor;
            // Ensuring that no other modes are active
            showalternateGray = false;
            showsepia = false; showsepiaVignett = false;
            showGray = false;
            showblurr1 = false; showblurr2 = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;            
            showChangeContrast = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'd':    // Toggle brightness change mode
        {
            showChangeBrightness = !showChangeBrightness;
            // Ensuring that no other modes are active
            showalternateGray = false;
            showsepia = false; showsepiaVignett = false;
            showGray = false;
            showblurr1 = false; showblurr2 = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;            
            showRetainCertainColor = false;
            showChangeContrast = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'e':    // Toggle contrast change mode
        {
            showChangeContrast = !showChangeContrast;
            // Ensuring that no other modes are active
            showalternateGray = false;
            showsepia = false; showsepiaVignett = false;
            showGray = false;
            showblurr1 = false; showblurr2 = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;            
            showRetainCertainColor = false;
            showChangeBrightness = false;
            showCartooning = false;
            showswapfaces = false;
            break;
        }
        case 'j':    // Toggle cartooning mode
        {
            showCartooning = !showCartooning;
            // Ensuring that no other modes are active
            showalternateGray = false;
            showsepia = false; showsepiaVignett = false;
            showGray = false;
            showblurr1 = false; showblurr2 = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;            
            showRetainCertainColor = false;
            showChangeBrightness = false;
            showChangeContrast = false;
            showswapfaces = false;
            break;
        }
        case 'k':    // Toggle swap faces mode
        {
            showswapfaces = !showswapfaces;
            // Ensuring that no other modes are active
            showalternateGray = false;
            showsepia = false; showsepiaVignett = false;
            showGray = false;
            showblurr1 = false; showblurr2 = false;
            showsobelX = false;
            showsobelY = false;
            showmagnitude = false;
            showquantizing = false;
            showfaces = false;
            showblurrfaces = false;
            showembossing = false;
            showcolorFace = false;
            showHalo = false;          
            showRetainCertainColor = false;
            showChangeBrightness = false;
            showChangeContrast = false;
            showCartooning = false;
            break;
        }
        case 's':     // Save the current frame with applied effect
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
            else if (showsepiaVignett) {
                filename = "saved_frame_sepia_vignett_" + timestamp + ".jpg";
                imwrite(filename, sepiaFil);
            }
            else if (showblurr1) {
                filename = "saved_frame_blurr1_" + timestamp + ".jpg";
                imwrite(filename, Blurr1Fil);
            }
            else if (showblurr2) {
                filename = "saved_frame_blurr2_" + timestamp + ".jpg";
                imwrite(filename, Blurr2Fil);
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

        // Default case is intentionally omitted to allow continuous processing
        }
    }

    // Clean up and release resources
    delete capdev;
    return 0;
}


