# Video special effects

## Project Overview
This project showcases advanced image processing techniques implemented using the OpenCV library in C++. It includes a variety of features such as grayscale conversion, sepia-tone filtering, Sobel edge detection, median filtering, and face swapping. The code is organized into multiple modules, each handling specific image processing tasks, and is designed to be both efficient and modular.

## Demo Video
- [Project Demonstration Video](https://northeastern.instructuremedia.com/embed/ffb02cf9-01fa-4e0d-9b67-df6048c2dbc2)

## Development Environment
- **Operating System**: Windows 10
- **IDE**: Visual Studio 2022

## Features
The application includes the following key features, each of which can be toggled using specific keyboard inputs:
- Grayscale and custom grayscale conversion
- Sepia-tone filtering
- Sobel edge detection for X and Y directions
- Median filtering
- Face swapping
- Dynamic brightness and contrast adjustments
- Selective color retention
- Cartoonization of video streams
- Blurring techniques
- Embossing effects
- Colorful face detection on grayscale background

## Instructions
To run this project:
1. Clone the repository.
2. Open the project in Visual Studio 2022.
3. Ensure OpenCV is correctly linked (see the `OpenCV Setup` section below).
4. Build and run the application.

## OpenCV Setup
This project requires OpenCV. Follow the steps below to set it up:
1. Download and install OpenCV.
2. Configure OpenCV with Visual Studio.
3. Include OpenCV directories in your project settings.
4. Link OpenCV libraries.

For detailed instructions, refer to the [OpenCV Installation Guide](https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html).

## Keybindings
The application responds to the following keybindings:
- `g`: Toggle grayscale mode
- `p`: Toggle sepia filter
- `x`, `y`: Toggle Sobel filters in respective directions
- `m`: Toggle gradient magnitude mode
- `f`: Toggle face detection
- `r`: Blur outside faces
- `t`: Apply embossing effect
- `c`: Colorful faces on grayscale
- `d`, `e`: Adjust brightness/contrast (use mouse scroll)
- `a`: Retain specific color
- `j`: Cartoonize video stream
- `k`: Swap faces

## Acknowledgements
- **Textbook**: "Computer Vision: Algorithms and Applications 2nd Edition" by Richard Szeliski.
- **Additional Resources**: Various online materials and forums including OpenCV documentation, image processing tutorials, and community discussions.

## Contact
- Name: Manushi
