/*! \file filter.cpp
    \brief Image Manipulation Functions.
    \author Manushi

    This file contains functions for image manipulation, including custom grayscale conversion.
*/

#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

/*!
 *  \brief Custom grayscale conversion function.
 *  \param src The source color image (input).
 *  \param dst The destination grayscale image (output).
 *  \return int Returns 0 on successful conversion, -1 if source image is empty.
 *
 *  This function converts a color image to grayscale using a custom method.
 *  The conversion is achieved by subtracting the red channel from 255 and copying the value to all three color channels.
 *  This creates a unique grayscale effect different from standard grayscale conversion methods.
 */

int greyscale(Mat& src, Mat& dst) {
    if (src.empty()) {
        return -1;
    }

    /*
       CV_8U: This part of the constant represents an 8-bit unsigned integer.
              In image processing, an 8-bit unsigned integer per channel is a common way to represent color values, as it allows for 256 different shades (from 0 to 255).
              In the context of grayscale images, this range is sufficient to represent shades from black (0) to white (255).

       C1: This indicates that the image has only one channel. In OpenCV, different types of images (e.g., grayscale, color) are represented by the number of channels they have.
           A grayscale image, which represents intensity information only, requires just one channel.
           In contrast, a standard color image in BGR format would typically use CV_8UC3 (three channels: blue, green, and red).
    */
    dst = Mat(src.rows, src.cols, CV_8UC1);

    for (int i = 0; i < src.rows; ++i) {
        // Get pointer to the i-th row of src
        Vec3b* pSrcRow = src.ptr<Vec3b>(i);

        // Get the pointer to the start of the ith row of grayscale output image, since we only need one channel information, no need of Vec3b
        uchar* pDstRow = dst.ptr<uchar>(i);

        for (int j = 0; j < src.cols; ++j) {
            // Custom grayscale transformation
            uchar green = pSrcRow[j][1];  // get the green color channel value at the pixel
            pDstRow[j] = 255 - green; // Subtracting the green channel from 255 at the pixel
        }
    }

    return 0;
}


/*!
 *  \brief Apply a sepia tone filter with vignetting to an image.
 *  \param src The source color image (input).
 *  \param dst The destination sepia-toned image (output).
 *  \return int Returns 0 on successful execution, -1 if source image is empty.
 *
 *  This function applies a sepia tone filter to the source image with vignetting effect.
 *  Vignetting makes the image get darker towards the edges.
 */
int sepia(Mat& src, Mat& dst) {
    if (src.empty()) {
        return -1;
    }

    src.copyTo(dst);  // makes a copy of the original image

    // need to get the center of the image
    Point center(src.cols / 2, src.rows / 2);

    /*
      We use euclian distance to get the distance from the center of the image to the one of its corners

      center.x is half the width of the image, and center.y is half the height.
      Squaring the distances is part of calculating the Euclidean distance
      The square root of the sum of the squared distances, calculates the Euclidean distance from the center of the image to one of its corners
    */
    double maxDist = sqrt(center.x * center.x + center.y * center.y);


    for (int i = 0; i < src.rows; ++i) {
        // Get pointer to the i-th row of src
        Vec3b* pSrcRow = src.ptr<Vec3b>(i);

        // Get the pointer to the start of the ith row of output image
        Vec3b* pDstRow = dst.ptr<Vec3b>(i);

        for (int j = 0; j < src.cols; ++j) {

            // Save the current BGR values
            int blue = pSrcRow[j][0];
            int green = pSrcRow[j][1];
            int red = pSrcRow[j][2];

            /*
                Apply the column wise co - efficients to the red, green, blue
            
                 0.393, 0.349, 0.272,    // Red coefficients
                 0.769, 0.686, 0.534,    // Green coefficients
                 0.189, 0.168, 0.131     // Blue coefficients
            */

            double sepiaRed = (0.393 * red) + (0.769 * green) + (0.189 * blue);
            double sepiaGreen = (0.349 * red) + (0.686 * green) + (0.168 * blue);
            double sepiaBlue = (0.272 * red) + (0.534 * green) + (0.131 * blue);

            //Apply vignetting effect to the image
            // Using distance formula get how far is the pixel from the centre of the image
            double dist = sqrt((center.x - j) * (center.x - j) + (center.y - i) * (center.y - i));
            double scale = 0.75 * (maxDist - dist) / maxDist + 0.25;  // Scale factor for vignetting, closer the pixel to center brigher the image, 

            // Now apply the vigentting to the pixel
            double vignetteRed = sepiaRed * scale;
            double vignetteGreen = sepiaGreen * scale;
            double vignetteBlue = sepiaBlue * scale;

            // normalize and Update pixel values in the new Mat
            pDstRow[j] = Vec3b(saturate_cast<uchar>(vignetteBlue), saturate_cast<uchar>(vignetteGreen), saturate_cast<uchar>(vignetteRed));
        }
    }

    return 0;
}


/*!
 *  \brief Apply a guassian filter 5x5.
 *  \param src The source color image (input).
 *  \param dst The destination guassian filtered (output)
 *  \return int Returns 0 on successful execution, -1 if source image is empty.
 *
 *  This function applies a guassian filter to the source image
 *  Applying this filter makes the image go blurr
 * 
 * Guassian Filter 
   [1 2 4 2 1; 
    2 4 8 4 2; 
    4 8 16 8 4;
    2 4 8 4 2; 
    1 2 4 2 1]

 */
int blur5x5_1(cv::Mat& src, cv::Mat& dst) {

    if (src.empty() || src.type() != CV_8UC3) {
        return -1; // Return error if source image is empty or not a 3-channel image
    }

    // create the dst iamge
    src.copyTo(dst);  // makes a copy of the original image

    //std::vector<std::vector<int>> kernel = { {1, 2, 4, 2, 1},
    //                                        {2, 4, 8, 4, 2},
    //                                        {4, 8, 16, 8, 4},
    //                                        {2, 4, 8, 4, 2},
    //                                        {1, 2, 4, 2, 1} };

    // only going to calcualte values for pixels where filter fits in the image
    // nested loop

    for (int i = 2; i < src.rows - 2; i++) {  // rows 
        for (int j = 2; j < src.cols - 2; j++) {   // cols
            for (int k = 0; k < src.channels(); k++) {   // color channels

                int sum = src.at<Vec3b>(i - 2, j - 2)[k] + 2 * src.at<Vec3b>(i - 2, j - 1)[k] + 4 * src.at<Vec3b>(i - 2, j)[k] + 2 * src.at<Vec3b>(i - 2, j + 1)[k] + src.at<Vec3b>(i - 2, j + 2)[k] +
                    2 * src.at<Vec3b>(i - 1, j - 2)[k] + 4 * src.at<Vec3b>(i - 1, j - 1)[k] + 8 * src.at<Vec3b>(i - 1, j)[k] + 4 * src.at<Vec3b>(i - 1, j + 1)[k] + 2 * src.at<Vec3b>(i - 1, j + 2)[k] +
                    4 * src.at<Vec3b>(i, j - 2)[k] + 8 * src.at<Vec3b>(i, j - 1)[k] + 16 * src.at<Vec3b>(i, j)[k] + 8 * src.at<Vec3b>(i, j + 1)[k] + 4 * src.at<Vec3b>(i, j + 2)[k] +
                    2 * src.at<Vec3b>(i + 1, j - 2)[k] + 4 * src.at<Vec3b>(i + 1, j - 1)[k] + 8 * src.at<Vec3b>(i + 1, j)[k] + 4 * src.at<Vec3b>(i + 1, j + 1)[k] + 2 * src.at<Vec3b>(i + 1, j + 2)[k] +
                    src.at<Vec3b>(i + 2, j - 2)[k] + 2 * src.at<Vec3b>(i + 2, j - 1)[k] + 4 * src.at<Vec3b>(i + 2, j)[k] + 2 * src.at<Vec3b>(i + 2, j + 1)[k] + src.at<Vec3b>(i + 2, j + 2)[k];
                
                
                // normalize the value back to a range of [0 255], which is within 1 byte
                sum /= 100;
                
                dst.at<Vec3b>(i, j)[k] = sum;

            }

            //Vec3b blurPixel(0,0,0);
            //for (int bi = -2; bi <= 2; bi++) {   // color channels
            //    for (int bj = -2; bj <= 2; bj++) {   // color channels

            //        for (int k = 0; k < src.channels(); k++) {   // color channels
            //            Vec3b pixel = src.at<Vec3b>(i + bi, j + bj);
            //            blurPixel[k] += pixel[k] * kernel[bi + 2][bj + 2];;   // at particular pixel keep adding the red, blue, red values from around pixels and save it in blurPixel
            //        }
            //    }
            //}
            //// If we add up all the values in the filter, the resulted sum is 100, we divide the calculated
            //// since we have added all the channels for all 5x5 of image add that to the dst image
            //dst.at<Vec3b>(i, j) = Vec3b(blurPixel[0]/100, blurPixel[1]/100, blurPixel[2]/100);
        }
    }

    return 0;
}

/*!
 *  \brief Apply a fast 5x5 blur filter using separable filters.
 *  \param src The source color image (input).
 *  \param dst The destination blurred image (output).
 *  \return int Returns 0 on successful execution, -1 if source image is empty or not a 3-channel image.
 *
 *  This function applies a 1x5 blur filter to the source image using separable matrix for improved performance.
 *  The blur is achieved by first applying a horizontal 1x5 matrix, followed by a vertical 5x1 matrix.
 *  This method avoids using the slower 'at' method for pixel access, instead using pointer access for efficiency.
 * 
 * Guassian 1x5 separable Filter 
   [1 2 4 2 1]
 */
int blur5x5_2(cv::Mat& src, cv::Mat& dst) {
    if (src.empty() || src.type() != CV_8UC3) {
        return -1; // Return error if source image is empty or not a 3-channel image
    }

    // create the dst iamge
    src.copyTo(dst);  // makes a copy of the original image

    // only going to calcualte values for pixels where filter fits in the image
    // nested loop

    for (int i = 0; i < src.rows; i++) {  // rows 

        Vec3b* pSrcRow = src.ptr<Vec3b>(i);  // getting the pointer to row i
        Vec3b* pDstRow = dst.ptr<Vec3b>(i);  // getting the DST pointer to row i

        for (int j = 2; j < src.cols - 2; j++) {   // cols
            for (int k = 0; k < src.channels(); k++) {   // color channels

                int sum = pSrcRow[j - 2][k] + 2 * pSrcRow[j - 1][k] + 4 * pSrcRow[j][k] + 2 * pSrcRow[j + 1][k] + pSrcRow[j + 2][k];

                // normalize the value back to a range of [0 255], which is within 1 byte
                sum /= 10;

                pDstRow[j][k] = sum;
            }
        }
    }

    return 0;

}



/*!
 *  \brief Apply a 3x3 Sobel X filter to a color image.
 *  \param src The source color image (input).
 *  \param dst The destination image with Sobel X filter applied (output).
 *  \return int Returns 0 on successful execution, -1 if source image is empty or not a 3-channel image.
 *
 *  This function applies a 3x3 Sobel X filter to the source image, emphasizing horizontal edges.
 *  The filter is positive to the right, and the output image is of type 16SC3 (signed short, 3 channels).
 * 
 */
int sobelX3x3(Mat& src, Mat& dst) {
    if (src.empty() || src.type() != CV_8UC3 || dst.type() != CV_16SC3) {
        return -1; // Return error if source image is empty or not a 3-channel image
    }

    for (int i = 1; i < src.rows - 1; i++) {  // rows 

        Vec3b* pSrcRow = src.ptr<Vec3b>(i);
        Vec3s* pDstRow = dst.ptr<Vec3s>(i);

        for (int j = 1; j < src.cols - 1; j++) {   // cols
            for (int k = 0; k < src.channels(); k++) {   // color channels
                int pixel_y = 0;
                int pixel_x = 0;

                if (pSrcRow) {
                     pixel_x = -1 * (j > 0 ? pSrcRow[j - 1][k] : 0) +
                         1 * (j < src.cols - 1 ? pSrcRow[j + 1][k] : 0) +
                         -2 * (j > 0 ? pSrcRow[j - 1][k] : 0) +
                         2 * (j < src.cols - 1 ? pSrcRow[j + 1][k] : 0) +
                         -1 * (j > 0 ? pSrcRow[j - 1][k] : 0) +
                         1 * (j < src.cols - 1 ? pSrcRow[j + 1][k] : 0);
                }

                /*if (pixel_x < 0) pixel_x = 0;
                if (pixel_x > 255) pixel_x = 255;*/
                pDstRow[j][k] = pixel_x;
            }
        }
    }

    return 0;

}

/*!
 *  \brief Apply a 3x3 Sobel Y filter to a color image.
 *  \param src The source color image (input).
 *  \param dst The destination image with Sobel Y filter applied (output).
 *  \return int Returns 0 on successful execution, -1 if source image is empty or not a 3-channel image.
 *
 *  This function applies a 3x3 Sobel Y filter to the source image, emphasizing vertical edges.
 *  The filter is positive upwards, and the output image is of type 16SC3 (signed short, 3 channels).
 */
int sobelY3x3(Mat& src, Mat& dst) {

    if (src.empty() || src.type() != CV_8UC3 || dst.type() != CV_16SC3) {
        return -1; // Return error if source image is empty or not a 3-channel image
    }

    for (int i = 1; i < src.rows - 1; i++) {  // rows 

        Vec3b* pSrcPrevRow = (i > 0) ? src.ptr<Vec3b>(i - 1) : nullptr;
        Vec3b* pSrcNextRow = (i < src.rows - 1) ? src.ptr<Vec3b>(i + 1) : nullptr;
        Vec3s* pDstRow = dst.ptr<Vec3s>(i);

        for (int j = 1; j < src.cols - 1; j++) {   // cols
            for (int k = 0; k < src.channels(); k++) {   // color channels
                int pixel_y = 0;
                int pixel_x = 0;

                if (pSrcPrevRow && pSrcNextRow) {
                    pixel_y = 1 * (j > 0 ? pSrcPrevRow[j - 1][k] : 0) +
                        2 * pSrcPrevRow[j][k] +
                        1 * (j < src.cols - 1 ? pSrcPrevRow[j + 1][k] : 0) -
                        1 * (j > 0 ? pSrcNextRow[j - 1][k] : 0) -
                        2 * pSrcNextRow[j][k] -
                        1 * (j < src.cols - 1 ? pSrcNextRow[j + 1][k] : 0);
                }

                /*if (pixel_y < 0) pixel_y = 0;
                if (pixel_y > 255) pixel_y = 255;*/
                pDstRow[j][k] = pixel_y;
            }
        }
    }

    return 0;
}

/**
 * @brief Generates a gradient magnitude image from X and Y Sobel images.
 *
 * @param sx Input image representing the Sobel X gradient (3-channel signed short).
 * @param sy Input image representing the Sobel Y gradient (3-channel signed short).
 * @param dst Output image representing the gradient magnitude (single-channel uchar).
 * @return int Returns 0 on success, -1 on failure.
 */
int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst) {

    if (sx.empty() || sy.empty() || sx.type() != CV_16SC3 || sy.type() != CV_16SC3 || sx.size != sy.size || sx.type() != sy.type() /* || dst.type() != CV_8UC3 */ ) {
        return -1; // Return error if source image is empty or not a 3-channel image
    }

    /*Mat gray_x;
    Mat gray_y;
    cvtColor(sx, gray_x, COLOR_BGR2GRAY);
    cvtColor(sx, gray_y, COLOR_BGR2GRAY);*/

    // Compute gradient magnitude
    for (int i = 0; i < sx.rows; i++) {
        Vec3b* pDstRow = dst.ptr<Vec3b>(i);

        for (int j = 0; j < sx.cols; j++) {
            double sum_magnitude = 0.0;
            for (int k = 0; k < sx.channels(); k++) {
                // Compute gradient magnitude for each channel using pointer access
                short val_x = sx.ptr<cv::Vec3s>(i)[j][k];
                short val_y = sy.ptr<cv::Vec3s>(i)[j][k];
                sum_magnitude += sqrt(val_x * val_x + val_y * val_y);
                //dst.ptr<Vec3b>(i)[j][k] = saturate_cast<uchar>(std::min(float(sum_magnitude), 255.0f));
                ///dst.ptr<Vec3b>(i)[j][k] = saturate_cast<uchar>(std::min(sum_magnitude, 255.0f));
                //dst.ptr<Vec3b>(i)[j][k] = saturate_cast<uchar>(sum_magnitude);
                pDstRow[j][k] = saturate_cast<uchar>(std::min(float(sum_magnitude), 255.0f));
                // Assign the magnitude to the destination pixel
                //pDstRow[j][k] = saturate_cast<short>(sum_magnitude);
                //dst.ptr<Vec3b>(i)[j][k] = saturate_cast<uchar>(std::min(float(sum_magnitude), 255.0f));
                //dst.ptr<Vec3b>(i)[j][k] = std::min(float(sum_magnitude), 255.0f);
            }

            // Average the magnitudes from all channels
            //float avg_magnitude = (float)(sum_magnitude / sx.channels());

            // Scaling and converting to uchar
            //dst.ptr<uchar>(i)[j] = saturate_cast<uchar>(std::min(avg_magnitude, 255.0f));
            //dst.ptr<uchar>(i)[j] = saturate_cast<uchar>(sum_magnitude);
            //dst.ptr<uchar>(i)[j] = saturate_cast<uchar>(std::min(float(sum_magnitude), 255.0f));
            //dst.ptr<Vec3b>(i)[j] = saturate_cast<uchar>(std::min(avg_magnitude, 255.0f));
            //dst.ptr<uchar>(i)[j] = saturate_cast<uchar>(std::min(avg_magnitude, 255.0f));
            //pDstRow[j] = saturate_cast<uchar>(std::min(float(avg_magnitude), 255.0f));
        }
    }

    return 0;
}

/**
 * @brief Blurs and quantizes a color image.
 *
 * @param src Input image (color image).
 * @param dst Output image (blurred and quantized color image).
 * @param levels The number of levels to quantize each color channel.
 * @return int Returns 0 on success, -1 on failure.
 */
int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels) {
    // Validate input
    if (src.empty() || src.type() != CV_8UC3 || levels <= 0) {
        return -1;
    }

    // Apply Gaussian blur
    blur5x5_2(src, dst);

    // Calculate bucket size
    int bucketSize = 255 / levels;

    // Quantize the image using pointer access
    for (int i = 0; i < dst.rows; i++) {
        Vec3b* pDstRow = dst.ptr<Vec3b>(i);
        for (int j = 0; j < dst.cols; j++) {
            for (int k = 0; k < 3; k++) {  // Loop through all 3 color channels
                int quantizedValue = (pDstRow[j][k] / bucketSize) * bucketSize;
                pDstRow[j][k] = saturate_cast<uchar>(quantizedValue);
            }
        }
    }

    return 0;
}


/**
 * @brief Creates an embossing effect using Sobel X and Y gradients on 3-channel images.
 *
 * This function applies an embossing effect to a 3-channel (e.g., RGB) image using the dot product
 * of Sobel X and Y gradients with a direction vector. The direction vector (0.7071, 0.7071) is used,
 * corresponding to a 45-degree angle, to simulate a light source from the top-left corner.
 *
 * @param sobelX The Sobel X gradient image (3-channel, 8-bit unsigned).
 * @param sobelY The Sobel Y gradient image (3-channel, 8-bit unsigned).
 * @param emboss The output image with the embossing effect applied (3-channel, 8-bit unsigned).
 */
void createEmbossEffect(const Mat& sobelX, const Mat& sobelY, Mat& emboss) {
    // Ensure that the input images are of type CV_8UC3
    CV_Assert(sobelX.type() == CV_8UC3 && sobelY.type() == CV_8UC3);

    // Direction vector components for a 45-degree angle
    //const float dirX = 0.7071f;
    //const float dirY = 0.7071f;

    //// Direction vector components for a 135-degree angle
    /*const float dirX = -0.7071f;
    const float dirY = 0.7071f;*/

    //// Direction vector components for a 225-degree angle
    /*const float dirX = -0.7071f;
    const float dirY = -0.7071f;*/

    //// Direction vector components for a 315-degree angle
    const float dirX = 0.7071f;
    const float dirY = -0.7071f;

    // Iterate through each pixel to calculate the emboss effect
    for (int i = 0; i < sobelX.rows; i++) {
        const Vec3b* ptrX = sobelX.ptr<Vec3b>(i);
        const Vec3b* ptrY = sobelY.ptr<Vec3b>(i);
        Vec3b* ptrEmboss = emboss.ptr<Vec3b>(i);

        for (int j = 0; j < sobelX.cols; j++) {
            for (int k = 0; k < 3; k++) {  // Iterate over each channel
                // Calculate dot product with the direction vector for each channel
                float dot = (ptrX[j][k] - 128) * dirX + (ptrY[j][k] - 128) * dirY;

                // Normalize and scale the result, then shift back to the range [0, 255]
                dot = std::min(std::max(dot + 128, 0.0f), 255.0f);

                // Assign the calculated value to the emboss image
                ptrEmboss[j][k] = saturate_cast<uchar>(dot);
            }
        }
    }
}

/**
 * @brief Applies a grayscale effect to the entire image except for detected faces.
 *
 * This function first converts the entire image to grayscale and then copies the color regions
 * of detected faces back onto the grayscale image using pointer access for efficiency.
 *
 * @param src The source image (color image).
 * @param dst The destination image (grayscale with color faces).
 * @param faces A vector of rectangles representing detected faces.
 */
void colorFacesOnGray(cv::Mat& src, cv::Mat& dst, const std::vector<cv::Rect>& faces) {
    // Ensure that the input images are of type CV_8UC3
    CV_Assert(src.type() == CV_8UC3 && dst.type() == CV_8UC3);

    // Convert the entire image to grayscale
    cvtColor(src, dst, COLOR_BGR2GRAY);
    cvtColor(dst, dst, COLOR_GRAY2BGR); // Convert back to BGR for color copy

    // Iterate over each detected face
    for (const auto& face : faces) {
        // Copy color region of each face from src to dst
        for (int y = face.y; y < face.y + face.height; ++y) {
            cv::Vec3b* ptrSrc = src.ptr<Vec3b>(y);
            cv::Vec3b* ptrDst = dst.ptr<Vec3b>(y);
            for (int x = face.x; x < face.x + face.width; ++x) {
                ptrDst[x] = ptrSrc[x]; // Copy color pixel
            }
        }
    }
}


/**
 * @brief Adds a sparkle halo effect above detected faces in an image.
 *
 * This function modifies the pixels above each detected face to create a sparkle or halo effect.
 * It assumes that the face rectangles are correctly identified and uses pointer access for pixel manipulation.
 *
 * @param image The source image (color image).
 * @param faces A vector of rectangles representing detected faces.
 */
void addSparkleHalo(cv::Mat& image, const std::vector<cv::Rect>& faces) {
    // Ensure that the input images are of type CV_8UC3
    CV_Assert(image.type() == CV_8UC3);

    // Iterate over each detected face
    for (const auto& face : faces) {
        // Define the region for the halo effect above the face
        int haloTop = std::max(face.y - face.height / 2, 0);
        int haloBottom = face.y;
        int haloLeft = face.x;
        int haloRight = face.x + face.width;

        // Add sparkle effect in the defined halo region
        for (int y = haloTop; y < haloBottom; ++y) {
            cv::Vec3b* ptr = image.ptr<cv::Vec3b>(y);
            for (int x = haloLeft; x < haloRight; ++x) {
                // Example sparkle effect: alternating pixel brightness
                if ((x + y) % 10 < 5) { // Simple pattern for illustration
                    ptr[x] = cv::Vec3b(255, 255, 255); // Set to white for sparkle
                }
            }
        }
    }
}


/**
 * @brief Applies a median filter to a 3-channel (8UC3) image using pointer method.
 *
 * This function uses pointer access to apply a median filter to a 3-channel color image.
 * It processes each pixel channel by determining the median value in its neighborhood
 * and setting the pixel channel to this median value.
 *
 * @param src The source image (3-channel, 8-bit).
 * @param dst The destination image after applying the median filter.
 * @param kernelSize The size of the kernel for the median filter.
 */
void medianFilterPtrColor(const cv::Mat& src, cv::Mat& dst, int kernelSize) {
    // Ensure that the input images are of type CV_8UC3
    CV_Assert(src.type() == CV_8UC3/* && dst.type() == CV_8UC3*/);

    // Ensure kernel size is odd
    if (kernelSize % 2 == 0) ++kernelSize;

    // Initialize the destination image
    dst = src.clone();

    // Define half kernel size
    int kHalf = kernelSize / 2;

    // Iterate over each pixel in the image
    for (int y = kHalf; y < src.rows - kHalf; ++y) {
        for (int x = kHalf; x < src.cols - kHalf; ++x) {
            // Process each channel separately
            for (int c = 0; c < 3; ++c) {
                std::vector<uchar> neighborhood;

                // Collect neighborhood pixel values using pointer access
                for (int ky = -kHalf; ky <= kHalf; ++ky) {
                    const cv::Vec3b* ptrSrc = src.ptr<cv::Vec3b>(y + ky);
                    for (int kx = -kHalf; kx <= kHalf; ++kx) {
                        neighborhood.push_back(ptrSrc[x + kx][c]);
                    }
                }

                // Sort and find the median
                std::nth_element(neighborhood.begin(), neighborhood.begin() + neighborhood.size() / 2, neighborhood.end());
                uchar median = neighborhood[neighborhood.size() / 2];

                // Set the median value to the central pixel using pointer access
                dst.ptr<cv::Vec3b>(y)[x][c] = median;
            }
        }
    }
}


/**
 * @brief Converts an image to grayscale except for a specified color.
 *
 * This function processes a 3-channel color image, turning it to grayscale
 * except for pixels that match a specified color within a tolerance range.
 *
 * @param src The source image (3-channel, 8-bit).
 * @param dst The destination image (grayscale with preserved color).
 * @param targetColor The color to preserve in the image.
 * @param tolerance The tolerance range for color matching.
 */
void colorPreserveGray(cv::Mat& src, cv::Mat& dst, cv::Scalar& targetColor, int tolerance) {
    // Initialize the destination image
    dst = src.clone();

    // Iterate over each pixel in the image
    for (int y = 0; y < src.rows; ++y) {
        cv::Vec3b* ptrSrc = src.ptr<cv::Vec3b>(y);
        cv::Vec3b* ptrDst = dst.ptr<cv::Vec3b>(y);

        for (int x = 0; x < src.cols; ++x) {
            // Check if the pixel color is within the tolerance of the target color
            if (abs(ptrSrc[x][0] - targetColor[0]) > tolerance ||
                abs(ptrSrc[x][1] - targetColor[1]) > tolerance ||
                abs(ptrSrc[x][2] - targetColor[2]) > tolerance) {
                // Convert to grayscale if not within tolerance
                /*
                     Red: The eye is moderately sensitive to red, so it's given a weight of 0.299.
                     Green: The eye is most sensitive to green, so it has the highest weight of 0.587.
                     Blue: The eye is least sensitive to blue, hence the smallest weight of 0.114.
                     By multiplying each color component by these weights and summing them, we obtain a single intensity 
                     value that approximates the perceived brightness of the original color pixel.
                */
                uchar gray = saturate_cast<uchar>(0.299 * ptrSrc[x][2] + 0.587 * ptrSrc[x][1] + 0.114 * ptrSrc[x][0]);
                ptrDst[x] = cv::Vec3b(gray, gray, gray);
            }
        }
    }
}


/**
 * @brief Adjusts the brightness of an image.
 *
 * This function modifies the brightness of an image by adding a constant value to all pixel intensities.
 * The operation is performed using pointer access for efficiency.
 *
 * @param src The source image (3-channel, 8-bit).
 * @param dst The destination image with adjusted brightness.
 * @param brightness The brightness adjustment value.
 */
void adjustBrightness(cv::Mat& src, cv::Mat& dst, int brightness) {
    // Initialize the destination image
    dst = src.clone();

    // Iterate over each pixel in the image
    for (int y = 0; y < src.rows; ++y) {
        cv::Vec3b* ptrSrc = src.ptr<cv::Vec3b>(y);
        cv::Vec3b* ptrDst = dst.ptr<cv::Vec3b>(y);

        for (int x = 0; x < src.cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                // Apply brightness adjustment and clamp the result
                int pixelVal = ptrSrc[x][c] + brightness;
                ptrDst[x][c] = saturate_cast<uchar>(std::max(0, std::min(pixelVal, 255)));
            }
        }
    }
}

/**
 * @brief Adjusts the contrast of an image.
 *
 * This function modifies the contrast of an image by scaling the difference from the mean intensity.
 * The operation is performed using pointer access for efficiency.
 *
 * @param src The source image (3-channel, 8-bit).
 * @param dst The destination image with adjusted contrast.
 * @param contrast The contrast scaling factor.
 */
void adjustContrast(cv::Mat& src, cv::Mat& dst, double contrast) {
    // Initialize the destination image
    dst = src.clone();

    // Calculate the average pixel value for contrast adjustment
    double averageBrightness = 0.0;
    for (int y = 0; y < src.rows; ++y) {
        const cv::Vec3b* ptrSrc = src.ptr<cv::Vec3b>(y);
        for (int x = 0; x < src.cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                averageBrightness += ptrSrc[x][c];
            }
        }
    }
    averageBrightness /= (src.rows * src.cols * 3);

    // Apply contrast adjustment
    for (int y = 0; y < src.rows; ++y) {
        cv::Vec3b* ptrSrc = src.ptr<cv::Vec3b>(y);
        cv::Vec3b* ptrDst = dst.ptr<cv::Vec3b>(y);

        for (int x = 0; x < src.cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                // Scale pixel contrast and clamp the result
                double pixelVal = averageBrightness + (ptrSrc[x][c] - averageBrightness) * contrast;
                ptrDst[x][c] = saturate_cast<uchar>(std::max(0.0, std::min(pixelVal, 255.0)));
            }
        }
    }
}



/**
 * @brief Cartoonifies an image.
 *
 * This function processes an image to give it a cartoon-like effect by:
 * - Converting to grayscale and applying a median blur to smoothen it.
 * - Detecting edges using adaptive thresholding.
 * - Applying a bilateral filter to reduce noise and preserve edges.
 * - Creating a mask from the edges and combining it with the color image.
 *
 * @param src The source image (3-channel, 8-bit).
 * @param dst The destination image after applying the cartoon effect.
 */
void cartoonify(cv::Mat& src, cv::Mat& dst) {
    // Ensure the input is a 3-channel image
    if (src.channels() != 3) {
        std::cerr << "Input must be a 3-channel image." << std::endl;
        return;
    }

    // Convert to grayscale
    cv::Mat grayScaleImage;
    cv::cvtColor(src, grayScaleImage, cv::COLOR_BGR2GRAY);

    // Applying median blur to smoothen the grayscale image
    cv::medianBlur(grayScaleImage, grayScaleImage, 5);

    // Retrieving the edges for the cartoon effect using adaptive thresholding
    cv::Mat edgeMask;
    cv::adaptiveThreshold(grayScaleImage, edgeMask, 255,
        cv::ADAPTIVE_THRESH_MEAN_C,
        cv::THRESH_BINARY, 9, 9);

    // Applying bilateral filter to reduce noise and keep the edge sharp as required
    cv::Mat colorImage;
    cv::bilateralFilter(src, colorImage, 9, 300, 300);

    // Preparing the dst image container with the same size and type as src
    dst.create(src.size(), src.type());

    // Iterate over each pixel in the image using the ptr method for pointer access
    for (int y = 0; y < src.rows; ++y) {
        cv::Vec3b* ptrColorImage = colorImage.ptr<cv::Vec3b>(y);
        cv::Vec3b* ptrEdgeMask = edgeMask.ptr<cv::Vec3b>(y);
        cv::Vec3b* ptrDst = dst.ptr<cv::Vec3b>(y);

        for (int x = 0; x < src.cols; ++x) {
            // Masking the color image with the edge mask to produce the cartoon effect
            uchar maskValue = ptrEdgeMask[x][0]; // Assuming single channel for edge mask
            for (int c = 0; c < 3; c++) { // Iterate over each color channel
                ptrDst[x][c] = maskValue == 255 ? ptrColorImage[x][c] : 0;
            }
        }
    }
}


/**
 * @brief Swaps two faces in an image, retaining the rest of the image.
 *
 * This function interchanges two faces in an image. It resizes the faces to fit each other's
 * locations and then swaps them, while keeping the rest of the image unchanged.
 *
 * @param src The source image (color image).
 * @param dst The destination image with swapped faces.
 * @param faces A vector of rectangles representing detected faces.
 * @return int Returns 0 on success, -1 if there are less than two faces.
 */
int swapFaces(cv::Mat& src, cv::Mat& dst, std::vector<cv::Rect>& faces) {
    // Check if there are at least two faces
    if (faces.size() < 2) {
        return -1;
    }

    // Clone the source image to the destination
    dst = src.clone();

    // Only swap the first two faces for simplicity
    const cv::Rect& face1 = faces[0];
    const cv::Rect& face2 = faces[1];


    // Temporary matrices to hold resized face regions
    cv::Mat face1Resized, face2Resized;
    cv::resize(src(face1), face1Resized, cv::Size(face2.width, face2.height));
    cv::resize(src(face2), face2Resized, cv::Size(face1.width, face1.height));

    // Create masks for each face
    cv::Mat mask1 = cv::Mat::zeros(face1.height, face1.width, CV_8UC1);
    mask1.setTo(cv::Scalar(255));
    cv::Mat mask2 = cv::Mat::zeros(face2.height, face2.width, CV_8UC1);
    mask2.setTo(cv::Scalar(255));

    // Swap the resized face regions
    face1Resized.copyTo(dst(cv::Rect(face2.x, face2.y, face1Resized.cols, face1Resized.rows)), mask2);
    face2Resized.copyTo(dst(cv::Rect(face1.x, face1.y, face2Resized.cols, face2Resized.rows)), mask1);

#if 0
    // Temporary matrices to hold face regions
    cv::Mat face1Region = src(face1).clone();
    cv::Mat face2Region = src(face2).clone();

    /*imshow("Video Capture 2 ", face1Region);
    imshow("Video Capture 3", face2Region);*/

    // Swap face regions using the ptr method
    for (int y = 0; y < face1.height; ++y) {
        cv::Vec3b* ptrDstFace1 = dst.ptr<cv::Vec3b>(face1.y + y);
        cv::Vec3b* ptrSrcFace2 = face2Region.ptr<cv::Vec3b>(y);

        for (int x = 0; x < face1.width; ++x) {
            ptrDstFace1[face1.x + x] = ptrSrcFace2[x];
        }
    }

    imshow("Video Capture 2 ", dst);
    
    //for (int y = 0; y < face2.height; ++y) {
    //    cv::Vec3b* ptrDstFace2 = dst.ptr<cv::Vec3b>(face2.y + y);
    //    cv::Vec3b* ptrSrcFace1 = face1Region.ptr<cv::Vec3b>(y);

    //    for (int x = 0; x < face2.width; ++x) {
    //        ptrDstFace2[face2.x + x] = ptrSrcFace1[x];
    //    }
    //}

#endif

    return 0;
}