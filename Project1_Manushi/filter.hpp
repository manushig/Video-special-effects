/*! \file filter.hpp
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
int greyscale(Mat& src, Mat& dst);

/*!
 *  \brief Apply a sepia tone filter with vignetting to an image.
 *  \param src The source color image (input).
 *  \param dst The destination sepia-toned image (output).
 *  \return int Returns 0 on successful execution, -1 if source image is empty.
 *
 *  This function applies a sepia tone filter to the source image with vignetting effect.
 *  Vignetting makes the image get darker towards the edges.
 */
int sepia(Mat& src, Mat& dst);



/*!
 *  \brief Apply a guassian filter 5x5.
 *  \param src The source color image (input).
 *  \param dst The destination guassian filtered (output)
 *  \return int Returns 0 on successful execution, -1 if source image is empty.
 *
 *  This function applies a guassian filter to the source image 
 *  Applying this filter makes the image go blurr
 */
int blur5x5_1(cv::Mat& src, cv::Mat& dst);



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
int blur5x5_2(cv::Mat& src, cv::Mat& dst);



/*!
 *  \brief Apply a 3x3 Sobel X filter to a color image.
 *  \param src The source color image (input).
 *  \param dst The destination image with Sobel X filter applied (output).
 *  \return int Returns 0 on successful execution, -1 if source image is empty or not a 3-channel image.
 *
 *  This function applies a 3x3 Sobel X filter to the source image, emphasizing horizontal edges.
 *  The filter is positive to the right, and the output image is of type 16SC3 (signed short, 3 channels).
 */
int sobelX3x3(Mat& src, Mat& dst);

/*!
 *  \brief Apply a 3x3 Sobel Y filter to a color image.
 *  \param src The source color image (input).
 *  \param dst The destination image with Sobel Y filter applied (output).
 *  \return int Returns 0 on successful execution, -1 if source image is empty or not a 3-channel image.
 *
 *  This function applies a 3x3 Sobel Y filter to the source image, emphasizing vertical edges.
 *  The filter is positive upwards, and the output image is of type 16SC3 (signed short, 3 channels).
 */
int sobelY3x3(Mat& src, Mat& dst);

/**
 * @brief Generates a gradient magnitude image from X and Y Sobel images.
 *
 * @param sx Input image representing the Sobel X gradient (3-channel signed short).
 * @param sy Input image representing the Sobel Y gradient (3-channel signed short).
 * @param dst Output image representing the gradient magnitude (single-channel uchar).
 * @return int Returns 0 on success, -1 on failure.
 */
int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst);
    
/**
 * @brief Blurs and quantizes a color image.
 *
 * @param src Input image (color image).
 * @param dst Output image (blurred and quantized color image).
 * @param levels The number of levels to quantize each color channel.
 * @return int Returns 0 on success, -1 on failure.
 */
int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels);

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
void createEmbossEffect(const Mat& sobelX, const Mat& sobelY, Mat& emboss);


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
void colorFacesOnGray(cv::Mat& src, cv::Mat& dst, const std::vector<cv::Rect>& faces);



/**
 * @brief Adds a sparkle halo effect above detected faces in an image.
 *
 * This function modifies the pixels above each detected face to create a sparkle or halo effect.
 * It assumes that the face rectangles are correctly identified and uses pointer access for pixel manipulation.
 *
 * @param image The source image (color image).
 * @param faces A vector of rectangles representing detected faces.
 */
void addSparkleHalo(cv::Mat& image, const std::vector<cv::Rect>& faces);



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
void medianFilterPtrColor(const cv::Mat& src, cv::Mat& dst, int kernelSize);



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
void colorPreserveGray(cv::Mat& src, cv::Mat& dst, cv::Scalar& targetColor, int tolerance);



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
void adjustBrightness(cv::Mat& src, cv::Mat& dst, int brightness);



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
void adjustContrast(cv::Mat& src, cv::Mat& dst, double contrast);

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
void cartoonify(cv::Mat& src, cv::Mat& dst);


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
int swapFaces(cv::Mat& src, cv::Mat& dst, std::vector<cv::Rect>& faces);