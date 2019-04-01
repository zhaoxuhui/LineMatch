/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2014, Biagio Montesano, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/


#include "opencv2/line_descriptor.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <iostream>

#define MATCHES_DIST_THRESHOLD 25

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;

static const char *keys =
        {
                "{ 1 | | | path to image1 }"
                "{ 2 | | | path to image2 }"
        };

static void help() {
    std::cout << "\nThis example shows the functionalities of lines extraction "
              << "and descriptors computation furnished by BinaryDescriptor class\n"
              << "Please, run this sample using a command in the form\n"
              << "./example_line_descriptor_compute_descriptors <path_to_input_image 1> "
              << "<path_to_input_image 2>" << std::endl;
}

int main(int argc, const char **argv) {
    /* get parameters from command line */
    CommandLineParser parser(argc, argv, keys);
    String image_path1 = parser.get<String>("1");
    String image_path2 = parser.get<String>("2");

    if (image_path1.empty() || image_path2.empty()) {
        help();
        return -1;
    }

    /* load image */
    cv::Mat imageMat1 = imread(image_path1, IMREAD_COLOR);
    cv::Mat imageMat2 = imread(image_path2, IMREAD_COLOR);

    if (imageMat1.data == NULL || imageMat2.data == NULL) {
        std::cout << "Error, images could not be loaded. Please, check their path" << std::endl;
    }

    /* create binary masks */
    cv::Mat mask1 = Mat::ones(imageMat1.size(), CV_8UC1);
    cv::Mat mask2 = Mat::ones(imageMat2.size(), CV_8UC1);

    /* create a pointer to a BinaryDescriptor object with default parameters */
    Ptr<BinaryDescriptor> bd = BinaryDescriptor::createBinaryDescriptor();

    /* compute lines and descriptors */
    std::vector<KeyLine> keylines1, keylines2;
    cv::Mat descr1, descr2;

    (*bd)(imageMat1, mask1, keylines1, descr1, false, false);
    (*bd)(imageMat2, mask2, keylines2, descr2, false, false);

    /* select keylines from first octave and their descriptors */
    std::vector<KeyLine> lbd_octave1, lbd_octave2;
    Mat left_lbd, right_lbd;
    for (int i = 0; i < (int) keylines1.size(); i++) {
        if (keylines1[i].octave == 0) {
            lbd_octave1.push_back(keylines1[i]);
            left_lbd.push_back(descr1.row(i));
        }
    }

    for (int j = 0; j < (int) keylines2.size(); j++) {
        if (keylines2[j].octave == 0) {
            lbd_octave2.push_back(keylines2[j]);
            right_lbd.push_back(descr2.row(j));
        }
    }

    /* create a BinaryDescriptorMatcher object */
    Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

    /* require match */
    std::vector<DMatch> matches;
    bdm->match(left_lbd, right_lbd, matches);

    /* select best matches */
    std::vector<DMatch> good_matches;
    for (int i = 0; i < (int) matches.size(); i++) {
        if (matches[i].distance < MATCHES_DIST_THRESHOLD)
            good_matches.push_back(matches[i]);
    }

    /* plot matches */
    cv::Mat outImg;
    cv::Mat scaled1, scaled2;
    std::vector<char> mask(matches.size(), 1);
    drawLineMatches(imageMat1, lbd_octave1, imageMat2, lbd_octave2, good_matches, outImg, Scalar::all(-1),
                    Scalar::all(-1), mask,
                    DrawLinesMatchesFlags::DEFAULT);

    imwrite("matches.jpg", outImg);
    imshow("Matches", outImg);
    waitKey();
//    imwrite("/home/ubisum/Desktop/images/env_match/matches.jpg", outImg);

    /* create an LSD detector */
    Ptr<LSDDetector> lsd = LSDDetector::createLSDDetector();

    /* detect lines */
    std::vector<KeyLine> klsd1, klsd2;
    Mat lsd_descr1, lsd_descr2;
    lsd->detect(imageMat1, klsd1, 2, 2, mask1);
    lsd->detect(imageMat2, klsd2, 2, 2, mask2);

    /* compute descriptors for lines from first octave */
    bd->compute(imageMat1, klsd1, lsd_descr1);
    bd->compute(imageMat2, klsd2, lsd_descr2);

    /* select lines and descriptors from first octave */
    std::vector<KeyLine> octave0_1, octave0_2;
    Mat leftDEscr, rightDescr;
    for (int i = 0; i < (int) klsd1.size(); i++) {
        if (klsd1[i].octave == 1) {
            octave0_1.push_back(klsd1[i]);
            leftDEscr.push_back(lsd_descr1.row(i));
        }
    }

    for (int j = 0; j < (int) klsd2.size(); j++) {
        if (klsd2[j].octave == 1) {
            octave0_2.push_back(klsd2[j]);
            rightDescr.push_back(lsd_descr2.row(j));
        }
    }

    /* compute matches */
    std::vector<DMatch> lsd_matches;
    bdm->match(leftDEscr, rightDescr, lsd_matches);

    /* select best matches */
    good_matches.clear();
    for (int i = 0; i < (int) lsd_matches.size(); i++) {
        if (lsd_matches[i].distance < MATCHES_DIST_THRESHOLD)
            good_matches.push_back(lsd_matches[i]);
    }


//    for (int i = 0; i < good_matches.size(); ++i) {
//
//        DMatch dm = good_matches[i];
//        KeyLine left = keylines1[dm.queryIdx];
//        KeyLine right = keylines2[dm.trainIdx];
//        cout << "lsd left points" << endl;
//        cout << Point2f(left.sPointInOctaveX, left.sPointInOctaveY) << " "
//             << Point2f(left.ePointInOctaveX, left.ePointInOctaveY) << endl;
//        cout << "lsd right points" << endl;
//        cout << Point2f(right.sPointInOctaveX, right.sPointInOctaveY) << " "
//             << Point2f(right.ePointInOctaveX, right.ePointInOctaveY) << endl << endl;
//    }


    /* plot matches */
    cv::Mat lsd_outImg;
    resize(imageMat1, imageMat1, Size(imageMat1.cols / 2, imageMat1.rows / 2));
    resize(imageMat2, imageMat2, Size(imageMat2.cols / 2, imageMat2.rows / 2));
    std::vector<char> lsd_mask(matches.size(), 1);
    drawLineMatches(imageMat1, octave0_1, imageMat2, octave0_2, good_matches, lsd_outImg, Scalar::all(-1),
                    Scalar::all(-1), lsd_mask,
                    DrawLinesMatchesFlags::DEFAULT);
    imwrite("lsd_matches.jpg", lsd_outImg);
    imshow("LSD matches", lsd_outImg);
    waitKey();


}

