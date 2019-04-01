//
// Created by root on 19-4-1.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "opencv2/line_descriptor.hpp"

using namespace std;
using namespace cv;

#define DISTANCE_THRESHOLD 25

void LBDMatch(String img_path1, String img_path2) {
    // 二进制检测子+二进制描述子

    Mat img1 = imread(img_path1);
    Mat img2 = imread(img_path2);

    Ptr<line_descriptor::BinaryDescriptor> bd = line_descriptor::BinaryDescriptor::createBinaryDescriptor();
    vector<line_descriptor::KeyLine> keylines1, keylines2;
    Mat des1, des2;

    // 检测线段
    bd->detect(img1, keylines1);
    bd->detect(img2, keylines2);

    // 计算二进制描述子
    bd->compute(img1, keylines1, des1);
    bd->compute(img2, keylines2, des2);

    // 二进制匹配
    Ptr<line_descriptor::BinaryDescriptorMatcher> bdm = line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    vector<DMatch> matches;
    bdm->match(des1, des2, matches);

    // 筛选好的匹配，距离小于给定阈值
    vector<DMatch> good_matches;
    for (int i = 0; i < matches.size(); ++i) {
        if (matches[i].distance < DISTANCE_THRESHOLD) {
            good_matches.push_back(matches[i]);
        }
    }

    // 绘制线段
    Mat img1_lines = Mat::zeros(img1.size(), CV_8UC3);
    line_descriptor::drawKeylines(img1, keylines1, img1_lines);
    imwrite("img1_binary_lines.jpg", img1_lines);

    Mat img2_lines = Mat::zeros(img2.size(), CV_8UC3);
    line_descriptor::drawKeylines(img2, keylines2, img2_lines);
    imwrite("img2_binary_lines.jpg", img2_lines);

    // 绘制匹配的线段
    Mat matched_img;
    std::vector<char> mask(good_matches.size(), 1);
    line_descriptor::drawLineMatches(img1, keylines1,
                                     img2, keylines2,
                                     good_matches,
                                     matched_img,
                                     Scalar::all(-1),
                                     Scalar::all(-1), mask,
                                     line_descriptor::DrawLinesMatchesFlags::DEFAULT);
    imwrite("match_binary_lines.jpg", matched_img);
}

void LSDMatch(String img_path1, String img_path2) {
    // LSD检测子+二进制描述子

    Mat img1 = imread(img_path1);
    Mat img2 = imread(img_path2);

    // LSD检测器
    Ptr<line_descriptor::LSDDetector> lsd = line_descriptor::LSDDetector::createLSDDetector();
    // 二进制描述子
    Ptr<line_descriptor::BinaryDescriptor> bd = line_descriptor::BinaryDescriptor::createBinaryDescriptor();
    vector<line_descriptor::KeyLine> keylines1, keylines2;
    Mat des1, des2;

    // 检测线段
    lsd->detect(img1, keylines1, 2, 2);
    lsd->detect(img2, keylines2, 2, 2);

    // 描述线段
    bd->compute(img1, keylines1, des1);
    bd->compute(img2, keylines2, des2);

    // 选择金字塔第一层的线段以及对应描述子
    vector<line_descriptor::KeyLine> keylines1_octave0, keylines2_octave0;
    Mat des1_octave0, des2_octave0;
    for (int i = 0; i < keylines1.size(); ++i) {
        if (keylines1[i].octave == 1) {
            keylines1_octave0.push_back(keylines1[i]);
            des1_octave0.push_back(des1.row(i));
        }
    }
    for (int i = 0; i < keylines2.size(); ++i) {
        if (keylines2[i].octave == 1) {
            keylines2_octave0.push_back(keylines2[i]);
            des2_octave0.push_back(des2.row(i));
        }
    }

    // 绘制线段
    Mat img1_lines, img2_lines;
    Mat img1_lines_oct0, img2_lines_oct0;
    line_descriptor::drawKeylines(img1, keylines1, img1_lines);
    line_descriptor::drawKeylines(img2, keylines2, img2_lines);
    line_descriptor::drawKeylines(img1, keylines1_octave0, img1_lines_oct0);
    line_descriptor::drawKeylines(img2, keylines2_octave0, img2_lines_oct0);
    imwrite("img1_lsd_lines.jpg", img1_lines);
    imwrite("img2_lsd_lines.jpg", img2_lines);
    imwrite("img1_lsd_lines_oct0.jpg", img1_lines_oct0);
    imwrite("img2_lsd_lines_oct0.jpg", img2_lines_oct0);

    // 匹配线段
    vector<DMatch> matches, good_matches;
    Ptr<line_descriptor::BinaryDescriptorMatcher> bdm = line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    bdm->match(des1_octave0, des2_octave0, matches);

    for (int j = 0; j < matches.size(); ++j) {
        if (matches[j].distance < DISTANCE_THRESHOLD) {
            good_matches.push_back(matches[j]);
        }
    }

    // 绘制匹配的线段
    Mat matched_img;
    // lsd的匹配结果是缩小了一倍的，所以绘图的时候也要对应缩小
    resize(img1, img1, Size(img1.cols / 2, img1.rows / 2));
    resize(img2, img2, Size(img2.cols / 2, img2.rows / 2));
    std::vector<char> mask(good_matches.size(), 1);
    line_descriptor::drawLineMatches(img1, keylines1_octave0,
                                     img2, keylines2_octave0,
                                     good_matches,
                                     matched_img,
                                     Scalar::all(-1),
                                     Scalar::all(-1), mask,
                                     line_descriptor::DrawLinesMatchesFlags::DEFAULT);
    imwrite("match_lsd_lines.jpg", matched_img);
}

int main() {
    String img_path1 = "img1.png";
    String img_path2 = "img2.png";

    LBDMatch(img_path1, img_path2);
    LSDMatch(img_path1, img_path2);
}