//
// Created by levin on 2022/7/8.
//
#include <algorithm>
#include "BoxDetector.h"

using namespace bgslibrary::tools;

int Box::distance(const Box &other) const {
    int dw = std::max(x1, other.x1) - std::min(x2, other.x2);
    int dh = std::max(y1, other.y1) - std::min(y2, other.y2);
    return std::max(dw, dh);
}

int Box::width() const {
    return x2 - x1;
}

int Box::height() const {
    return y2 - y1;
}

Box Box::merge(const Box &other) const {
    return {
            std::min(x1, other.x1),
            std::min(y1, other.y1),
            std::max(x2, other.x2),
            std::max(y2, other.y2)
            };
}

BoxDetector::BoxDetector(int mergeDist, int minSize): mergeDist(mergeDist), minSize(minSize) {}

std::list<Box> BoxDetector::Detect(const cv::Mat &img) {
    // 使用open和dilate形态学运算对img二值图分别进行去噪和增强
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat mask;
    cv::morphologyEx(img, mask, cv::MORPH_OPEN, kernel);
    cv::dilate(mask, mask, kernel, cv::Point(-1,-1), 10);
    // 查找二值图的轮廓线
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    // 基于轮廓线拟合矩形
    std::vector<cv::Rect> rects;
    for (const auto& contour : contours) {
        std::vector<cv::Point> approx;
        // 拟合多边形
        cv::approxPolyDP(contour, approx, 2, true);
        // 计算多边形的外接矩形
        cv::Rect rect = cv::boundingRect(approx);
        rects.push_back(rect);
        // 这里要把rect重复添加到rects，否则接下来的cv::groupRectangles无法工作
        rects.push_back(rect);
    }
    // 基于矩形位置和大小对矩形进行合并
    cv::groupRectangles(rects, 1, 0.2);
    std::list<Box> boxes;
    for (const auto& rect : rects) {
        boxes.emplace_back(rect);
    }
    // 基于矩形位置进行融合
    boxes = mergeBoxes(boxes);
    // 移除边长小于minSize的矩形框
    boxes.remove_if([this](Box box){
        return box.width() < minSize || box.height() < minSize;
    });
    return boxes;
}

// 当box之间的距离小于mergeDist时，它们会被融合为一个矩形
// 使用list而不是vector，因为需要进行很多增删的操作，list更高效
std::list<Box> BoxDetector::mergeBoxes(std::list<Box>& boxes) {
    for (auto i = boxes.begin(); i != boxes.end(); ++i){
        for (auto j = boxes.begin(); j != boxes.end(); ++j){
            if (i == j)
                continue;
            // box i和j之间的距离
            if ((*i).distance(*j) < mergeDist) {
                // 移除box i和j
                // (这种移除方式是安全的，无论i和j哪个靠前，因为i和j在两个循环里，且只删除一次)
                boxes.erase(i);
                boxes.erase(j);
                // 添加i和j融合后的矩形
                boxes.push_front((*i).merge(*j));
                // 终止该阶段的融合，开启新的融合阶段——加入ij融合矩形
                return mergeBoxes(boxes);
            }
        }
    }
    return boxes;
}
