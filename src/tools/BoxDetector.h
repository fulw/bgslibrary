//
// Created by levin on 2022/7/8.
//
#pragma once

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

namespace bgslibrary
{
    namespace tools
    {
        class Box
        {
        public:
            int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
            Box()= default;
            Box(int x1, int y1, int x2, int y2): x1(x1), y1(y1), x2(x2), y2(y2){}
            explicit Box(cv::Rect rect): Box(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height){}
            int distance(const Box &other) const;
            Box merge(const Box &other) const;
            void multiply(double scale);
            int width() const;
            int height() const;
        };

        class BoxDetector
        {
        public:
            BoxDetector(int mergeDist, int minSize);

            std::list<Box> Detect(const cv::Mat &mask);

            bool saveBoxes(std::list<Box> &boxes, std::string fpath);

        private:
            int mergeDist = 30;
            int minSize = 100;
            std::list<Box> mergeBoxes(std::list<Box>& boxes);
        };
    }
}
