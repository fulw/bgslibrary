#include <iostream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>

#include "../src/algorithms/algorithms.h"
#include "../src/tools/BoxDetector.h"


#if CV_MAJOR_VERSION >= 4
#define CV_CAP_PROP_POS_FRAMES cv::CAP_PROP_POS_FRAMES
#define CV_CAP_PROP_FRAME_COUNT cv::CAP_PROP_FRAME_COUNT
#endif

double duration = 0.0;

void tic() {
    duration = static_cast<double>(cv::getTickCount());
}

double toc() {
    duration = (static_cast<double>(cv::getTickCount()) - duration) / cv::getTickFrequency();
    return duration;
}

double fps() {
    return 1 / duration;
}

bool mkdir(std::string dir) {
    const int dir_err = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (-1 == dir_err) {
        return false;
    }
    return true;
}

bool hasEnding(std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

std::string size2string(int width, int height) {
    return std::to_string(width) + "x" + std::to_string(height);
}

bool saveList(std::vector<std::string>& items, std::string fpath) {
    if (items.size() == 0) {
        return true;
    }
    std::ofstream file(fpath);
    if (file.is_open())
    {
        for (auto item : items) {
            file << item << "\n";
        }
        file.close();
        return true;
    }
    return false;
}

std::string str(double v) {
    return std::to_string(v);
}

std::string str(int v) {
    return std::to_string(v);
}

std::vector<std::string> split(std::string s, char delim) {
    std::stringstream ss(s);
    std::string seg;
    std::vector<std::string> seglist;
    while(std::getline(ss, seg, delim)){
        if (!seg.empty()) {
            seglist.push_back(seg);
        }
    }
    return seglist;
}

bool canInputGray(std::string alg) {
    // 是否可以使用灰度图
    return (alg != "TwoPoints" && alg != "SigmaDelta" && alg != "ViBe");
}

bool canIgnore(std::string alg) {
    // 效果不好或太慢，忽略不用
    return (alg == "LOBSTER" || alg == "PAWCS" || alg == "SuBSENSE" || alg == "PixelBasedAdaptiveSegmenter");
}

int main(int argc, char **argv) {
    std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;
    cv::VideoCapture capture;
    std::vector<std::string> algorithmsName;
    std::string saveDir = "";
    int frameStep = 1;
    bool show = false;
    if (argc > 1) {
        std::cout << "Openning: " << argv[1] << std::endl;
        capture.open(argv[1]);
        if (argc > 2) {
            std::string algs = argv[2];
            algorithmsName = split(algs, ',');
            if (argc > 3) {
                saveDir = argv[3];
                if (!hasEnding(saveDir, "/")) {
                    saveDir = saveDir + "/";
                }
                if (argc > 4) {
                    frameStep = atoi(argv[4]);
                    if (argc > 5) {
                        show = argv[5];
                    }
                }
            }
        }
    } else {
        capture.open(0);
    }

    if (!capture.isOpened()) {
        std::cerr << "Cannot initialize video!" << std::endl;
        return -1;
    }

    int width_org = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int height_org = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    double scale = 1.0 / 4;
    int width = width_org * scale;
    int height = height_org * scale;
    std::cout << "Resize: " << width_org << "," << height_org << " -> " << width << "," << height << std::endl;

    /* Background Subtraction Methods */
    if (algorithmsName.size() == 0) {
        algorithmsName = BGS_Factory::Instance()->GetRegisteredAlgorithmsName();
    }
    std::cout << "List of algorithms:" << std::endl;
    std::copy(algorithmsName.begin(), algorithmsName.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
    // 解析mask为box
    bgslibrary::tools::BoxDetector boxDetector(50, 30);
    bgslibrary::tools::BoxDetector boxDetector_org(50, 100);

    for (const std::string &algorithmName: algorithmsName) {
        if (canIgnore(algorithmName)) {
            continue;
        }

        std::string saveDir_ = saveDir + algorithmName;
        std::string saveDir_rsz = saveDir_ + "/" + size2string(width, height) + "_" + str(frameStep);
        std::string saveDir_org = saveDir_ + "/" + size2string(width_org, height_org) + "_" + str(frameStep);
        if (!saveDir.empty()) {
            mkdir(saveDir_);
            mkdir(saveDir_rsz);
            mkdir(saveDir_org);
        }

        auto bgs = BGS_Factory::Instance()->Create(algorithmName);
        bgs->setShowOutput(false);
        auto bgs_org = BGS_Factory::Instance()->Create(algorithmName);
        bgs_org->setShowOutput(false);

        cv::Mat imgInput_org;
        cv::Mat imgInput;

        cv::Mat grayImgInput_org;
        cv::Mat grayImgInput;

        auto frameCounter = 0;
        double processSum = 0.0;
        double colorSum = 0.0;
        double boxSum = 0.0;
        double resizeSum = 0.0;

        double processSum_org = 0.0;
        double colorSum_org = 0.0;
        double boxSum_org = 0.0;

        std::vector<std::string> fpsList;
        std::vector<std::string> fpsList_org;

        capture.set(CV_CAP_PROP_POS_FRAMES, 0); // Set index to 0 (start frame)
        while (true) {
            // Capture frame-by-frame
            capture >> imgInput_org;
            if (imgInput_org.empty()) {
                break;
            }

            frameCounter += 1;

            // 跳帧处理，帧索引从0开始
            if ((frameCounter - 1) % frameStep != 0) {
                continue;
            }

            cv::Mat imgMask;
            cv::Mat imgBkgmodel;
            cv::Mat imgYuv;
            cv::Mat imgBgr;

            tic();
            cv::resize(imgInput_org, imgInput, cv::Size(width, height), 0, 0, CV_INTER_LINEAR);
            resizeSum += toc();

            cv::cvtColor(imgInput, imgYuv, cv::COLOR_BGR2YUV);
            tic();
            cv::cvtColor(imgYuv, imgBgr, cv::COLOR_YUV2BGR);
            colorSum += toc();

            cv::cvtColor(imgInput, grayImgInput, cv::COLOR_BGR2GRAY);
            auto imgInput_ = canInputGray(algorithmName) ? grayImgInput : imgInput;
            tic();
            bgs->process(imgInput_, imgMask,imgBkgmodel);
            processSum += toc();

            tic();
            auto boxes = boxDetector.Detect(imgMask);
            for (auto& box: boxes) {
                box.multiply(1.0/scale);
            }
            boxSum += toc();

            double fpsOverall = frameCounter / (resizeSum + colorSum + processSum + boxSum);

            // 原始大小的图片
            cv::Mat imgMask_org;
            cv::Mat imgBkgmodel_org;
            cv::Mat imgYuv_org;
            cv::Mat imgBgr_org;

            cv::cvtColor(imgInput_org, imgYuv_org, cv::COLOR_BGR2YUV);
            tic();
            cv::cvtColor(imgYuv_org, imgBgr_org, cv::COLOR_YUV2BGR);
            colorSum_org += toc();

            cv::cvtColor(imgInput_org, grayImgInput_org, cv::COLOR_BGR2GRAY);
            auto imgInput_org_ = canInputGray(algorithmName) ? grayImgInput_org : imgInput_org;
            tic();
            bgs_org->process(imgInput_org_, imgMask_org,imgBkgmodel_org);
            processSum_org += toc();

            tic();
            auto boxes_org = boxDetector_org.Detect(imgMask_org);
            boxSum_org += toc();

            double fpsOverall_org = frameCounter / (colorSum_org + processSum_org + boxSum_org);

            if (!saveDir.empty()) {
                boxDetector.saveBoxes(boxes, saveDir_rsz + "/" + str(frameCounter-1) + ".txt");
                boxDetector_org.saveBoxes(boxes_org, saveDir_org + "/" + str(frameCounter-1) + ".txt");
                fpsList.push_back(str(fpsOverall) + " " + str(frameCounter/colorSum)  + " " + str(frameCounter/processSum) + " " + str(frameCounter/boxSum) + " " + str(frameCounter/resizeSum));
                fpsList_org.push_back(str(fpsOverall_org) + " " + str(frameCounter/colorSum_org)  + " " + str(frameCounter/processSum_org) + " " + str(frameCounter/boxSum_org));
            }

            if (show) {
                for (auto& box : boxes_org) {
                    auto pt1 = cv::Point(box.x1, box.y1);
                    auto pt2 = cv::Point(box.x2, box.y2);
                    cv::rectangle(imgInput_org, pt1, pt2, cv::Scalar(255, 0, 0), 3);
                }
                for (auto& box : boxes) {
                    auto pt1 = cv::Point((int) (box.x1), (int) (box.y1));
                    auto pt2 = cv::Point((int) (box.x2), (int) (box.y2));
                    cv::rectangle(imgInput_org, pt1, pt2, cv::Scalar(0, 0, 255), 3);
                }
                cv::putText(imgInput_org,
                            "Algorithm: " + algorithmName, //text
                            cv::Point(230, 20), //top-left position
                            cv::FONT_HERSHEY_DUPLEX,
                            1.0,
                            cv::Scalar(255, 0, 255), //font color
                            2);
                cv::putText(imgInput_org,
                            "Blue  box: from original image " + std::to_string(width_org) + " x " + std::to_string(height_org) + " FPS: " + std::to_string(fpsOverall_org),
                            cv::Point(230, 60), //top-left position
                            cv::FONT_HERSHEY_DUPLEX,
                            1.0,
                            cv::Scalar(255, 0, 0), //font color
                            2);
                cv::putText(imgInput_org,
                            "Red   box: from resized image " + std::to_string(width) + "  x  " + std::to_string(height) + " FPS: " + std::to_string(fpsOverall),
                            cv::Point(230, 100), //top-left position
                            cv::FONT_HERSHEY_DUPLEX,
                            1.0,
                            cv::Scalar(0, 0, 255), //font color
                            2);
                cv::imshow("input", imgInput_org);
                cv::waitKey(30);
            }
        }

        double avg_fps = frameCounter / (resizeSum + colorSum + processSum + boxSum);
        std::cout << "FPS of resized frame:" << std::endl;
        std::cout << " resize avg fps: " << frameCounter / resizeSum << std::endl;
        std::cout << "  color avg fps: " << frameCounter / colorSum << std::endl;
        std::cout << "process avg fps: " << frameCounter / processSum << std::endl;
        std::cout << "    box avg fps: " << frameCounter / boxSum << std::endl;
        std::cout << "overall avg fps: " << avg_fps << std::endl;
        double avg_fps_org = frameCounter / (colorSum_org + processSum_org + boxSum_org);
        std::cout << "FPS of original frame:" << std::endl;
        std::cout << "  color avg fps: " << frameCounter / colorSum_org << std::endl;
        std::cout << "process avg fps: " << frameCounter / processSum_org << std::endl;
        std::cout << "    box avg fps: " << frameCounter / boxSum_org << std::endl;
        std::cout << "overall avg fps: " << avg_fps_org << std::endl;

        if (!saveDir.empty()) {
            fpsList.push_back(str(avg_fps) + " " + str(frameCounter/colorSum)  + " " + str(frameCounter/processSum) + " " + str(frameCounter/boxSum) + " " + str(frameCounter/resizeSum));
            fpsList_org.push_back(str(avg_fps_org) + " " + str(frameCounter/colorSum_org)  + " " + str(frameCounter/processSum_org) + " " + str(frameCounter/boxSum_org));
            saveList(fpsList, saveDir_rsz + "/fps.txt");
            saveList(fpsList_org, saveDir_org + "/fps.txt");
        }

        cv::destroyAllWindows();
    }

    capture.release();

    return 0;
}
