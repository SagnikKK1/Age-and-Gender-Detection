#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <tuple>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <iterator>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

tuple<Mat, vector<vector<int>>> getFaceBox(Net net, Mat& frame, double conf_threshold)
{
    Mat frameOpenCVDNN = frame.clone();
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;
    double inScaleFactor = 1.0;
    Size size = Size(300, 300);
    Scalar meanVal = Scalar(104, 117, 123);

    cv::Mat inputBlob;
    inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, size, meanVal, true, false);

    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    vector<vector<int>> bboxes;

    for (int i = 0; i < detectionMat.rows; i++) {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > conf_threshold) {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
            vector<int> box = { x1, y1, x2, y2 };
            bboxes.push_back(box);
            cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255));
        }
    }

    return make_tuple(frameOpenCVDNN, bboxes);
}

int main()
{
    string choice;
    cout << "Enter 'image' for processing an image or 'video' for processing a video: ";
    cin >> choice;

    if (choice != "image" && choice != "video") {
        cout << "Invalid choice. Please enter 'image' or 'video'." << endl;
        return -1;
    }

    string inputPath, outputPath;
    cout << "Enter input file path: ";
    cin.ignore(); // Ignore newline character from previous cin
    getline(cin, inputPath);

    Net ageNet = readNet("age_net.caffemodel", "age_deploy.prototxt");
    Net genderNet = readNet("gender_net.caffemodel", "gender_deploy.prototxt");
    Net faceNet = readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt");

    Scalar MODEL_MEAN_VALUES = Scalar(78.4263377603, 87.7689143744, 114.895847746);
    vector<string> ageList = { "(0-2)","(4-6)","(8-12)","(15-20)","(25-32)","(38-43)","(48-53)","(60-100)" };
    vector<string> genderList = { "Male","Female" };

    if (choice == "image") {
        cout << "Enter output file path: ";
        getline(cin, outputPath);
        Mat image = imread(inputPath);
        if (image.empty()) {
            cout << "Error: Unable to open image file." << endl;
            return -1;
        }

        vector<vector<int>> bboxes;
        Mat frameFace;
        tie(frameFace, bboxes) = getFaceBox(faceNet, image, 0.7);

        if (bboxes.size() == 0) {
            cout << "No face detected in the image." << endl;
            return -1;
        }

        for (auto it = begin(bboxes); it != end(bboxes); ++it) {
            Rect rec(it->at(0), it->at(1), it->at(2) - it->at(0), it->at(3) - it->at(1));
            if (rec.x >= 0 && rec.y >= 0 && (rec.width + rec.x) <= image.cols && (rec.height + rec.y) <= image.rows) {
                Mat face = image(rec);

                Mat blob = blobFromImage(face, 1, Size(227, 227), MODEL_MEAN_VALUES, false);

                genderNet.setInput(blob);
                vector<float> genderPreds = genderNet.forward();
                int max_index_gender = std::distance(genderPreds.begin(), max_element(genderPreds.begin(), genderPreds.end()));
                string gender = genderList[max_index_gender];


                ageNet.setInput(blob);
                vector<float> agePreds = ageNet.forward();
                int max_indice_age = std::distance(agePreds.begin(), max_element(agePreds.begin(), agePreds.end()));
                string age = ageList[max_indice_age];

                cout << "Gender: " << gender << "  Age: " << age << endl;

                string label = gender + ", " + age;
                cv::putText(frameFace, label, Point(it->at(0), it->at(1) - 15), cv::FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255));
            }
        }

        imwrite(outputPath, frameFace);
        cout << "Processed image saved successfully." << endl;
    }
    else if (choice == "video") {
        cout << "Enter output file path: ";
        getline(cin, outputPath);
        VideoCapture cap(inputPath);
        if (!cap.isOpened()) {
            cout << "Error opening video stream or file" << endl;
            return -1;
        }

        int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
        int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
        VideoWriter video(outputPath, VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height));

        while (true) {
            Mat frame;
            cap >> frame;
            if (frame.empty())
                break;

            vector<vector<int>> bboxes;
            Mat frameFace;
            tie(frameFace, bboxes) = getFaceBox(faceNet, frame, 0.7);

            if (bboxes.size() == 0) {
                cout << "No face detected in this frame." << endl;
            }
            else {
                for (auto it = begin(bboxes); it != end(bboxes); ++it) {
                    Rect rec(it->at(0), it->at(1), it->at(2) - it->at(0), it->at(3) - it->at(1));
                    if (rec.x >= 0 && rec.y >= 0 && (rec.width + rec.x) <= frame.cols && (rec.height + rec.y) <= frame.rows) {
                        Mat face = frame(rec);

                        Mat blob = blobFromImage(face, 1, Size(227, 227), MODEL_MEAN_VALUES, false);

                        genderNet.setInput(blob);
                        vector<float> genderPreds = genderNet.forward();
                        int max_index_gender = std::distance(genderPreds.begin(), max_element(genderPreds.begin(), genderPreds.end()));
                        string gender = genderList[max_index_gender];

                        ageNet.setInput(blob);
                        vector<float> agePreds = ageNet.forward();
                        int max_indice_age = std::distance(agePreds.begin(), max_element(agePreds.begin(), agePreds.end()));
                        string age = ageList[max_indice_age];

                        cout << "Gender: " << gender << "  Age: " << age << endl;

                        string label = gender + ", " + age;
                        cv::putText(frameFace, label, Point(it->at(0), it->at(1) - 15), cv::FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255));
                    }
                }
                video.write(frameFace);
            }
        }
    }
}
