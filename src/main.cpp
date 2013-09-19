#include <iostream>
#include "functions.h"
#include "densetracker.h"

using namespace std;

int main( int argC, char* argV[] ) {
    dense_tracker::DenseTracker tracker;

    for (uint32_t i = 0; i < 1000; i++) {
        stringstream ss;
        ss << "/local/imaged/stixels/bahnhof/seq03-img-left/image_";
        ss.fill('0');
        ss.width(8);
        ss << i;
        ss << "_0.png";
        
        IplImage * img = cvLoadImage(ss.str().c_str());
        tracker.compute(img);
        
        cv::Mat denseTrack;
        tracker.drawTracks(denseTrack);
        cv::imshow("DenseTrack", denseTrack);
        int c = cvWaitKey(3);
        if ((char)c == 27) break;
        if ((char)c == 81) break;
        if ((char)c == 113) break;
        if ((char)c == 32) cvWaitKey(0);
        
        cvReleaseImage(&img);
    }
}