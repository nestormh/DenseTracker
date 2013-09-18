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
        
        cvShowImage( "DenseTrack", tracker.getImage());
        int c = cvWaitKey(3);
        if((char)c == 27) break;
        
        cvReleaseImage(&img);
    }
}