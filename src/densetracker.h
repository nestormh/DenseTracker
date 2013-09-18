/*
 *  Copyright 2013 Néstor Morales Hernández <nestor@isaatc.ull.es>
 * 
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#ifndef DENSETRACKER_H
#define DENSETRACKER_H

#include "opencv/IplImageWrapper.h"
#include "opencv/IplImagePyramid.h"
#include "descriptors_structures.h"

namespace dense_tracker {
    
typedef struct TrackerInfo
{
    int trackLength; // length of the trajectory
    int initGap; // initial gap for feature detection
}TrackerInfo;

class DenseTracker
{

public:
    DenseTracker();
    virtual ~DenseTracker();
    
    int loop();
    
    int compute(IplImage * frame);
    
    IplImage * getImage() { return image; }
protected:
    // parameters for descriptors
    int patch_size;
    int nxy_cell;
    int nt_cell;
    bool fullOrientation;
    float epsilon;
    float min_flow;
    
    // parameters for tracking
    int start_frame;
    int end_frame;
    double quality;
    double min_distance;
    int init_gap;
    int track_length;
    
    // parameters for the trajectory descriptor
    float min_var;
    float max_var;
    float max_dis;
    
    // parameters for multi-scale
    int scale_num;  
    float scale_stride;
    
    int frameNum;
    int init_counter;
    std::vector<std::list<Track> > xyScaleTracks;
    
    TrackerInfo tracker;
    DescInfo hogInfo;
    DescInfo hofInfo;
    DescInfo mbhInfo;
    
    // Images
    IplImageWrapper image, prev_image, grey, prev_grey;
    IplImagePyramid grey_pyramid, prev_grey_pyramid, eig_pyramid;
    
    // I/O
    CvCapture* capture;
    float* fscales; // float scale values
    int show_track; // set show_track = 1, if you want to visualize the trajectories
    
    // METHODS SECTION
    void InitTrackerInfo(TrackerInfo* tracker, int track_length, int init_gap);
    DescMat* InitDescMat(int height, int width, int nBins);
    void ReleDescMat( DescMat* descMat);
    void InitDescInfo(DescInfo* descInfo, int nBins, int flag, int orientation, int size, int nxy_cell, int nt_cell);
    CvScalar getRect(const CvPoint2D32f point, const CvSize size, const DescInfo descInfo);
    void BuildDescMat(const IplImage* xComp, const IplImage* yComp, DescMat* descMat, const DescInfo descInfo);
    std::vector<float> getDesc(const DescMat* descMat, CvScalar rect, DescInfo descInfo);
    void HogComp(IplImage* img, DescMat* descMat, DescInfo descInfo);
    void HofComp(IplImage* flow, DescMat* descMat, DescInfo descInfo);
    void MbhComp(IplImage* flow, DescMat* descMatX, DescMat* descMatY, DescInfo descInfo);
    void OpticalFlowTracker(IplImage* flow, std::vector<CvPoint2D32f>& points_in, 
                            std::vector<CvPoint2D32f>& points_out, std::vector<int>& status);
    int isValid(std::vector<CvPoint2D32f>& track, float& mean_x, float& mean_y, float& var_x, float& var_y, float& length);
    void cvDenseSample(IplImage* grey, IplImage* eig, std::vector<CvPoint2D32f>& points,
                       const double quality, const double min_distance);
    void cvDenseSample(IplImage* grey, IplImage* eig, std::vector<CvPoint2D32f>& points_in,
                       std::vector<CvPoint2D32f>& points_out, const double quality, const double min_distance);
};

}

#endif // DENSETRACKER_H
