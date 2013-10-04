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
    int compute(const cv::Mat & frame);
    void drawTracks(cv::Mat & output);
    
    cv::Point2i getPrevPoint(const cv::Point2i & currPoint);
    cv::Point2i getCurrPoint(const cv::Point2i & prevPoint);
    
    IplImage * getImage() { return m_image; }
protected:
    // parameters for descriptors
    int m_patch_size;
    int m_nxy_cell;
    int m_nt_cell;
    bool m_fullOrientation;
    float m_epsilon;
    float m_min_flow;
    
    // parameters for tracking
    int m_start_frame;
    int m_end_frame;
    double m_quality;
    double m_min_distance;
    int m_init_gap;
    int m_track_length;
    
    // parameters for the trajectory descriptor
    float m_min_var;
    float m_max_var;
    float m_max_dis;
    
    // parameters for multi-scale
    int m_scale_num;  
    float m_scale_stride;
    
    int m_frameNum;
    int m_init_counter;
    std::vector<std::list<Track> > m_xyScaleTracks;
    
    TrackerInfo m_tracker;
    DescInfo m_hogInfo;
    DescInfo m_hofInfo;
    DescInfo m_mbhInfo;
    
    // Images
    IplImageWrapper m_image, m_prev_image, m_grey, m_prev_grey;
    IplImagePyramid m_grey_pyramid, m_prev_grey_pyramid, m_eig_pyramid;
    
    // I/O
    CvCapture* m_capture;
    float* m_fscales; // float scale values
    int m_show_track; // set show_track = 1, if you want to visualize the trajectories
    
    // Determines if a given point has been tracked
    std::vector<int> m_status;
    cv::Mat m_correspondencesPrev2Curr;
    cv::Mat m_correspondencesCurr2Prev;
    
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
