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
//  */

#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <omp.h>

#include <iostream>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/concept_check.hpp>
#include <tiff.h>

#include "densetracker.h"

using namespace std;

namespace dense_tracker {

DenseTracker::DenseTracker(const t_Tracking trackingType)
{
    // parameters for descriptors
    m_patch_size = 32;
    m_nxy_cell = 2;
    m_nt_cell = 3;
    m_fullOrientation = true;
    m_epsilon = 0.05;
    m_min_flow = 0.4f * 0.4f;
    
    // parameters for tracking
    m_start_frame = 0;
    m_end_frame = 1000000;
    m_quality = 0.001;
    m_min_distance = 1; //5;
    m_init_gap = 1;
    m_track_length = 1;
    
    // parameters for the trajectory descriptor
    m_min_var = sqrt(3);
    m_max_var = 50;
    m_max_dis = 20;
    
    // parameters for multi-scale
    m_scale_num = 8;
    m_scale_stride = sqrt(2);
    
    // I/O
    m_capture = 0;
    m_fscales = 0; // float scale values
    m_show_track = 0; // set m_show_track = 1, if you want to visualize the trajectories
    
    InitTrackerInfo(&m_tracker, m_track_length, m_init_gap);
    InitDescInfo(&m_hogInfo, 8, 0, 1, m_patch_size, m_nxy_cell, m_nt_cell);
    InitDescInfo(&m_hofInfo, 9, 1, 1, m_patch_size, m_nxy_cell, m_nt_cell);
    InitDescInfo(&m_mbhInfo, 8, 0, 1, m_patch_size, m_nxy_cell, m_nt_cell);
    
    m_frameNum = 0;
    
    m_trackingType = trackingType;
}

DenseTracker::~DenseTracker()
{

}

int DenseTracker::loop()
{
    m_frameNum = 0;

    char* video = "/home/nestor/Dropbox/projects/DenseTracker/examples/person01_boxing_d1_uncomp.avi";
//     arg_parse(argc, argv);

    //      std::cerr << "m_start_frame: " << m_start_frame << " m_end_frame: " << m_end_frame << " m_track_length: " << m_track_length << std::endl;
    //      std::cerr << "m_min_distance: " << m_min_distance << " m_patch_size: " << m_patch_size << " m_nxy_cell: " << m_nxy_cell << " m_nt_cell: " << m_nt_cell << std::endl;

    m_capture = cvCreateFileCapture(video);

    if( !m_capture ) {
        printf( "Could not initialize capturing..\n" );
        return -1;
    }

    if( m_show_track == 1 )
        cvNamedWindow( "DenseTrack", 0 );

    m_init_counter = 0; // indicate when to detect new feature points
    while( true ) {
        IplImage* frame = 0;

        // get a new frame
        frame = cvQueryFrame( m_capture );
        if( !frame ) {
            //printf("break");
            break;
        }
        if( m_frameNum >= m_start_frame && m_frameNum <= m_end_frame ) {
            compute(frame);
        }

        if( m_show_track == 1 ) {
            cvShowImage( "DenseTrack", m_image);
            int c = cvWaitKey(3);
            if((char)c == 27) break;
        }
        // get the next frame
    }

    if( m_show_track == 1 )
        cvDestroyWindow("DenseTrack");
}

int DenseTracker::compute(const cv::Mat & frame)
{
    switch(m_trackingType) {
        case TRACKING_FARNEBACK: {
            computeFarneback(frame);
            break;
        }
        case TRACKING_LEAR: {
            computeLEAR(frame);
            break;
        }
        case TRACKING_LK: {
            computeLK(frame);
            break;
        }
    }
    
    return 0;
}

int DenseTracker::computeFarneback(const cv::Mat & frame)
{
    if (m_prevImgMat.empty()) {
        m_prevImgMat = cv::Mat(frame.rows, frame.cols, CV_8UC3);
        m_correspondencesPrev2Curr = cv::Mat(frame.rows, frame.cols, CV_32SC2);
        m_correspondencesCurr2Prev = cv::Mat(frame.rows, frame.cols, CV_32SC2);
        m_correspondencesPrev2Curr.setTo(cv::Vec2i(-1, -1));
        m_correspondencesCurr2Prev.setTo(cv::Vec2i(-1, -1));
        m_image = cvCreateImage(cvSize(frame.cols, frame.rows), IPL_DEPTH_8U, 3);
    }
 
    cv::Mat flow(frame.rows, frame.cols, CV_32FC2);
    cv::Mat prevGray, currGray;
    cv::cvtColor(frame, currGray, CV_BGR2GRAY);
    cv::cvtColor(m_prevImgMat, prevGray, CV_BGR2GRAY);
    cv::calcOpticalFlowFarneback(prevGray, currGray, flow,
                                            0.5, 1, 5, 2, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN );
//                                     sqrt(2)/2.0, 5, 10, 2, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN );
    for (uint32_t y = 0; y < flow.rows; y++) {
        for (uint32_t x = 0; x < flow.cols; x++) {
            const cv::Point2i pointCurr(x, y);
            const cv::Point2i pointFlow = cv::Point2i(flow.at<cv::Vec2f>(y, x)[0], flow.at<cv::Vec2f>(y, x)[1]);
            const cv::Point2i pointPrev = pointCurr + pointFlow;
            if ((pointPrev.x >= 0) || (pointPrev.y >= 0) ||
                (pointPrev.x < flow.cols) || (pointPrev.y < flow.rows)) {
                
                m_correspondencesCurr2Prev.at<cv::Vec2i>(pointCurr.y, pointCurr.x) = cv::Vec2i(pointPrev.x, pointPrev.y);
            m_correspondencesPrev2Curr.at<cv::Vec2i>(pointPrev.y, pointPrev.x) = cv::Vec2i(pointCurr.x, pointCurr.y);
                }
        }
    }
    
//     cvCopy( frame, m_prev_image, 0 );
    frame.copyTo(m_prevImgMat);
//     frame.copyTo(m_image);
    memcpy((void *)m_image->imageData, (void *)frame.data, frame.cols * frame.rows * frame.channels() * sizeof(uchar));
//     m_image = m_prev_image;
    
    m_frameNum++;
    
    return 0;
}

int DenseTracker::computeLEAR(const cv::Mat & frame)
{
    
    IplImage * img = cvCreateImage(cvSize(frame.cols, frame.rows), IPL_DEPTH_8U, 3);
    memcpy((void *)img->imageData, (void *)frame.data, frame.cols * frame.rows * frame.channels() * sizeof(uchar));
    
    compute(img);
    
    cvReleaseImage(&img);
    
    return 0;
}

int DenseTracker::computeLEAR(IplImage* frame)
{
    double cp0 = omp_get_wtime();
    int i, j;
    if( ! m_image ) {
        m_init_counter = 0; // indicate when to detect new feature points

        // initailize all the buffers
        m_image = IplImageWrapper( cvGetSize(frame), 8, 3 );
        m_image->origin = frame->origin;
        m_prev_image= IplImageWrapper( cvGetSize(frame), 8, 3 );
        m_prev_image->origin = frame->origin;
        m_grey = IplImageWrapper( cvGetSize(frame), 8, 1 );
        m_grey_pyramid = IplImagePyramid( cvGetSize(frame), 8, 1, m_scale_stride );
        m_prev_grey = IplImageWrapper( cvGetSize(frame), 8, 1 );
        m_prev_grey_pyramid = IplImagePyramid( cvGetSize(frame), 8, 1, m_scale_stride );
        m_eig_pyramid = IplImagePyramid( cvGetSize(frame), 32, 1, m_scale_stride );
        
        m_correspondencesPrev2Curr = cv::Mat(frame->height, frame->width, CV_32SC2);
        m_correspondencesCurr2Prev = cv::Mat(frame->height, frame->width, CV_32SC2);
        
        cvCopy( frame, m_image, 0 );
        cvCvtColor( m_image, m_grey, CV_BGR2GRAY );
        m_grey_pyramid.rebuild( m_grey );
        
        // how many scale we can have
        m_scale_num = std::min<std::size_t>(m_scale_num, m_grey_pyramid.numOfLevels());
        m_fscales = (float*)cvAlloc(m_scale_num*sizeof(float));
        m_xyScaleTracks.resize(m_scale_num);

        for( int ixyScale = 0; ixyScale < m_scale_num; ++ixyScale ) {
            std::list<Track>& tracks = m_xyScaleTracks[ixyScale];
            m_fscales[ixyScale] = pow(m_scale_stride, ixyScale);
            
            // find good features at each scale separately
            IplImage *m_grey_temp = 0, *eig_temp = 0;
            std::size_t temp_level = (std::size_t)ixyScale;
            m_grey_temp = cvCloneImage(m_grey_pyramid.getImage(temp_level));
            eig_temp = cvCloneImage(m_eig_pyramid.getImage(temp_level));
            std::vector<CvPoint2D32f> points(0);
            cvDenseSample(m_grey_temp, eig_temp, points, m_quality, m_min_distance);
            
            // save the feature points
            for( i = 0; i < points.size(); i++ ) {
                Track track(m_tracker.trackLength);
                PointDesc point(m_hogInfo, m_hofInfo, m_mbhInfo, points[i]);
                track.addPointDesc(point);
                tracks.push_back(track);
            }
            
            cvReleaseImage( &m_grey_temp );
            cvReleaseImage( &eig_temp );
        }
        
    }

    m_correspondencesPrev2Curr.setTo(cv::Vec2i(-1, -1));
    m_correspondencesCurr2Prev.setTo(cv::Vec2i(-1, -1));
    
    // build the m_image pyramid for the current frame
    cvCopy( frame, m_image, 0 );
    cvCvtColor( m_image, m_grey, CV_BGR2GRAY );
    m_grey_pyramid.rebuild(m_grey);

    if( m_frameNum > 0 ) {
        m_init_counter++;
        
        for( int ixyScale = 0; ixyScale < m_scale_num; ++ixyScale ) {
            double cp1 = omp_get_wtime();
            // track feature points in each scale separately
            std::vector<CvPoint2D32f> points_in(0);
            std::list<Track>& tracks = m_xyScaleTracks[ixyScale];
            for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); ++iTrack) {
                CvPoint2D32f point = iTrack->pointDescs.back().point;
                points_in.push_back(point); // collect all the feature points
            }
            int count = points_in.size();
            IplImage *m_prev_grey_temp = 0, *m_grey_temp = 0;
            std::size_t temp_level = ixyScale;
            m_prev_grey_temp = cvCloneImage(m_prev_grey_pyramid.getImage(temp_level));
            m_grey_temp = cvCloneImage(m_grey_pyramid.getImage(temp_level));
            
            cv::Mat m_prev_grey_mat = cv::cvarrToMat(m_prev_grey_temp);
            cv::Mat m_grey_mat = cv::cvarrToMat(m_grey_temp);
            
            std::vector<CvPoint2D32f> points_out(count);
            m_status.clear();
            m_status.resize(count);
            
            double cp2 = omp_get_wtime();
            
            // compute the optical flow
            IplImage* flow = cvCreateImage(cvGetSize(m_grey_temp), IPL_DEPTH_32F, 2);
            cv::Mat flow_mat = cv::cvarrToMat(flow);
            cv::calcOpticalFlowFarneback( m_prev_grey_mat, m_grey_mat, flow_mat,
                                          sqrt(2)/2.0, 5, 10, 2, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN );
            
            double cp3 = omp_get_wtime();
            
            // track feature points by median filtering
            OpticalFlowTracker(flow, points_in, points_out, m_status);
            
            double cp4 = omp_get_wtime();
            
            int width = m_grey_temp->width;
            int height = m_grey_temp->height;
            // compute the integral histograms
            DescMat* hogMat = InitDescMat(height, width, m_hogInfo.nBins);
            HogComp(m_prev_grey_temp, hogMat, m_hogInfo);
            
            double cp4a = omp_get_wtime();
            
            DescMat* hofMat = InitDescMat(height, width, m_hofInfo.nBins);
            HofComp(flow, hofMat, m_hofInfo);
            
            double cp4b = omp_get_wtime();
            
            DescMat* mbhMatX = InitDescMat(height, width, m_mbhInfo.nBins);
            DescMat* mbhMatY = InitDescMat(height, width, m_mbhInfo.nBins);
            MbhComp(flow, mbhMatX, mbhMatY, m_mbhInfo);
            
            double cp4c = omp_get_wtime();
            
            double cp5 = omp_get_wtime();
            
            i = 0;
            for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); ++i) {
                if( m_status[i] == 1 ) { // if the feature point is successfully tracked
                    PointDesc& pointDesc = iTrack->pointDescs.back();
                    CvPoint2D32f prev_point = points_in[i];
                    // get the descriptors for the feature point
                    CvScalar rect = getRect(prev_point, cvSize(width, height), m_hogInfo);
                    pointDesc.hog = getDesc(hogMat, rect, m_hogInfo);
                    pointDesc.hof = getDesc(hofMat, rect, m_hofInfo);
                    pointDesc.mbhX = getDesc(mbhMatX, rect, m_mbhInfo);
                    pointDesc.mbhY = getDesc(mbhMatY, rect, m_mbhInfo);
                    
                    PointDesc point(m_hogInfo, m_hofInfo, m_mbhInfo, points_out[i]);
                    iTrack->addPointDesc(point);
                    
                    {
                        std::list<PointDesc>::iterator iDesc = iTrack->pointDescs.begin();
                        if (iTrack->pointDescs.size() > 1) {
                            cv::Point2d pointCurr = cv::Point2d(iDesc->point.x, iDesc->point.y) * m_fscales[ixyScale];
                            iDesc++;
                            cv::Point2d pointPrev = cv::Point2d(iDesc->point.x, iDesc->point.y) * m_fscales[ixyScale];
                            
                            m_correspondencesCurr2Prev.at<cv::Vec2i>(pointCurr.y, pointCurr.x) = cv::Vec2i(pointPrev.x, pointPrev.y);
                            m_correspondencesPrev2Curr.at<cv::Vec2i>(pointPrev.y, pointPrev.x) = cv::Vec2i(pointCurr.x, pointCurr.y);
                        }
                    }
                    
                    // draw this track
                    if( m_show_track == 1 ) {
                        std::list<PointDesc>& descs = iTrack->pointDescs;
                        std::list<PointDesc>::iterator iDesc = descs.begin();
                        float length = descs.size();
                        CvPoint2D32f point0 = iDesc->point;
                        point0.x *= m_fscales[ixyScale]; // map the point to first scale
                        point0.y *= m_fscales[ixyScale];
                        
                        float j = 0;
                        for (iDesc++; iDesc != descs.end(); ++iDesc, ++j) {
                            CvPoint2D32f point1 = iDesc->point;
                            point1.x *= m_fscales[ixyScale];
                            point1.y *= m_fscales[ixyScale];
                            
                            cvLine(m_image, cvPointFrom32f(point0), cvPointFrom32f(point1),
                                   CV_RGB(0,cvFloor(255.0*(j+1.0)/length),0), 2, 8,0);
                            point0 = point1;
                        }
                        cvCircle(m_image, cvPointFrom32f(point0), 2, CV_RGB(255,0,0), -1, 8,0);
                    }
                    ++iTrack;
                }
                else // remove the track, if we lose feature point
                    iTrack = tracks.erase(iTrack);
            }

            ReleDescMat(hogMat);
            ReleDescMat(hofMat);
            ReleDescMat(mbhMatX);
            ReleDescMat(mbhMatY);
            cvReleaseImage( &m_prev_grey_temp );
            cvReleaseImage( &m_grey_temp );
            cvReleaseImage( &flow );
            
            double cp6 = omp_get_wtime();
            
//             cout << "1-2 " << cp2 - cp1 << endl;
//             cout << "2-3 " << cp3 - cp2 << endl;
//             cout << "3-4 " << cp4 - cp3 << endl;
//             cout << "4-5 " << cp5 - cp4 << endl;
//             cout << "4-4a " << cp4a - cp4 << endl;
//             cout << "4a-4b " << cp4b - cp4a << endl;
//             cout << "4b-4c " << cp4c - cp4b << endl;
//             cout << "5-6 " << cp6 - cp5 << endl;     
//             cout << "********************************" << endl;
        }
        
        double cp7 = omp_get_wtime();
        
        for( int ixyScale = 0; ixyScale < m_scale_num; ++ixyScale ) {
            std::list<Track>& tracks = m_xyScaleTracks[ixyScale]; // output the features for each scale
            for( std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); ) {
                if( iTrack->pointDescs.size() >= m_tracker.trackLength+1 ) { // if the trajectory achieves the length we want
                    std::vector<CvPoint2D32f> trajectory(m_tracker.trackLength+1);
                    std::list<PointDesc>& descs = iTrack->pointDescs;
                    std::list<PointDesc>::iterator iDesc = descs.begin();
                    
                    for (int count = 0; count <= m_tracker.trackLength; ++iDesc, ++count) {
                        trajectory[count].x = iDesc->point.x*m_fscales[ixyScale];
                        trajectory[count].y = iDesc->point.y*m_fscales[ixyScale];
                    }
                    float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
                    if( isValid(trajectory, mean_x, mean_y, var_x, var_y, length) == 1 ) {
//                         printf("%d\t", m_frameNum);
//                         printf("%f\t%f\t", mean_x, mean_y);
//                         printf("%f\t%f\t", var_x, var_y);
//                         printf("%f\t", length);
//                         printf("%f\t", m_fscales[ixyScale]);
                        
//                         for (int count = 0; count < m_tracker.trackLength; ++count)
//                             printf("%f\t%f\t", trajectory[count].x,trajectory[count].y );
                        
                        iDesc = descs.begin();
                        int t_stride = cvFloor(m_tracker.trackLength/m_hogInfo.ntCells);
                        for( int n = 0; n < m_hogInfo.ntCells; n++ ) {
                            std::vector<float> vec(m_hogInfo.dim);
                            for( int t = 0; t < t_stride; t++, iDesc++ )
                                for( int m = 0; m < m_hogInfo.dim; m++ )
                                    vec[m] += iDesc->hog[m];
//                                 for( int m = 0; m < m_hogInfo.dim; m++ )
//                                     printf("%f\t", vec[m]/float(t_stride));
                        }
                        
                        iDesc = descs.begin();
                        t_stride = cvFloor(m_tracker.trackLength/m_hofInfo.ntCells);
                        for( int n = 0; n < m_hofInfo.ntCells; n++ ) {
                            std::vector<float> vec(m_hofInfo.dim);
                            for( int t = 0; t < t_stride; t++, iDesc++ )
                                for( int m = 0; m < m_hofInfo.dim; m++ )
                                    vec[m] += iDesc->hof[m];
//                                 for( int m = 0; m < m_hofInfo.dim; m++ )
//                                     printf("%f\t", vec[m]/float(t_stride));
                        }
                        
                        iDesc = descs.begin();
                        t_stride = cvFloor(m_tracker.trackLength/m_mbhInfo.ntCells);
                        for( int n = 0; n < m_mbhInfo.ntCells; n++ ) {
                            std::vector<float> vec(m_mbhInfo.dim);
                            for( int t = 0; t < t_stride; t++, iDesc++ )
                                for( int m = 0; m < m_mbhInfo.dim; m++ )
                                    vec[m] += iDesc->mbhX[m];
//                                 for( int m = 0; m < m_mbhInfo.dim; m++ )
//                                     printf("%f\t", vec[m]/float(t_stride));
                        }
                        
                        iDesc = descs.begin();
                        t_stride = cvFloor(m_tracker.trackLength/m_mbhInfo.ntCells);
                        for( int n = 0; n < m_mbhInfo.ntCells; n++ ) {
                            std::vector<float> vec(m_mbhInfo.dim);
                            for( int t = 0; t < t_stride; t++, iDesc++ )
                                for( int m = 0; m < m_mbhInfo.dim; m++ )
                                    vec[m] += iDesc->mbhY[m];
//                                 for( int m = 0; m < m_mbhInfo.dim; m++ )
//                                     printf("%f\t", vec[m]/float(t_stride));
                        }
                        
//                         printf("\n");
                    }
                    iTrack = tracks.erase(iTrack);
                }
                else
                    iTrack++;
            }
        }
        
        double cp8 = omp_get_wtime();
        
        if( m_init_counter == m_tracker.initGap ) { // detect new feature points every initGap frames
            m_init_counter = 0;
            for (int ixyScale = 0; ixyScale < m_scale_num; ++ixyScale) {
                std::list<Track>& tracks = m_xyScaleTracks[ixyScale];
                std::vector<CvPoint2D32f> points_in(0);
                std::vector<CvPoint2D32f> points_out(0);
                for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++, i++) {
                    std::list<PointDesc>& descs = iTrack->pointDescs;
                    CvPoint2D32f point = descs.back().point; // the last point in the track
                    points_in.push_back(point);
                }
                
                IplImage *m_grey_temp = 0, *eig_temp = 0;
                std::size_t temp_level = (std::size_t)ixyScale;
                m_grey_temp = cvCloneImage(m_grey_pyramid.getImage(temp_level));
                eig_temp = cvCloneImage(m_eig_pyramid.getImage(temp_level));
                
                cvDenseSample(m_grey_temp, eig_temp, points_in, points_out, m_quality, m_min_distance);
                // save the new feature points
                for( i = 0; i < points_out.size(); i++) {
                    Track track(m_tracker.trackLength);
                    PointDesc point(m_hogInfo, m_hofInfo, m_mbhInfo, points_out[i]);
                    track.addPointDesc(point);
                    tracks.push_back(track);
                }
                cvReleaseImage( &m_grey_temp );
                cvReleaseImage( &eig_temp );
            }
        }
        
        double cp9 = omp_get_wtime();
        
//         cout << "7-8 " << cp8 - cp7 << endl;
//         cout << "8-9 " << cp9 - cp8 << endl;
        
    }
    cvCopy( frame, m_prev_image, 0 );
    m_frameNum++;
    cvCvtColor( m_prev_image, m_prev_grey, CV_BGR2GRAY );
    m_prev_grey_pyramid.rebuild(m_prev_grey);
}

int DenseTracker::computeLK(const cv::Mat& frame)
{
    if (m_prevImgMat.empty()) {
        m_prevImgMat = cv::Mat(frame.rows, frame.cols, CV_8UC3);
        m_correspondencesPrev2Curr = cv::Mat(frame.rows, frame.cols, CV_32SC2);
        m_correspondencesCurr2Prev = cv::Mat(frame.rows, frame.cols, CV_32SC2);
        m_correspondencesPrev2Curr.setTo(cv::Vec2i(-1, -1));
        m_correspondencesCurr2Prev.setTo(cv::Vec2i(-1, -1));
        m_image = cvCreateImage(cvSize(frame.cols, frame.rows), IPL_DEPTH_8U, 3);
        
        frame.copyTo(m_prevImgMat);
        memcpy((void *)m_image->imageData, (void *)frame.data, frame.cols * frame.rows * frame.channels() * sizeof(uchar));
        
        m_frameNum++;
        
        return 0;
    }
    
    cv::Mat flow(frame.rows, frame.cols, CV_32FC2);
    
    cv::Mat prevGray, currGray;
    cv::cvtColor(frame, currGray, CV_BGR2GRAY);
    cv::cvtColor(m_prevImgMat, prevGray, CV_BGR2GRAY);
    vector<cv::Point2f> prevPoints, currPoints;
    findPairsOFlow(prevGray, currGray, prevPoints, currPoints);
    m_correspondencesCurr2Prev.setTo(cv::Scalar::all(-1));
    m_correspondencesPrev2Curr.setTo(cv::Scalar::all(-1));
    for (uint32_t i = 0; i < prevPoints.size(); i++) {
        const cv::Point2i pointCurr(currPoints[i].x, currPoints[i].y);
        const cv::Point2i pointPrev(prevPoints[i].x, prevPoints[i].y);
        
        m_correspondencesCurr2Prev.at<cv::Vec2i>(pointCurr.y, pointCurr.x) = cv::Vec2i(pointPrev.x, pointPrev.y);
        m_correspondencesPrev2Curr.at<cv::Vec2i>(pointPrev.y, pointPrev.x) = cv::Vec2i(pointCurr.x, pointCurr.y);
    }
    
    //     cvCopy( frame, m_prev_image, 0 );
    frame.copyTo(m_prevImgMat);
    //     frame.copyTo(m_image);
    memcpy((void *)m_image->imageData, (void *)frame.data, frame.cols * frame.rows * frame.channels() * sizeof(uchar));
    //     m_image = m_prev_image;
    
    m_frameNum++;
    
    return 0;
}

inline void DenseTracker::findPairsOFlow(const cv::Mat & img1, const cv::Mat & img2, 
                                            vector<cv::Point2f> & outPoints1, vector<cv::Point2f> & outPoints2) {
                                                
    // We look for correspondences using Optical flow
    // vector of keypoints
    vector<cv::KeyPoint> keypoints1;
    cv::FastFeatureDetector fastDetector(1);
    fastDetector.detect(img1, keypoints1);
    
    cv::Mat imageMatches;
    vector<cv::DMatch> matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints1, matches, imageMatches, cv::Scalar(0,0,255));        
    cv::namedWindow("Matched");
    cv::imshow("Matched", imageMatches);
    
    if (keypoints1.size() == 0)
        return;
    
    vector<cv::Point2f> points1(keypoints1.size()), points2, points1B;
    {
        uint32_t idx = 0;
        for (vector<cv::KeyPoint>::iterator it = keypoints1.begin(); it != keypoints1.end(); it++, idx++) {
            points1[idx] = it->pt;
        }
    }    
    // Optical flow
    vector<uint8_t> status, statusB;
    vector<float_t> error, errorB;
    
    cv::calcOpticalFlowPyrLK(img1, img2, points1, points2, status, error, cv::Size(3, 3), 3);
    cv::calcOpticalFlowPyrLK(img2, img1, points2, points1B, statusB, errorB, cv::Size(3, 3), 3);
    
    vector<cv::Point2f> pointsA(points1.size()), pointsB(points2.size());
    {
        uint32_t idx = 0;
        for (uint32_t i = 0; i < points1.size(); i++) {
            if ((status[i] == 1) && (statusB[i] == 1)) {
                if (cv::norm(points1[i] - points1B[i]) < 1.0) {
                    pointsA[idx] = points1[i];
                    pointsB[idx] = points2[i];
                }
            }
            idx++;
        }
        pointsA.resize(idx);
        pointsB.resize(idx);
    }
    
    outPoints1 = pointsA;
    outPoints2 = pointsB;
}

inline void DenseTracker::findPairsSURF(const cv::Mat & img1, const cv::Mat & img2,
                                        vector<cv::Point2f> & outPoints1, vector<cv::Point2f> & outPoints2) {
    
    // We look for correspondences using SURF
    
    // vector of keypoints
    vector<cv::KeyPoint> keypoints1, keypoints2;
    
    cv::SurfFeatureDetector surf(50);
    surf.detect(img1, keypoints1);
    surf.detect(img2, keypoints2);
    
    // Descriptors are extracted
    cv::SurfDescriptorExtractor surfDesc;
    cv::Mat descriptors1, descriptors2;
    surfDesc.compute(img1, keypoints1, descriptors1);
    surfDesc.compute(img2, keypoints2, descriptors2);
    
    // Descriptors are matched
    cv::FlannBasedMatcher matcher;
    vector<cv::DMatch> matches;
    
    matcher.match(descriptors1, descriptors2, matches);
    
    nth_element(matches.begin(), matches.begin()+24, matches.end());
    matches.erase(matches.begin()+25, matches.end());
    
    cv::Mat imageMatches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, imageMatches, cv::Scalar(0,0,255));        
    cv::namedWindow("Matched");
    cv::imshow("Matched", imageMatches);
    
    // Fundamental matrix is found
    vector<cv::Point2f> points1, points2;
    
    points1.resize(matches.size());
    points2.resize(matches.size());
    
    {
        uint32_t idx2 = 0;
        for (int idx = 0; idx < matches.size(); idx++) {
            const cv::Point2f & p1 = keypoints1[matches[idx].queryIdx].pt;
            const cv::Point2f & p2 = keypoints2[matches[idx].trainIdx].pt;
            
            if (fabs(p1.x - p2.x < 10.0) && fabs(p1.y - p2.y < 10.0)) {
                points1[idx2] = p1;
                points2[idx2] = p2;
                idx2++;
            }
        }
        points1.resize(idx2);
        points2.resize(idx2);
    }
    
    outPoints1 = points1;
    outPoints2 = points2;
}


void DenseTracker::InitTrackerInfo(TrackerInfo* tracker, int m_track_length, int m_init_gap)
{
    tracker->trackLength = m_track_length;
    tracker->initGap = m_init_gap;
}

DescMat* DenseTracker::InitDescMat(int height, int width, int nBins)
{
    DescMat* descMat = (DescMat*)malloc(sizeof(DescMat));
    descMat->height = height;
    descMat->width = width;
    descMat->nBins = nBins;
    descMat->desc = (float*)malloc(height*width*nBins*sizeof(float));
    memset( descMat->desc, 0, height*width*nBins*sizeof(float));
    return descMat;
}

void DenseTracker::ReleDescMat( DescMat* descMat)
{
    free(descMat->desc);
    free(descMat);
}

void DenseTracker::InitDescInfo(DescInfo* descInfo, int nBins, int flag, int orientation, int size, int m_nxy_cell, int m_nt_cell)
{
    descInfo->nBins = nBins;
    descInfo->fullOrientation = orientation;
    descInfo->norm = 2;
    descInfo->threshold = m_min_flow;
    descInfo->flagThre = flag;
    descInfo->nxCells = m_nxy_cell;
    descInfo->nyCells = m_nxy_cell;
    descInfo->ntCells = m_nt_cell;
    descInfo->dim = descInfo->nBins*descInfo->nxCells*descInfo->nyCells;
    descInfo->blockHeight = size;
    descInfo->blockWidth = size;
}

/* get the rectangle for computing the descriptor */
CvScalar DenseTracker::getRect(const CvPoint2D32f point, // the interest point position
                 const CvSize size, // the size of the m_image
                 const DescInfo descInfo) // parameters about the descriptor
{
    int x_min = descInfo.blockWidth/2;
    int y_min = descInfo.blockHeight/2;
    int x_max = size.width - descInfo.blockWidth;
    int y_max = size.height - descInfo.blockHeight;
    
    CvPoint2D32f point_temp;
    
    float temp = point.x - x_min;
    point_temp.x = std::min<float>(std::max<float>(temp, 0.), x_max);
    
    temp = point.y - y_min;
    point_temp.y = std::min<float>(std::max<float>(temp, 0.), y_max);
    
    // return the rectangle
    CvScalar rect;
    rect.val[0] = point_temp.x;
    rect.val[1] = point_temp.y;
    rect.val[2] = descInfo.blockWidth;
    rect.val[3] = descInfo.blockHeight;
    
    return rect;
}

/* compute integral histograms for the whole m_image */
void DenseTracker::BuildDescMat(const IplImage* xComp, // x gradient component
                  const IplImage* yComp, // y gradient component
                  DescMat* descMat, // output integral histograms
                  const DescInfo descInfo) // parameters about the descriptor
{
    // whether use full orientation or not
    float fullAngle = descInfo.fullOrientation ? 360 : 180;
    // one additional bin for hof
    int nBins = descInfo.flagThre ? descInfo.nBins-1 : descInfo.nBins;
    // angle stride for quantization
    float angleBase = fullAngle/float(nBins);
    int width = descMat->width;
    int height = descMat->height;
    int histDim = descMat->nBins;
    int index = 0;
    for(int i = 0; i < height; i++) {
        const float* xcomp = (const float*)(xComp->imageData + xComp->widthStep*i);
        const float* ycomp = (const float*)(yComp->imageData + yComp->widthStep*i);
        
        // the histogram accumulated in the current line
        std::vector<float> sum(histDim);
        for(int j = 0; j < width; j++, index++) {
            float shiftX = xcomp[j];
            float shiftY = ycomp[j];
            float magnitude0 = sqrt(shiftX*shiftX+shiftY*shiftY);
            float magnitude1 = magnitude0;
            int bin0, bin1;
            
            // for the zero bin of hof
            if(descInfo.flagThre == 1 && magnitude0 <= descInfo.threshold) {
                bin0 = nBins; // the zero bin is the last one
                magnitude0 = 1.0;
                bin1 = 0;
                magnitude1 = 0;
            }
            else {
                float orientation = cvFastArctan(shiftY, shiftX);
                if(orientation > fullAngle)
                    orientation -= fullAngle;
                
                // split the magnitude to two adjacent bins
                float fbin = orientation/angleBase;
                bin0 = cvFloor(fbin);
                float weight0 = 1 - (fbin - bin0);
                float weight1 = 1 - weight0;
                bin0 %= nBins;
                bin1 = (bin0+1)%nBins;
                
                magnitude0 *= weight0;
                magnitude1 *= weight1;
            }
            
            sum[bin0] += magnitude0;
            sum[bin1] += magnitude1;
            
            int temp0 = index*descMat->nBins;
            if(i == 0) { // for the first line
                for(int m = 0; m < descMat->nBins; m++)
                    descMat->desc[temp0++] = sum[m];
            }
            else {
                int temp1 = (index - width)*descMat->nBins;
                for(int m = 0; m < descMat->nBins; m++)
                    descMat->desc[temp0++] = descMat->desc[temp1++]+sum[m];
            }
        }
    }
}

/* get a descriptor from the integral histogram */
std::vector<float> DenseTracker::getDesc(const DescMat* descMat, // input integral histogram
                           CvScalar rect, // rectangle area for the descriptor
                           DescInfo descInfo) // parameters about the descriptor
{
    int descDim = descInfo.dim;
    int height = descMat->height;
    int width = descMat->width;
    
    boost::numeric::ublas::vector<double> vec(descDim);
    int xOffset = rect.val[0];
    int yOffset = rect.val[1];
    int xStride = rect.val[2]/descInfo.nxCells;
    int yStride = rect.val[3]/descInfo.nyCells;
    
    // iterate over different cells
    int iDesc = 0;
    for (int iX = 0; iX < descInfo.nxCells; ++iX)
        for (int iY = 0; iY < descInfo.nyCells; ++iY) {
            // get the positions of the rectangle
            int left = xOffset + iX*xStride - 1;
            int right = std::min<int>(left + xStride, width-1);
            int top = yOffset + iY*yStride - 1;
            int bottom = std::min<int>(top + yStride, height-1);
            
            // get the index in the integral histogram
            int TopLeft = (top*width+left)*descInfo.nBins;
            int TopRight = (top*width+right)*descInfo.nBins;
            int BottomLeft = (bottom*width+left)*descInfo.nBins;
            int BottomRight = (bottom*width+right)*descInfo.nBins;
            
            for (int i = 0; i < descInfo.nBins; ++i, ++iDesc) {
                double sumTopLeft(0), sumTopRight(0), sumBottomLeft(0), sumBottomRight(0);
                if (top >= 0) {
                    if (left >= 0)
                        sumTopLeft = descMat->desc[TopLeft+i];
                    if (right >= 0)
                        sumTopRight = descMat->desc[TopRight+i];
                }
                if (bottom >= 0) {
                    if (left >= 0)
                        sumBottomLeft = descMat->desc[BottomLeft+i];
                    if (right >= 0)
                        sumBottomRight = descMat->desc[BottomRight+i];
                }
                float temp = sumBottomRight + sumTopLeft
                - sumBottomLeft - sumTopRight;
                vec[iDesc] = std::max<float>(temp, 0) + m_epsilon;
            }
        }
        if (descInfo.norm == 1) // L1 normalization
            vec *= 1 / boost::numeric::ublas::norm_1(vec);
        else // L2 normalization
            vec *= 1 / boost::numeric::ublas::norm_2(vec);
        
        std::vector<float> desc(descDim);
        for (int i = 0; i < descDim; i++)
            desc[i] = vec[i];
        return desc;
}

void DenseTracker::HogComp(IplImage* img, DescMat* descMat, DescInfo descInfo)
{
    int width = descMat->width;
    int height = descMat->height;
    IplImage* imgX = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
    IplImage* imgY = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
    cvSobel(img, imgX, 1, 0, 1);
    cvSobel(img, imgY, 0, 1, 1);
    BuildDescMat(imgX, imgY, descMat, descInfo);
    cvReleaseImage(&imgX);
    cvReleaseImage(&imgY);
}

void DenseTracker::HofComp(IplImage* flow, DescMat* descMat, DescInfo descInfo)
{
    int width = descMat->width;
    int height = descMat->height;
    IplImage* xComp = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
    IplImage* yComp = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
    for(int i = 0; i < height; i++) {
        const float* f = (const float*)(flow->imageData + flow->widthStep*i);
        float* xf = (float*)(xComp->imageData + xComp->widthStep*i);
        float* yf = (float*)(yComp->imageData + yComp->widthStep*i);
        for(int j = 0; j < width; j++) {
            xf[j] = f[2*j];
            yf[j] = f[2*j+1];
        }
    }
    BuildDescMat(xComp, yComp, descMat, descInfo);
    cvReleaseImage(&xComp);
    cvReleaseImage(&yComp);
}

void DenseTracker::MbhComp(IplImage* flow, DescMat* descMatX, DescMat* descMatY, DescInfo descInfo)
{
    int width = descMatX->width;
    int height = descMatX->height;
    IplImage* flowX = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
    IplImage* flowY = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
    IplImage* flowXdX = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
    IplImage* flowXdY = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
    IplImage* flowYdX = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
    IplImage* flowYdY = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 1);
    
    // extract the x and y components of the flow
    for(int i = 0; i < height; i++) {
        const float* f = (const float*)(flow->imageData + flow->widthStep*i);
        float* fX = (float*)(flowX->imageData + flowX->widthStep*i);
        float* fY = (float*)(flowY->imageData + flowY->widthStep*i);
        for(int j = 0; j < width; j++) {
            fX[j] = 100*f[2*j];
            fY[j] = 100*f[2*j+1];
        }
    }
    
    cvSobel(flowX, flowXdX, 1, 0, 1);
    cvSobel(flowX, flowXdY, 0, 1, 1);
    cvSobel(flowY, flowYdX, 1, 0, 1);
    cvSobel(flowY, flowYdY, 0, 1, 1);
    
    BuildDescMat(flowXdX, flowXdY, descMatX, descInfo);
    BuildDescMat(flowYdX, flowYdY, descMatY, descInfo);
    
    cvReleaseImage(&flowX);
    cvReleaseImage(&flowY);
    cvReleaseImage(&flowXdX);
    cvReleaseImage(&flowXdY);
    cvReleaseImage(&flowYdX);
    cvReleaseImage(&flowYdY);
}

/* tracking interest points by median filtering in the optical field */
void DenseTracker::OpticalFlowTracker(IplImage* flow, // the optical field
                        std::vector<CvPoint2D32f>& points_in, // input interest point positions
                        std::vector<CvPoint2D32f>& points_out, // output interest point positions
                        std::vector<int>& status) // status for successfully tracked or not
{
    if(points_in.size() != points_out.size())
        fprintf(stderr, "the numbers of points don't match!");
    if(points_in.size() != status.size())
        fprintf(stderr, "the number of status doesn't match!");
    int width = flow->width;
    int height = flow->height;
    
    for(int i = 0; i < points_in.size(); i++) {
        CvPoint2D32f point_in = points_in[i];
        std::list<float> xs;
        std::list<float> ys;
        int x = cvFloor(point_in.x);
        int y = cvFloor(point_in.y);
        for(int m = x-1; m <= x+1; m++)
            for(int n = y-1; n <= y+1; n++) {
                int p = std::min<int>(std::max<int>(m, 0), width-1);
                int q = std::min<int>(std::max<int>(n, 0), height-1);
                const float* f = (const float*)(flow->imageData + flow->widthStep*q);
                xs.push_back(f[2*p]);
                ys.push_back(f[2*p+1]);
            }
            
            xs.sort();
        ys.sort();
        int size = xs.size()/2;
        for(int m = 0; m < size; m++) {
            xs.pop_back();
            ys.pop_back();
        }
        
        CvPoint2D32f offset;
        offset.x = xs.back();
        offset.y = ys.back();
        CvPoint2D32f point_out;
        point_out.x = point_in.x + offset.x;
        point_out.y = point_in.y + offset.y;
        points_out[i] = point_out;
        if( point_out.x > 0 && point_out.x < width && point_out.y > 0 && point_out.y < height )
            status[i] = 1;
        else
            status[i] = -1;
    }
}

/* check whether a trajectory is valid or not */
int DenseTracker::isValid(std::vector<CvPoint2D32f>& track, float& mean_x, float& mean_y, float& var_x, float& var_y, float& length)
{
    int size = track.size();
    for(int i = 0; i < size; i++) {
        mean_x += track[i].x;
        mean_y += track[i].y;
    }
    mean_x /= size;
    mean_y /= size;
    
    for(int i = 0; i < size; i++) {
        track[i].x -= mean_x;
        var_x += track[i].x*track[i].x;
        track[i].y -= mean_y;
        var_y += track[i].y*track[i].y;
    }
    var_x /= size;
    var_y /= size;
    var_x = sqrt(var_x);
    var_y = sqrt(var_y);
    // remove static trajectory
    if(var_x < m_min_var && var_y < m_min_var)
        return 0;
    // remove random trajectory
    if( var_x > m_max_var || var_y > m_max_var )
        return 0;
    
    for(int i = 1; i < size; i++) {
        float temp_x = track[i].x - track[i-1].x;
        float temp_y = track[i].y - track[i-1].y;
        length += sqrt(temp_x*temp_x+temp_y*temp_y);
        track[i-1].x = temp_x;
        track[i-1].y = temp_y;
    }
    
    float len_thre = length*0.7;
    for( int i = 0; i < size-1; i++ ) {
        float temp_x = track[i].x;
        float temp_y = track[i].y;
        float temp_dis = sqrt(temp_x*temp_x + temp_y*temp_y);
        if( temp_dis > m_max_dis && temp_dis > len_thre )
            return 0;
    }
    
    track.pop_back();
    // normalize the trajectory
    for(int i = 0; i < size-1; i++) {
        track[i].x /= length;
        track[i].y /= length;
    }
    return 1;
}

/* detect new feature points in the whole m_image */
void DenseTracker::cvDenseSample(IplImage* m_grey, IplImage* eig, std::vector<CvPoint2D32f>& points,
                   const double m_quality, const double m_min_distance)
{
    int width = cvFloor(m_grey->width/m_min_distance);
    int height = cvFloor(m_grey->height/m_min_distance);
    double maxVal = 0;
    cvCornerMinEigenVal(m_grey, eig, 3, 3);
    cvMinMaxLoc(eig, 0, &maxVal, 0, 0, 0);
    const double threshold = maxVal*m_quality;
    
    int offset = cvFloor(m_min_distance/2);
    for(int i = 0; i < height; i++) 
        for(int j = 0; j < width; j++) {
            int x = cvFloor(j*m_min_distance+offset);
            int y = cvFloor(i*m_min_distance+offset);
            if(CV_IMAGE_ELEM(eig, float, y, x) > threshold) 
                points.push_back(cvPoint2D32f(x,y));
        }
}
/* detect new feature points in a m_image without overlapping to previous points */
void DenseTracker::cvDenseSample(IplImage* m_grey, IplImage* eig, std::vector<CvPoint2D32f>& points_in,
                   std::vector<CvPoint2D32f>& points_out, const double m_quality, const double m_min_distance)
{
    int width = cvFloor(m_grey->width/m_min_distance);
    int height = cvFloor(m_grey->height/m_min_distance);
    double maxVal = 0;
    cvCornerMinEigenVal(m_grey, eig, 3, 3);
    cvMinMaxLoc(eig, 0, &maxVal, 0, 0, 0);
    const double threshold = maxVal*m_quality;
    
    std::vector<int> counters(width*height);
    for(int i = 0; i < points_in.size(); i++) {
        CvPoint2D32f point = points_in[i];
        if(point.x >= m_min_distance*width || point.y >= m_min_distance*height)
            continue;
        int x = cvFloor(point.x/m_min_distance);
        int y = cvFloor(point.y/m_min_distance);
        counters[y*width+x]++;
    }
    
    int index = 0;
    int offset = cvFloor(m_min_distance/2);
    for(int i = 0; i < height; i++) 
        for(int j = 0; j < width; j++, index++) {
            if(counters[index] == 0) {
                int x = cvFloor(j*m_min_distance+offset);
                int y = cvFloor(i*m_min_distance+offset);
                if(CV_IMAGE_ELEM(eig, float, y, x) > threshold) 
                    points_out.push_back(cvPoint2D32f(x,y));
            }
        }
}

void DenseTracker::drawTracks(cv::Mat& output)
{
    if (m_image == NULL) {
        output = cv::Mat::zeros(200, 200, CV_8UC3);
        return;
    }
    
    output = cv::Mat(m_image->height, m_image->width, CV_8UC3);
    cv::Mat(m_image).copyTo(output);
    
    vector<cv::Scalar> color(output.cols);
    for (uint32_t i = 0; i < output.cols; i++) {
        color[i] = cv::Scalar(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
    }
    
    for (uint32_t i = 0; i < output.rows; i += 10) {
        for (uint32_t j = 0; j < output.cols; j ++) {
            const cv::Point2i pointCurr(j, i);
            const cv::Vec2i & tmpVect = m_correspondencesCurr2Prev.at<cv::Vec2i>(i, j);
            const cv::Point2i pointPrev(tmpVect);
            
            if (pointPrev != cv::Point2i(-1, -1)) {
                cv::line(output, pointCurr, pointPrev, color[j]);
            }
        }
    }
}

cv::Point2i DenseTracker::getPrevPoint(const cv::Point2i& currPoint)
{
    return cv::Point2i(m_correspondencesCurr2Prev.at<cv::Vec2i>(currPoint.y, currPoint.x));
}

cv::Point2i DenseTracker::getCurrPoint(const cv::Point2i& prevPoint)
{
    return cv::Point2i(m_correspondencesPrev2Curr.at<cv::Vec2i>(prevPoint.y, prevPoint.x));
}

}
