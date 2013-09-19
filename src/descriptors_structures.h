#ifndef DESCRIPTORS_STRUCTURES_H_
#define DESCRIPTORS_STRUCTURES_H_

#include <list>

struct point;
namespace dense_tracker {
    typedef struct DescInfo
    {
        int nBins; // number of bins for vector quantization
        int fullOrientation; // 0: 180 degree; 1: 360 degree
        int norm; // 1: L1 normalization; 2: L2 normalization
        float threshold; //threshold for normalization
        int flagThre; // whether thresholding or not
        int nxCells; // number of cells in x direction
        int nyCells; 
        int ntCells;
        int dim; // dimension of the descriptor
        int blockHeight; // size of the block for computing the descriptor
        int blockWidth;
    }DescInfo; 
    
    typedef struct DescMat
    {
        int height;
        int width;
        int nBins;
        float* desc;
    }DescMat;
    
    class PointDesc
    {
    public:
        std::vector<float> hog;
        std::vector<float> hof;
        std::vector<float> mbhX;
        std::vector<float> mbhY;
        CvPoint2D32f point;
        
        PointDesc(const DescInfo& hogInfo, const DescInfo& hofInfo, const DescInfo& mbhInfo, const CvPoint2D32f& point_)
        : hog(hogInfo.nxCells * hogInfo.nyCells * hogInfo.nBins),
        hof(hofInfo.nxCells * hofInfo.nyCells * hofInfo.nBins),
        mbhX(mbhInfo.nxCells * mbhInfo.nyCells * mbhInfo.nBins),
        mbhY(mbhInfo.nxCells * mbhInfo.nyCells * mbhInfo.nBins),
        point(point_)
        {}
    };
    
    class Track
    {
    public:
        std::list<PointDesc> pointDescs;
        int maxNPoints;
        
        Track(int maxNPoints_)
        : maxNPoints(maxNPoints_)
        {}
        
        void addPointDesc(const PointDesc& point)
        {
            pointDescs.push_back(point);
            if (pointDescs.size() > maxNPoints + 2) {
                pointDescs.pop_front();
            }
        }
    };
}

#endif