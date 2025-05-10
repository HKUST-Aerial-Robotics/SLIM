#include "frame.h"

namespace SLIM {

uint32_t Frame::factory_id_ = 0;
  
Frame::Frame(const uint64_t timestamp, const Transform& Twb)
: id_(factory_id_++), timestamp_(timestamp), Twb_(Twb), Tbl_(Transform()) {}

Frame::Frame(const uint32_t id, const uint64_t timestamp, const Transform& Twb)
: id_(id), timestamp_(timestamp), Twb_(Twb), Tbl_(Transform()) {}

Frame::Frame(const uint32_t id, const uint64_t timestamp, const Transform& Twb, 
            const std::vector<LineOB::Ptr>& line_obs,
            const std::vector<SurfaceOB::Ptr>& surf_obs) 
: id_(id), timestamp_(timestamp), Twb_(Twb), Tbl_(Transform()), line_obs_(line_obs), surface_obs_(surf_obs) {}


void Frame::DownSample(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, const float resolution) {
  pcl::VoxelGrid<pcl::PointXYZI> vf;
  vf.setLeafSize(resolution, resolution, resolution);
  vf.setInputCloud(cloud);
  vf.filter(*cloud);
}

} // namespace SLIM
