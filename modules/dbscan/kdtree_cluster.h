#ifndef KDTREE_CLUSTER_H
#define KDTREE_CLUSTER_H

#include <pcl/point_types.h>
#include "simple_cluster.h"

namespace SLIM {

template <typename PointT>
class DBSCANKdtreeCluster: public DBSCANSimpleCluster<PointT> {
 protected:
  virtual int radiusSearch (int index, double radius, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances) const {
    return this->search_method_->radiusSearch(index, radius, k_indices, k_sqr_distances);
  }
}; // class DBSCANKdtreeCluster

#endif // KDTREE_CLUSTER_H

}