/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#ifndef PCM_SOLVER_H
#define PCM_SOLVER_H

#include <unordered_set>
#include <map>
#include <vector>

#include <pmc.h>
#include "clipper.h"

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "vector_map.h"
#include "scene_graph.h"
#include "semantic_definition.h"

namespace SLIM {

using Association = Eigen::Matrix<int, Eigen::Dynamic, 2>;

inline std::tuple<size_t,size_t> k2ij(size_t k, size_t n) {
  k += 1;

  const size_t l = n*(n-1)/2 - k;
  const size_t o = std::floor( (std::sqrt(1 + 8*l) - 1) / 2. );
  const size_t p = l - o*(o+1)/2;
  const size_t i = n - (o + 1);
  const size_t j = n - p;
  return {i-1, j-1};
}

inline Association createMatch(const std::vector<SceneGraphNode::Ptr>& L1, const std::vector<SceneGraphNode::Ptr>& L2) {
  std::vector<std::pair<int, int>> Avec;
  for (size_t i=0; i<L1.size(); ++i) {
    for (size_t j=0; j<L2.size(); ++j) {
      if(L1[i]->semantic_type() == L2[j]->semantic_type()) {
        Avec.push_back(std::make_pair(i, j));
      }
    }
  }
  Association A = Association(Avec.size(), 2);
  for (size_t i=0; i<Avec.size(); ++i) {
    A(i, 0) = Avec[i].first;
    A(i, 1) = Avec[i].second;
  }
  return A;
}


struct OverlapInfo {
  Block::Ptr bi, bj;
  VectorMap::Ptr mi, mj;
  Transform Tij; // frame relative pose
  Transform Trc; // block relative pose
  Eigen::MatrixXd sqrt_info;
  OverlapInfo(): bi(nullptr), bj(nullptr), Tij(Transform()), Trc(Transform()) {}
  OverlapInfo(const Block::Ptr& _bi, const Block::Ptr& _bj, const Transform& _Tij, const Transform& _Trc, const Eigen::MatrixXd& _sqrt_info)
  : bi(_bi), bj(_bj), Tij(_Tij), Trc(_Trc), sqrt_info(_sqrt_info) {}

  OverlapInfo(const Block::Ptr& _bi, const Block::Ptr& _bj, const VectorMap::Ptr& _mi, const VectorMap::Ptr& _mj, const Transform& _Tij, const Transform& _Trc, const Eigen::MatrixXd& _sqrt_info)
  : bi(_bi), bj(_bj), mi(_mi), mj(_mj), Tij(_Tij), Trc(_Trc), sqrt_info(_sqrt_info) {}

};

struct BilateralEdge {
  Frame::Ptr fi, fj;
  Transform Tij;
  BilateralEdge(): fi(nullptr), fj(nullptr), Tij(Transform()) {}
  BilateralEdge(const Frame::Ptr& _fi, const Frame::Ptr& _fj, const Transform& _Tij) : fi(_fi), fj(_fj), Tij(_Tij) {}
};

/**
 * A simple undirected graph class
 *
 * This graph assumes that vertices are numbered. In addition, the vertices numbers have to be
 * consecutive starting from 0.
 *
 * For example, if the graph have 3 vertices, they have to be named 0, 1, and 2.
 */
class AssociationGraph {
public:
  AssociationGraph() : num_edges_(0){};

  /**
   * Constructor that takes in an adjacency list. Notice that for an edge connecting two arbitrary
   * vertices v1 & v2, we assume that v2 exists in v1's list, and v1 also exists in v2's list. This
   * condition is not enforced. If violated, removeEdge() function might exhibit undefined
   * behaviors.
   * @param [in] adj_list an map representing an adjacency list
   */
  explicit AssociationGraph(const std::map<int, std::vector<int>>& adj_list) {
    adj_list_.resize(adj_list.size());
    num_edges_ = 0;
    for (const auto& e_list : adj_list) {
      const auto& v = e_list.first;
      adj_list_[e_list.first] = e_list.second;
      num_edges_ += e_list.second.size();
    }
    num_edges_ /= 2;
  };

  void build(const std::vector<OverlapInfo>& info);

  void build(const std::vector<BilateralEdge>& info);

  void buildMutualAffinity(const SceneGraph::Ptr& g1, const SceneGraph::Ptr& g2);

  /**
   * Add a vertex with no edges.
   * @param [in] id the id of vertex to be added
   */
  void addVertex(const int& id) {
    if (id < adj_list_.size()) {
      // TEASER_DEBUG_ERROR_MSG("Vertex already exists.");
    } else {
      adj_list_.resize(id + 1);
    }
  }

  /**
   * Populate the graph with the provided number of vertices without any edges.
   * @param num_vertices
   */
  void populateVertices(const int& num_vertices) { 
    adj_list_.resize(num_vertices);
  }

  /**
   * Return true if said edge exists
   * @param [in] vertex_1
   * @param [in] vertex_2
   */
  bool hasEdge(const int& vertex_1, const int& vertex_2) {
    if (vertex_1 >= adj_list_.size() || vertex_2 >= adj_list_.size()) {
      return false;
    }
    auto& connected_vs = adj_list_[vertex_1];
    bool exists =
        std::find(connected_vs.begin(), connected_vs.end(), vertex_2) != connected_vs.end();
    return exists;
  }

  /**
   * Return true if the vertex exists.
   * @param vertex
   * @return
   */
  bool hasVertex(const int& vertex) { return vertex < adj_list_.size(); }

  /**
   * Add an edge between two vertices
   * @param [in] vertex_1 one vertex of the edge
   * @param [in] vertex_2 another vertex of the edge
   */
  void addEdge(const int& vertex_1, const int& vertex_2) {
    if (hasEdge(vertex_1, vertex_2)) {
      // TEASER_DEBUG_ERROR_MSG("Edge exists.");
      return;
    }
    adj_list_[vertex_1].push_back(vertex_2);
    adj_list_[vertex_2].push_back(vertex_1);
    num_edges_++;
  }

  /**
   * Remove the edge between two vertices.
   * @param [in] vertex_1 one vertex of the edge
   * @param [in] vertex_2 another vertex of the edge
   */
  void removeEdge(const int& vertex_1, const int& vertex_2) {
    if (vertex_1 >= adj_list_.size() || vertex_2 >= adj_list_.size()) {
      // TEASER_DEBUG_ERROR_MSG("Trying to remove non-existent edge.");
      return;
    }

    adj_list_[vertex_1].erase(
        std::remove(adj_list_[vertex_1].begin(), adj_list_[vertex_1].end(), vertex_2),
        adj_list_[vertex_1].end());
    adj_list_[vertex_2].erase(
        std::remove(adj_list_[vertex_2].begin(), adj_list_[vertex_2].end(), vertex_1),
        adj_list_[vertex_2].end());

    num_edges_--;
  }

  /**
   * Get the number of vertices
   * @return total number of vertices
   */
  [[nodiscard]] int numVertices() const { return adj_list_.size(); }

  /**
   * Get the number of edges
   * @return total number of edges
   */
  [[nodiscard]] int numEdges() const { return num_edges_; }

  /**
   * Get edges originated from a specific vertex
   * @param [in] id
   * @return an unordered set of edges
   */
  [[nodiscard]] const std::vector<int>& getEdges(int id) const { return adj_list_[id]; }

  /**
   * Get all vertices
   * @return a vector of all vertices
   */
  [[nodiscard]] std::vector<int> getVertices() const {
    std::vector<int> v;
    for (int i = 0; i < adj_list_.size(); ++i) {
      v.push_back(i);
    }
    return v;
  }

  [[nodiscard]] Eigen::MatrixXi getAdjMatrix() const {
    const int num_v = numVertices();
    Eigen::MatrixXi adj_matrix(num_v, num_v);
    for (size_t i = 0; i < num_v; ++i) {
      const auto& c_edges = getEdges(i);
      for (size_t j = 0; j < num_v; ++j) {
        if (std::find(c_edges.begin(), c_edges.end(), j) != c_edges.end()) {
          adj_matrix(i, j) = 1;
        } else {
          adj_matrix(i, j) = 0;
        }
      }
    }
    return adj_matrix;
  }

  [[nodiscard]] std::vector<std::vector<int>> getAdjList() const { return adj_list_; }

  [[nodiscard]] std::vector<std::pair<uint32_t, uint32_t>> getFinalMatches(const std::vector<int>& clique) const {
    std::vector<std::pair<uint32_t, uint32_t>> matches;
    matches.reserve(clique.size());
    for(int i = 0; i < clique.size(); ++i) {
      matches.push_back(potential_matches_[clique[i]]);
    }
    return matches;
  }

  /**
   * Preallocate spaces for vertices
   * @param num_vertices
   */
  void reserve(const int& num_vertices) { adj_list_.reserve(num_vertices); }

  /**
   * Clear the contents of the graph
   */
  void clear() {
    adj_list_.clear();
    num_edges_ = 0;
  }

  /**
   * Reserve space for complete graph. A complete undirected graph should have N*(N-1)/2 edges
   * @param num_vertices
   */
  void reserveForCompleteGraph(const int& num_vertices) {
    adj_list_.reserve(num_vertices);
    for (int i = 0; i < num_vertices - 1; ++i) {
      std::vector<int> c_edges;
      c_edges.reserve(num_vertices - 1);
      adj_list_.push_back(c_edges);
    }
    adj_list_.emplace_back(std::initializer_list<int>{});
  }

  

public:
  std::vector<std::vector<int>> adj_list_;
  std::vector<std::pair<uint32_t, uint32_t>> potential_matches_;
  size_t num_edges_;
  Association A_;
};

class GlobalRegister {
 public:
  GlobalRegister() {
    inlier_graphs_.resize(bound_vec_.size());
    max_cliques_.resize(bound_vec_.size());
  }

  Association solveMatch(const SceneGraph::Ptr& g1, const SceneGraph::Ptr& g2);

  std::vector<int> checkSolution(const SceneGraph::Ptr& gref, const SceneGraph::Ptr& gcur, 
                                               const Association& association, const Transform& Trc);

 private:
  Association A_;
  std::vector<AssociationGraph> inlier_graphs_;
  std::vector<std::vector<int>> max_cliques_;
  std::vector<double> bound_vec_ = {0.2};

};
/**
 * A facade to the Parallel Maximum Clique (PMC) library.
 *
 * For details about PMC, please refer to:
 * https://github.com/ryanrossi/pmc
 * and
 * Ryan A. Rossi, David F. Gleich, Assefaw H. Gebremedhin, Md. Mostofa Patwary, A Fast Parallel
 * Maximum Clique Algorithm for Large Sparse Graphs and Temporal Strong Components, arXiv preprint
 * 1302.6256, 2013.
 */
class MaxCliqueSolver {
 public:
  /**
   * Enum representing the solver algorithm to use
   */
  enum class CLIQUE_SOLVER_MODE {
    PMC_EXACT = 0,
    PMC_HEU = 1,
    KCORE_HEU = 2,
  };

  /**
   * Parameter struct for MaxCliqueSolver
   */
  struct Params {

    /**
     * Algorithm used for finding max clique.
     */
    CLIQUE_SOLVER_MODE solver_mode = CLIQUE_SOLVER_MODE::PMC_EXACT;

    /**
     * \deprecated Use solver_mode instead
     * Set this to false to enable heuristic-only max clique finding.
     */
    bool solve_exactly = true;

    /**
     * The threshold ratio for determining whether to skip max clique and go straightly to
     * GNC rotation estimation. Set this to 1 to always use exact max clique selection, 0 to always
     * skip exact max clique selection.
     */
    double kcore_heuristic_threshold = 1;

    /**
     * Time limit on running the solver.
     */
    double time_limit = 3600;
  };

  MaxCliqueSolver() = default;

  MaxCliqueSolver(Params params) : params_(params){};

  /**
   * Find the maximum clique within the graph provided. By maximum clique, it means the clique of
   * the largest size in an undirected graph.
   * @param graph
   * @return a vector of indices of cliques
   */
  std::vector<int> findMaxClique(const AssociationGraph& graph, const int lower_bound = 0);

 private:
  Params params_;
};


} // namespace SLIM

#endif