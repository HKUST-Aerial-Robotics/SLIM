/**
 * @file clipper.h
 * @brief Clipper data association framework
 * @author Parker Lusk <plusk@mit.edu>
 * @date 3 October 2020
 */

#pragma once

#include <tuple>
#include <memory>
#include <random>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <pmc.h>
#include "scene_graph.h"
#include "scene_graph_node.h"
#include "landmark.h"
#include "pcm_solver.h"

namespace SLIM {
namespace clipper {
  
  using SpMat = Eigen::SparseMatrix<double>;

  using Association = Eigen::Matrix<int, Eigen::Dynamic, 2>;
  using Affinity = Eigen::MatrixXd;
  using Constraint = Eigen::MatrixXd;

  using SpAffinity = SpMat;
  using SpConstraint = SpMat;

  class AssociationGraph;

  struct BlockPair {
    Block::Ptr ref, cur;
    SceneGraph::Ptr gref, gcur;
    Association association;

    BlockPair(const Block::Ptr& _ref, const Block::Ptr& _cur, 
              const SceneGraph::Ptr& _gref, const SceneGraph::Ptr& _gcur,
              const Association& _association)
    : ref(_ref), cur(_cur), gref(_gref), gcur(_gcur), association(_association) {}
  };

  struct MetricParams {
    double sigma = 0.01; ///< spread / "variance" of exponential kernel
    double epsilon = 0.06; ///< bound on consistency score, determines if inlier/outlier
    std::vector<double> epsilon_list = {0.06}; ///<multiple bound on consistency score, determines if inlier/outli
    double mindist = 0; ///< minimum allowable distance between inlier points in the same dataset
  };

  /**
   * @brief      Clipper parameters
   */
  struct Params {

    // \brief Basic gradient descent stopping criteria
    double tol_u = 1e-8; ///< stop when change in u < tol
    double tol_F = 1e-9; ///< stop when change in F < tol
    double tol_Fop = 1e-10; ///< stop when ||dFop|| < tol
    int maxiniters = 200; ///< max num of gradient ascent steps for each d
    int maxoliters = 1000; ///< max num of outer loop iterations to find d

    // \brief Line search parameters
    double beta = 0.25; ///< backtracking step size reduction, in (0, 1)
    int maxlsiters = 99; ///< maximum number of line search iters per grad step

    double eps = 1e-9; ///< numerical threshold around 0

    double affinityeps = 1e-4; ///< sparsity-promoting threshold for affinities
  };

  /**
   * @brief      Data associated with a Clipper dense clique solution
   */
  struct Solution
  {
    double t; ///< duration spent solving [s]
    int ifinal; ///< number of outer iterations before convergence
    std::vector<int> nodes; ///< indices of graph vertices in dense clique
    Eigen::VectorXd u; ///< characteristic vector associated with graph
    double score; ///< value of objective function / largest eigenvalue
  };


namespace utils {
  /**
   * @brief      Produce an nx1 vector where each element is drawn from U[0, 1).
   *
   * @param[in]  n     Dimension of produced vector
   *
   * @return     Uniform random vector
   */
  Eigen::VectorXd randvec(size_t n);

  /**
   * @brief      Find indices of k largest elements of vector (similar to MATLAB find)
   *
   * @param[in]  x     Vector to find large elements in
   * @param[in]  k     How many of the largest elements to find (k > 0)
   *
   * @return     Indices of the largest elements in vector x
   */
  std::vector<int> findIndicesOfkLargest(const Eigen::VectorXd& x, int k);

  /**
   * @brief      Creates an all-to-all association hypothesis
   *
   * @param[in]  n1    Number of items in view 1
   * @param[in]  n2    Number of items in view 2
   *
   * @return     an (n1*n2)x2 association matrix
   */
  inline Association createAllToAll(size_t n1, size_t n2)
  {
    Association A = Association(n1*n2, 2);
    for (size_t i=0; i<n1; ++i) {
      for (size_t j=0; j<n2; ++j) {
        A(j + i*n2, 0) = i;
        A(j + i*n2, 1) = j;
      }
    }
    return A;
  }

  /**
   * @brief      Convenience function to select inlier associations
   *
   * @param[in]  soln  The solution of the dense cluster
   * @param[in]  A     The initial set of associations
   *
   * @return     The subset of associations deemed as inliers via solution
   */
  Association selectInlierAssociations(const Solution& soln, const Association& A);

  /**
 * @brief      Maps a flat index to coordinate of a square symmetric matrix
 *
 * @param[in]  k     The flat index to find the corresponding r,c of
 * @param[in]  n     Dimension of the square, symmetric matrix
 *
 * @return     row, col of a matrix corresponding to flat index k
 */
  std::tuple<size_t,size_t> k2ij(size_t k, size_t n);
}


inline Association createGraffToGraff(const std::vector<SceneGraphNode::Ptr>& L1, const std::vector<SceneGraphNode::Ptr>& L2) {
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


/**
 * A simple undirected graph class
 *
 * This graph assumes that vertices are numbered. In addition, the vertices numbers have to be
 * consecutive starting from 0.
 *
 * For example, if the graph have 3 vertices, they have to be named 0, 1, and 2.
 */
class AdjGraph {

public:
    using Ptr = boost::shared_ptr<AdjGraph>;

    // using SpMat = Eigen::SparseMatrix<double>;

    // using Association = Eigen::Matrix<int, Eigen::Dynamic, 2>;
    // using Affinity = Eigen::MatrixXd;
    // using Constraint = Eigen::MatrixXd;

    // using SpAffinity = SpMat;
    // using SpConstraint = SpMat;


    AdjGraph() : num_edges_(0) {
        use_adj_matrix_ = false;
    };

    /**
     * Constructor that takes in an adjacency list. Notice that for an edge connecting two arbitrary
     * vertices v1 & v2, we assume that v2 exists in v1's list, and v1 also exists in v2's list. This
     * condition is not enforced. If violated, removeEdge() function might exhibit undefined
     * behaviors.
     * @param [in] adj_list an map representing an adjacency list
     */
    explicit AdjGraph(const std::map<int, std::vector<int>> &adj_list) {
        adj_list_.resize(adj_list.size());
        num_edges_ = 0;
        for (const auto &e_list: adj_list) {
            const auto &v = e_list.first;
            adj_list_[e_list.first] = e_list.second;
            num_edges_ += e_list.second.size();
        }
        num_edges_ /= 2;
    };

    /**
     * Add a vertex with no edges.
     * @param [in] id the id of vertex to be added
     */
    void addVertex(const int &id) {
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
    void populateVertices(const int &num_vertices) {
        adj_list_.resize(num_vertices);
        if (use_adj_matrix_) {
            M_ = Eigen::MatrixXd::Zero(num_vertices, num_vertices);
        }
    }

    /**
     * Return true if said edge exists
     * @param [in] vertex_1
     * @param [in] vertex_2
     */
    bool hasEdge(const int &vertex_1, const int &vertex_2) {
        if (vertex_1 >= adj_list_.size() || vertex_2 >= adj_list_.size()) {
            return false;
        }

        if (use_adj_matrix_) {
            if (vertex_1 <= vertex_2) {
                return M_(vertex_1, vertex_2) != 0;
            } else {
                return M_(vertex_2, vertex_1) != 0;
            }
        }

        auto &connected_vs = adj_list_[vertex_1];
        bool exists =
                std::find(connected_vs.begin(), connected_vs.end(), vertex_2) != connected_vs.end();
        return exists;
    }

    /**
     * Return true if the vertex exists.
     * @param vertex
     * @return
     */
    bool hasVertex(const int &vertex) { return vertex < adj_list_.size(); }

    /**
     * Add an edge between two vertices
     * @param [in] vertex_1 one vertex of the edge
     * @param [in] vertex_2 another vertex of the edge
     */
    void addEdge(const int &vertex_1, const int &vertex_2) {
        if (hasEdge(vertex_1, vertex_2)) {
            // TEASER_DEBUG_ERROR_MSG("Edge exists.");
            return;
        }
        adj_list_[vertex_1].push_back(vertex_2);
        adj_list_[vertex_2].push_back(vertex_1);
        num_edges_++;
        if (use_adj_matrix_){
            if (vertex_1 <= vertex_2) {
                M_(vertex_1, vertex_2) = 1.0;
            } else{
                M_(vertex_2, vertex_1) = 1.0;
            }
        }
    }

    /**
     * Add an edge between two vertices
     * @param [in] vertex_1 one vertex of the edge
     * @param [in] vertex_2 another vertex of the edge
     */
    void addEdge(const int &vertex_1, const int &vertex_2, const double &weight) {
        if (hasEdge(vertex_1, vertex_2)) {
            // TEASER_DEBUG_ERROR_MSG("Edge exists.");
            return;
        }
        adj_list_[vertex_1].push_back(vertex_2);
        adj_list_[vertex_2].push_back(vertex_1);
        num_edges_++;
        if (use_adj_matrix_) {
            if (vertex_1 <= vertex_2) {
                M_(vertex_1, vertex_2) = weight;
            } else{
                M_(vertex_2, vertex_1) = weight;
            }
        }
    }

    /**
     * Remove the edge between two vertices.
     * @param [in] vertex_1 one vertex of the edge
     * @param [in] vertex_2 another vertex of the edge
     */
    void removeEdge(const int &vertex_1, const int &vertex_2) {
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
        if (use_adj_matrix_){
            if (vertex_1 <= vertex_2) {
                M_(vertex_1, vertex_2) = 0;
            } else{
                M_(vertex_2, vertex_1) = 0;
            }
        }
    }

    void pruneGraph(const int &degree) {
    //    remove the edges of the vertices with degree less than degree
        for (int i = 0; i < adj_list_.size(); ++i) {
            if (adj_list_[i].size() < degree) {
                for (const auto &v: adj_list_[i]) {
                    removeEdge(i, v);
                }
            }
        }
        if (use_adj_matrix_)
            initAffinityMatrix();
    }

    void printStatistics() {
        std::cout << "Number of vertices: " << numVertices() << std::endl;
        std::cout << "Number of edges: " << numEdges() << std::endl;

        std::map<int, int> degree_count;
        for (int i = 0; i < adj_list_.size(); ++i) {
            if (degree_count.find(adj_list_[i].size()) == degree_count.end()) {
                degree_count[adj_list_[i].size()] = 1;
            } else {
                degree_count[adj_list_[i].size()]++;
            }
        }
        std::cout << "Degree distribution: ";
        int total_degree = 0;
        for (const auto &dc: degree_count) {
            std::cout << dc.first << ": " << dc.second << ", ";
            total_degree += dc.first * dc.second;
        }
        std::cout << std::endl;
        std::cout << "Average degree: " << total_degree / adj_list_.size() << ", total degree: " << total_degree << std::endl;
    }

    void setType(const bool &use_adj_matrix) { use_adj_matrix_ = use_adj_matrix; }

    void initAffinityMatrix(){
        if (!use_adj_matrix_) return;
        // make the diagonal elements of M_ to be 0
        for (int i = 0; i < M_.rows(); ++i) {
            M_(i, i) = 0;
        }
        affinity_ = M_.sparseView();
        constraint_ = affinity_;
        constraint_.coeffs() = 1;
    }

    SpAffinity affinity() const { return affinity_; }

    SpConstraint constraint() const { return constraint_; }

    void setAffinity(const SpAffinity &affinity) { affinity_ = affinity; }

    void setConstraint(const SpConstraint &constraint) { constraint_ = constraint; }

    /**
     * Get the number of vertices
     * @return total number of vertices
     */
    // [[nodiscard]] tells the compiler if return value of the function is not used
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
    [[nodiscard]] const std::vector<int> &getEdges(int id) const { return adj_list_[id]; }

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
            const auto &c_edges = getEdges(i);
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

    /**
     * Preallocate spaces for vertices
     * @param num_vertices
     */
    void reserve(const int &num_vertices) { adj_list_.reserve(num_vertices); }

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
    void reserveForCompleteGraph(const int &num_vertices) {
        adj_list_.reserve(num_vertices);
        for (int i = 0; i < num_vertices - 1; ++i) {
            std::vector<int> c_edges;
            c_edges.reserve(num_vertices - 1);
            adj_list_.push_back(c_edges);
        }
        adj_list_.emplace_back(std::initializer_list<int>{});
    }

    // void saveToFile(const std::string& file_path) {
    //     std::ofstream out_file(file_path);
    //     if (!out_file.is_open()) {
    //         TEASER_DEBUG_ERROR_MSG("Cannot open file " + file_path);
    //         return;
    //     }
    //     out_file << numVertices() << " " << numEdges() * 2 << std::endl;
    //     for (int i = 0; i < numVertices(); ++i) {
    //         const auto& c_edges = getEdges(i);
    //         for (const auto& e : c_edges) {
    //             out_file << i << " " << e << std::endl;
    //         }
    //     }
    //     out_file.close();
    // }

    // void loadFromFile(const std::string& file_path) {
    //     std::ifstream in_file(file_path);
    //     if (!in_file.is_open()) {
    //         TEASER_DEBUG_ERROR_MSG("Cannot open file " + file_path);
    //         return;
    //     }
    //     clear();
    //     int num_v, num_e;
    //     in_file >> num_v >> num_e;
    //     populateVertices(num_v);
    //     for (int i = 0; i < num_e; ++i) {
    //         int v1, v2;
    //         in_file >> v1 >> v2;
    //         addEdge(v1, v2);
    //     }
    //     in_file.close();
    // }

private:
    std::vector<std::vector<int>> adj_list_;
    size_t num_edges_;
    SpAffinity affinity_;
    SpConstraint constraint_;
    Eigen::MatrixXd M_;
    bool use_adj_matrix_ = false;
};


class MACSolver {
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
   * Parameter struct for MACSolver
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
    double kcore_heuristic_threshold = 0.5;

    /**
     * Time limit on running the solver.
     */
    double time_limit = 3600;
  };

  MACSolver() = default;

  MACSolver(Params params) : params_(params) {};

  /**
   * Find the maximum clique within the graph provided. By maximum clique, it means the clique of
   * the largest size in an undirected graph.
   * @param graph
   * @return a vector of indices of cliques
   */
  std::vector<int> findMaxClique(AdjGraph graph, int lower_bound = 0) {
    // Handle deprecated field
    if (!params_.solve_exactly) {
      params_.solver_mode = CLIQUE_SOLVER_MODE::PMC_HEU;
    }

    // Create a PMC graph from the TEASER graph
    int num_vertices = graph.numVertices();
    vector<int> edges;
    edges.reserve(graph.numEdges() * 2);
    vector<long long> vertices;
    vertices.reserve(num_vertices + 1);

    vertices.push_back(0);
    for (size_t i = 0; i < num_vertices; ++i) {
      const auto &c_edges = graph.getEdges(i);
      edges.insert(edges.end(), c_edges.begin(), c_edges.end());
      vertices.push_back(edges.size());
    }

    // Use PMC to calculate
    pmc::pmc_graph G(vertices, edges); // typically takes 0.005 ms
    // upper-bound of max clique
    G.compute_cores();
    int max_core = G.get_max_core(); // typically takes 0.040 ms, get the upper bound of clique size

    vector<int> C; // vector to represent max clique
    if (params_.solver_mode == CLIQUE_SOLVER_MODE::PMC_EXACT){
      pmc::input in; // use default input
      in.time_limit = params_.time_limit;
      in.threads = 12;
      in.lb = lower_bound;
      in.ub = in.ub == 0 ? max_core + 1 : in.ub;

      // lower-bound of max clique
      pmc::pmc_heu maxclique(G, in);
      in.lb = maxclique.search(G, C); // typically takes 0.120 ms

      // This means that max clique has a size of one
      if (in.lb == 0) return C; //error

      if (in.lb == in.ub) return C;

      if (G.num_vertices() < in.adj_limit) {
        G.create_adj();
        pmc::pmcx_maxclique finder(G, in);
        finder.search_dense(G, C);
      } else {
        std::cout << "PMC: Graph is too dense, so don't use adj matrix to speed up" << std::endl;
        pmc::pmcx_maxclique finder(G, in);
        finder.search(G, C);
      }
    } else if (params_.solver_mode == CLIQUE_SOLVER_MODE::KCORE_HEU){
      // check for k-core heuristic threshold
      // check whether threshold equals 1 to short circuit the comparison
      if (params_.kcore_heuristic_threshold != 1 && max_core > params_.kcore_heuristic_threshold * num_vertices){
        // TEASER_DEBUG_INFO_MSG("Using K-core heuristic finder.");
        // remove all nodes with core number less than max core number
        // k_cores is a vector saving the core number of each vertex
        auto k_cores = G.get_kcores();
        for (int i = 1; i < k_cores->size(); ++i) {
          // Note: k_core has size equals to num vertices + 1
          if ((*k_cores)[i] >= max_core) {
            C.push_back(i - 1);
          }
        }
        return C;
      }
    }

    return C;
  }

 private:
  AdjGraph graph_;
  Params params_;
};


/**
 * @brief      Convenience class to use Clipper for data association.
 */
class Clipper
{
public:
  // Clipper(const invariants::PairwiseInvariantPtr& invariant, const Params& params);
  // Clipper(const invariants::GraffDistancePtr& graff_func, const Params& params);
  Clipper(const Params& params, const MetricParams& metric_params);
  ~Clipper() = default;

  /**
 * @brief      Creates an affinity matrix containing consistency scores for
 *             each of the m pairwise associations listed in matrix A.
 *
 * @param[in]  D1           Dataset 1 of n1 d-dim elements (dxn1)
 * @param[in]  D2           Dataset 2 of n2 d-dim elements (dxn2)
 * @param[in]  A            Associations to score (mx2)
 */
  // void scorePairwiseConsistency(const std::vector<)

  void scoreGraffConsistency(const SLIM::SceneGraph::Ptr& g1,
                            const SLIM::SceneGraph::Ptr& g2,
                              const Association& A = Association());
                    
  void buildMutualAffinity(const SceneGraph::Ptr &g1, const SceneGraph::Ptr& g2, const Association& A = Association());

  void buildTruncatedMutualAffinity(const SceneGraph::Ptr &g1, const SceneGraph::Ptr& g2, const Association& A = Association());

  void solveRelaxtion(const Eigen::VectorXd& u0 = Eigen::VectorXd());

  bool solveRelativePose(const SceneGraph::Ptr& gref, 
                        const SceneGraph::Ptr& gcur,
                        const Association& matches,
                        Transform& Trc);

  const Solution& getSolution() const { return soln_; }
  Affinity getAffinityMatrix();
  Constraint getConstraintMatrix();

  /**
   * @brief      Skip using scorePairwiseConsistency and directly set the
   *             affinity and constraint matrices. Note that this function
   *             accepts dense matrices. Use the sparse version for better
   *             performance if you already have sparse matrices available.
   *
   * @param[in]  M     Affinity matrix
   * @param[in]  C     Constraint matrix
   */
  void setMatrixData(const Affinity& M, const Constraint& C);

  /**
   * @brief      Skip using scorePairwiseConsistency and directly set the
   *             affinity and constraint matrices. Note that this function
   *             accepts sparse matrices. These matrices should be upper
   *             triangular and should not have diagonal values set.
   *
   * @param[in]  M     Affinity matrix
   * @param[in]  C     Constraint matrix
   */
  void setSparseMatrixData(const SpAffinity& M, const SpConstraint& C);

  Association getInitialAssociations();
  Association getSelectedAssociations();

  void setParallelize(bool parallelize) { parallelize_ = parallelize; };

private:
  Params params_;
  MetricParams metric_params_;
  // invariants::PairwiseInvariantPtr invariant_;
  // invariants::GraffDistancePtr graff_func_;

  bool parallelize_ = true; ///< should affinity calculation be parallelized

  Association A_; ///< initial (putative) set of associations

  // \brief Problem data from latest instance of data association
  Solution soln_; ///< solution information from Clipper dense clique solver
  SpAffinity M_; ///< affinity matrix (i.e., weighted consistency graph)
  SpConstraint C_; ///< constraint matrix (i.e., prevents forming links)
  std::vector<SpAffinity> Ms_;
  std::vector<AdjGraph> inlier_graphs_;
  std::vector<std::vector<int>> max_cliques_;
  MACSolver mac_solver_;

  /**
   * @brief      Identifies a dense clique of an undirected graph G from its
   *             weighted affinity matrix M while satisfying any active
   *             constraints in C (indicated with zeros).
   *
   *             If M is binary and C==M then Clipper returns a maximal clique.
   *
   *             This algorithm employs a projected gradient descent method to
   *             solve a symmetric rank-one nonnegative matrix approximation.
   *
   * @param[in]  M        Symmetric, non-negative nxn affinity matrix where
   *                      each element is in [0,1]. Nodes can also be weighted
   *                      between [0,1] (e.g., if there is a prior indication
   *                      that a node belongs to the desired cluster). In the
   *                      case that all nodes are equally likely to be in the
   *                      densest cluster/node weights should not be considered
   *                      set the diagonal of M to identity.
   * @param[in]  C        nxn binary constraint matrix. Active const. are 0.
   */
  void findDenseClique(const Eigen::VectorXd& u0);
};

} // ns clipper
} // namespace SLIM