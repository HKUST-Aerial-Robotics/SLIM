/**
 * @file clipper.cpp
 * @brief Clipper data association framework
 * @author Parker Lusk <plusk@mit.edu>
 * @date 19 March 2022
 */

#include <iostream>

#include "clipper.h"

namespace SLIM {
namespace clipper {
namespace utils {

Eigen::VectorXd randvec(size_t n)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0, 1);

  return Eigen::VectorXd::NullaryExpr(n, 1, [&](){ return dis(gen); });
}

// ----------------------------------------------------------------------------

std::vector<int> findIndicesOfkLargest(const Eigen::VectorXd& x, int k)
{
  using T = std::pair<double, int>; // pair value to be compared and index
  if (k < 1) return {}; // invalid input
  // n.b., the top of this queue is smallest element
  std::priority_queue<T, std::vector<T>, std::greater<T>> q;
  for (size_t i=0; i<x.rows(); ++i) {
    if (q.size() < k) {
      q.push({x(i), i});
    } else if (q.top().first < x(i)) {
      q.pop();
      q.push({x(i), i});
    }
  }

  std::vector<int> indices(k);
  for (size_t i=0; i<k; ++i) {
    indices[k - i - 1] = q.top().second;
    q.pop();
  }

  return indices;
}

// ----------------------------------------------------------------------------

std::tuple<size_t,size_t> k2ij(size_t k, size_t n)
{
  k += 1;

  const size_t l = n*(n-1)/2 - k;
  const size_t o = std::floor( (std::sqrt(1 + 8*l) - 1) / 2. );
  const size_t p = l - o*(o+1)/2;
  const size_t i = n - (o + 1);
  const size_t j = n - p;
  return {i-1, j-1};
}



// vector<int> MACSolver::findMaxClique(AdjGraph graph, int lower_bound) {

//   // Handle deprecated field
//   if (!params_.solve_exactly) {
//     params_.solver_mode = CLIQUE_SOLVER_MODE::PMC_HEU;
//   }

//   // Create a PMC graph from the TEASER graph
//   int num_vertices = graph.numVertices();
//   vector<int> edges;
//   edges.reserve(graph.numEdges() * 2);
//   vector<long long> vertices;
//   vertices.reserve(num_vertices + 1);

//   vertices.push_back(0);
//   for (size_t i = 0; i < num_vertices; ++i) {
//       const auto &c_edges = graph.getEdges(i);
//       edges.insert(edges.end(), c_edges.begin(), c_edges.end());
//       vertices.push_back(edges.size());
//   }

//   // Use PMC to calculate
//   pmc::pmc_graph G(vertices, edges); // typically takes 0.005 ms
//   // upper-bound of max clique
//   G.compute_cores();
//   int max_core = G.get_max_core(); // typically takes 0.040 ms, get the upper bound of clique size

//   vector<int> C; // vector to represent max clique
//   if (params_.solver_mode == CLIQUE_SOLVER_MODE::PMC_EXACT){
//       pmc::input in; // use default input
//       in.time_limit = params_.time_limit;
//       in.threads = 12;
//       in.lb = lower_bound;
//       in.ub = in.ub == 0 ? max_core + 1 : in.ub;

//       // lower-bound of max clique
//       pmc::pmc_heu maxclique(G, in);
//       in.lb = maxclique.search(G, C); // typically takes 0.120 ms

//       // This means that max clique has a size of one
//       if (in.lb == 0) return C; //error

//       if (in.lb == in.ub) return C;

//       if (G.num_vertices() < in.adj_limit) {
//           G.create_adj();
//           pmc::pmcx_maxclique finder(G, in);
//           finder.search_dense(G, C);
//       } else {
//           std::cout << "PMC: Graph is too dense, so don't use adj matrix to speed up" << std::endl;
//           pmc::pmcx_maxclique finder(G, in);
//           finder.search(G, C);
//       }
//   } else if (params_.solver_mode == CLIQUE_SOLVER_MODE::KCORE_HEU){
//       // check for k-core heuristic threshold
//       // check whether threshold equals 1 to short circuit the comparison
//       if (params_.kcore_heuristic_threshold != 1 && max_core > params_.kcore_heuristic_threshold * num_vertices){
//           // TEASER_DEBUG_INFO_MSG("Using K-core heuristic finder.");
//           // remove all nodes with core number less than max core number
//           // k_cores is a vector saving the core number of each vertex
//           auto k_cores = G.get_kcores();
//           for (int i = 1; i < k_cores->size(); ++i) {
//               // Note: k_core has size equals to num vertices + 1
//               if ((*k_cores)[i] >= max_core) {
//                   C.push_back(i - 1);
//               }
//           }
//           return C;
//       }
//   }

//   return C;
// }

// ----------------------------------------------------------------------------

Association selectInlierAssociations(const Solution& soln, const Association& A)
{
  Association Ainliers = Association::Zero(soln.nodes.size(), 2);
  for (size_t i=0; i<soln.nodes.size(); ++i) {
    Ainliers.row(i) = A.row(soln.nodes[i]);
  }
  return Ainliers;
}
} // ns utils

Clipper::Clipper(const Params& params, const MetricParams& metric_params)
: params_(params), metric_params_(metric_params) {
  if(metric_params.epsilon_list.empty()) {
    inlier_graphs_.resize(1);
    max_cliques_.resize(1);
  }
  else {
    inlier_graphs_.resize(metric_params.epsilon_list.size());
    max_cliques_.resize(metric_params.epsilon_list.size());
  }
}

// ----------------------------------------------------------------------------

void Clipper::scoreGraffConsistency(const SLIM::SceneGraph::Ptr& g1,
                                    const SLIM::SceneGraph::Ptr& g2,
                                    const Association& A) 
{
  auto N1 = g1->nodes_;
  auto N2 = g2->nodes_;
  if (A.size() == 0) 
    A_ = createGraffToGraff(N1, N2);
  else 
    A_ = A;
  const size_t m = A_.rows();

  auto &afm1 = g1->self_affinity_mat_;
  auto &afm2 = g2->self_affinity_mat_;

  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(m, m);
// #pragma omp parallel for shared(A_, N1, N2, M_, C_) if(parallelize_)
  for (size_t k=0; k<m*(m-1)/2; ++k) {
    size_t i, j; std::tie(i, j) = utils::k2ij(k, m);

    if (A_(i,0) == A_(j,0) || A_(i,1) == A_(j,1)) {
      // violates distinctness constraint
      continue;
    }
    
    Eigen::MatrixXd g1i = N1[A_(i,0)]->graff_coord_;
    Eigen::MatrixXd g1j = N1[A_(j,0)]->graff_coord_;
    Eigen::MatrixXd g2i = N2[A_(i,1)]->graff_coord_;
    Eigen::MatrixXd g2j = N2[A_(j,1)]->graff_coord_;

    const Eigen::Vector3d b01i = g1i.topRightCorner(3, 1);
    g1i.topRightCorner(3, 1).setZero();
    g1j.topRightCorner(3, 1) -= b01i;
    const Eigen::Vector3d b01j = g1j.topRightCorner(3, 1);
    g1j.rightCols<1>() /= std::sqrt(1 + b01j.squaredNorm());

    const Eigen::Vector3d b02i = g2i.topRightCorner(3, 1);
    g2i.topRightCorner(3, 1).setZero();
    g2j.topRightCorner(3, 1) -= b02i;
    const Eigen::Vector3d b02j = g2j.topRightCorner(3, 1);
    g2j.rightCols<1>() /= std::sqrt(1 + b02j.squaredNorm());

    auto s1 = Eigen::JacobiSVD<Eigen::MatrixXd>(g1i.transpose() * g1j).singularValues();
    auto s2 = Eigen::JacobiSVD<Eigen::MatrixXd>(g2i.transpose() * g2j).singularValues();

    

    const double l1 = s1.norm();
    const double l2 = s2.norm();

    if (l1 != afm1[A_(i,0)][A_(j,0)]) {
      printf("[Warning] L1: %lf, Pre-comp L1: %lf\n", l1, afm1[A_(i,0)][A_(j,0)]);
    }

    if (l2 != afm1[A_(i,1)][A_(j,1)]) {
      printf("[Warning] L1: %lf, Pre-comp L1: %lf\n", l2, afm1[A_(i,1)][A_(j,1)]);
    }

    // double l1 = 0.0, l2 = 0.0;
    // for(int i = 0; i < s1.rows(); ++i) {
    //   double li = std::acos(s1(i));
    //   l1 += li*li;
    // }
    // l1 = std::sqrt(l1);
    // for(int i = 0; i < s2.rows(); ++i) {
    //   double li = std::acos(s2(i));
    //   l2 += li*li;
    // }
    // l2 = std::sqrt(l2);

    // enforce minimum distance criterion -- if points in the same dataset are too close, then this pair of associations cannot be selected
    if (metric_params_.mindist > 0 && (l1 < metric_params_.mindist || l2 < metric_params_.mindist))
      continue;

    // consistency score
    const double c = std::abs(l1 - l2);
    const double scr = (c<metric_params_.epsilon) ? std::exp(-0.5*c*c/(metric_params_.sigma*metric_params_.sigma)) : 0;
    if (scr > params_.affinityeps) { // does not violate inconsistency constraint
      M(i,j) = scr;
    }
  }
  M_ = M.sparseView();
  C_ = M_;
  C_.coeffs() = 1;
}


void Clipper::buildMutualAffinity(const SceneGraph::Ptr &g1, const SceneGraph::Ptr& g2, const Association& A) {
  if (A.size() == 0) 
    A_ = createGraffToGraff(g1->nodes_, g2->nodes_);
  else 
    A_ = A;
    
  const size_t m = A_.rows();
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(m, m);
  auto &afm1 = g1->self_affinity_mat_;
  auto &afm2 = g2->self_affinity_mat_;
#pragma omp parallel for shared(A_, afm1, afm2, M_, C_) if(parallelize_)
  for (size_t k=0; k<m*(m-1)/2; ++k) {
    size_t i, j; std::tie(i, j) = utils::k2ij(k, m);

    if (A_(i,0) == A_(j,0) || A_(i,1) == A_(j,1)) {
      // violates distinctness constraint
      continue;
    }

    const auto l1 = afm1[A_(i,0)][A_(j,0)];
    const auto l2 = afm2[A_(i,1)][A_(j,1)];
    
    // enforce minimum distance criterion -- if points in the same dataset are too close, then this pair of associations cannot be selected
    if (metric_params_.mindist > 0 && (l1 < metric_params_.mindist || l2 < metric_params_.mindist))
      continue;

    // consistency score
    const double c = std::abs(l1 - l2);
    const double scr = (c<metric_params_.epsilon) ? std::exp(-0.5*c*c/(metric_params_.sigma*metric_params_.sigma)) : 0;
    if (scr > params_.affinityeps) { // does not violate inconsistency constraint
      M(i,j) = scr;
    }
  }
  M_ = M.sparseView();
  C_ = M_;
  C_.coeffs() = 1;
}

void Clipper::buildTruncatedMutualAffinity(const SceneGraph::Ptr &g1, const SceneGraph::Ptr& g2, const Association& A) {
  if (A.size() == 0) 
    A_ = createGraffToGraff(g1->nodes_, g2->nodes_);
  else 
    A_ = A;
    
  const size_t num_corr = A_.rows();
  int num_tims = num_corr * (num_corr - 1) / 2;
  inlier_graphs_[0].populateVertices(num_corr);
  auto &afm1 = g1->self_affinity_mat_;
  auto &afm2 = g2->self_affinity_mat_;

#pragma omp parallel for shared(num_corr, num_tims, afm1, afm2, inlier_graphs_, A_) if(parallelize_)
  for (size_t k = 0; k < num_tims; ++k) {
    size_t i, j; std::tie(i, j) = utils::k2ij(k, num_corr);

    if (A_(i,0) == A_(j,0) || A_(i,1) == A_(j,1)) {
      // violates distinctness constraint
      continue;
    }
    const auto l1 = afm1[A_(i,0)][A_(j,0)];
    const auto l2 = afm2[A_(i,1)][A_(j,1)];
    const double c = std::abs(l1 - l2);
    inlier_graphs_[0].addEdge(i, j, 0.99);
  }
  max_cliques_[0] = mac_solver_.findMaxClique(inlier_graphs_[0], 3);
  soln_.nodes = max_cliques_[0];
}


// ----------------------------------------------------------------------------

void Clipper::solveRelaxtion(const Eigen::VectorXd& _u0)
{
  Eigen::VectorXd u0;
  if (_u0.size() == 0) {
    u0 = utils::randvec(M_.cols());
  } else {
    u0 = _u0;
  }

  
  findDenseClique(u0);
}

// ----------------------------------------------------------------------------

Association Clipper::getInitialAssociations()
{
  return A_;
}

// ----------------------------------------------------------------------------

Association Clipper::getSelectedAssociations()
{
  return utils::selectInlierAssociations(soln_, A_);
}

// ----------------------------------------------------------------------------

Affinity Clipper::getAffinityMatrix()
{
  Affinity M = SpAffinity(M_.selfadjointView<Eigen::Upper>())
                + Affinity::Identity(M_.rows(), M_.cols());
  return M;
}

// ----------------------------------------------------------------------------

Constraint Clipper::getConstraintMatrix()
{
  Constraint C = SpConstraint(C_.selfadjointView<Eigen::Upper>())
                  + Constraint::Identity(C_.rows(), C_.cols());
  return C;
}

// ----------------------------------------------------------------------------

void Clipper::setMatrixData(const Affinity& M, const Constraint& C)
{
  Eigen::MatrixXd MM = M.triangularView<Eigen::Upper>();
  MM.diagonal().setZero();
  M_ = MM.sparseView();

  Eigen::MatrixXd CC = C.triangularView<Eigen::Upper>();
  CC.diagonal().setZero();
  C_ = CC.sparseView();
}

// ----------------------------------------------------------------------------

void Clipper::setSparseMatrixData(const SpAffinity& M, const SpConstraint& C)
{
  M_ = M;
  C_ = C;
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

void Clipper::findDenseClique(const Eigen::VectorXd& u0)
{
  const auto t1 = std::chrono::high_resolution_clock::now();

  //
  // Initialization
  //

  const size_t n = M_.cols();
  const Eigen::VectorXd ones = Eigen::VectorXd::Ones(n);

  // one step of power method to have a good scaling of u
  Eigen::VectorXd u = M_.selfadjointView<Eigen::Upper>() * u0 + u0;
  u /= u.norm();

  // initial value of d
  double d = 0; // zero if there are no active constraints
  Eigen::VectorXd Cbu = ones * u.sum() - C_.selfadjointView<Eigen::Upper>() * u - u;
  const auto idxD = ((Cbu.array()>params_.eps) && (u.array()>params_.eps));
  if (idxD.sum() > 0) {
    Eigen::VectorXd Mu = M_.selfadjointView<Eigen::Upper>() * u + u;
    const Eigen::VectorXd num = idxD.select(Mu, std::numeric_limits<double>::infinity());
    const Eigen::VectorXd den = idxD.select(Cbu, 1);
    d = (num.array() / den.array()).minCoeff();
  }

  // initialize memory
  Eigen::VectorXd gradF = Eigen::VectorXd(n);
  Eigen::VectorXd gradFnew = Eigen::VectorXd(n);
  Eigen::VectorXd unew = Eigen::VectorXd(n);
  Eigen::VectorXd Mu = Eigen::VectorXd(n);
  Eigen::VectorXd num = Eigen::VectorXd(n);
  Eigen::VectorXd den = Eigen::VectorXd(n);

  //
  // Orthogonal projected gradient ascent with homotopy
  //

  double F = 0; // objective value

  size_t i, j, k; // iteration counters
  for (i=0; i<params_.maxoliters; ++i) {
    gradF = (1 + d) * u - d * ones * u.sum() + M_.selfadjointView<Eigen::Upper>() * u + C_.selfadjointView<Eigen::Upper>() * u * d;
    F = u.dot(gradF); // current objective value

    //
    // Orthogonal projected gradient ascent
    //

    for (j=0; j<params_.maxiniters; ++j) {
      double alpha = 1;

      //
      // Backtracking line search on gradient ascent
      //

      double Fnew = 0, deltaF = 0;
      for (k=0; k<params_.maxlsiters; ++k) {
        unew = u + alpha * gradF;                     // gradient step
        unew = unew.cwiseMax(0);                      // project onto positive orthant
        unew.normalize();                             // project onto S^n
        gradFnew = (1 + d) * unew // because M/C is missing identity on diagonal
                    - d * ones * unew.sum()
                    + M_.selfadjointView<Eigen::Upper>() * unew
                    + C_.selfadjointView<Eigen::Upper>() * unew * d;
        Fnew = unew.dot(gradFnew);                    // new objective value after step

        deltaF = Fnew - F;                            // change in objective value

        if (deltaF < -params_.eps) {
          // objective value decreased---we need to backtrack, so reduce step size
          alpha = alpha * params_.beta;
        } else {
          break; // obj value increased, stop line search
        }
      }
      const double deltau = (unew - u).norm();

      // update values
      F = Fnew;
      u = unew;

      // check if desired accuracy has been reached by gradient ascent 
      if (deltau < params_.tol_u || std::abs(deltaF) < params_.tol_F) break;
    }

    //
    // Increase d
    //

    Cbu = ones * u.sum() - C_.selfadjointView<Eigen::Upper>() * u - u;
    const auto idxD = ((Cbu.array() > params_.eps) && (u.array() > params_.eps));
    if (idxD.sum() > 0) {
      Mu = M_.selfadjointView<Eigen::Upper>() * u + u;
      num = idxD.select(Mu, std::numeric_limits<double>::infinity());
      den = idxD.select(Cbu, 1);
      const double deltad = (num.array() / den.array()).abs().minCoeff();

      d += deltad;

    } else {
      break;
    }
  }

  //
  // Generate output
  //

  // estimate cluster size using largest eigenvalue
  const int omega = std::round(F);

  // extract indices of nodes in identified dense cluster
  std::vector<int> I = utils::findIndicesOfkLargest(u, omega);

  const auto t2 = std::chrono::high_resolution_clock::now();
  const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);
  const double elapsed = static_cast<double>(duration.count()) / 1e9;

  // set solution
  soln_.t = elapsed;
  soln_.ifinal = i;
  std::swap(soln_.nodes, I);
  soln_.u.swap(u);
  soln_.score = F;
}

bool Clipper::solveRelativePose(const SceneGraph::Ptr& gref, 
                                const SceneGraph::Ptr& gcur,
                                const Association& matches,
                                Transform& Trc) {

  const int size = matches.rows();

  ceres::Problem problem;
  ceres::LossFunction *huber_loss = new ceres::HuberLoss(1.0);
  ceres::LocalParameterization *pose_local_param = new PoseLocalParameterization();  
  Trc.SetIdentity();

  problem.AddParameterBlock(Trc.parameters().data(), 7, pose_local_param);

  for(int i = 0; i < size; ++i) {
    SceneGraphNode::Ptr const& ref_node = gref->nodes_[matches(i, 0)];
    SceneGraphNode::Ptr const& cur_node = gcur->nodes_[matches(i, 1)];
    auto const geo_type = ref_node->geometry_type();
    if(geo_type == SceneGraphNode::GeoType::LINE) {
      LaserPointToLineFactor *f = new LaserPointToLineFactor(cur_node->node_value(), ref_node->node_value(), ref_node->normal_value(), 1/0.2);
      problem.AddResidualBlock(f, huber_loss, Trc.parameters().data());
    }
    else if(geo_type == SceneGraphNode::GeoType::SURFACE) {
      LaserPointToSurfaceFactor *f = new LaserPointToSurfaceFactor(cur_node->node_value(), ref_node->node_value(), ref_node->normal_value(), 1/0.2);
      problem.AddResidualBlock(f, huber_loss, Trc.parameters().data());
    }
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = 50;
  options.function_tolerance = 1e-4;
  options.gradient_tolerance = 1e-4;
  // options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  return true;
}

} // ns clipper
} // namespace SLIM