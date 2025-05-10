#include "pcm_solver.h"

namespace SLIM {

void AssociationGraph::build(const std::vector<OverlapInfo>& info) {

  int mutual_affinity_size = info.size();
  // printf("[AssociationGraph][build] MutualAffinity Size: %d\n", mutual_affinity_size);
  Eigen::MatrixXf mutual_affinity_mat(mutual_affinity_size, mutual_affinity_size);
  Eigen::MatrixXf mutual_affinity_binary_mat(mutual_affinity_size, mutual_affinity_size);
  Eigen::MatrixXf mutual_affinity_r_mat(mutual_affinity_size, mutual_affinity_size);
  mutual_affinity_mat.setZero();
  mutual_affinity_binary_mat.setZero();
  mutual_affinity_r_mat.setZero(); 

  std::map<int, std::vector<int>> adj_list;
  for(int k = 0; k < mutual_affinity_size; ++k) {
    for(int l = k; l < mutual_affinity_size; ++l) {
      if(k == l) {
        mutual_affinity_binary_mat(k, l) = 1;
        continue;
      }
      Frame::Ptr const fki = info[k].bi->getHostFrame();
      Frame::Ptr const fkj = info[k].bj->getHostFrame();
      Frame::Ptr const fli = info[l].bi->getHostFrame();
      Frame::Ptr const flj = info[l].bj->getHostFrame();

      // graph i and graph j, frame ki and frame li are in same map, similarly, frame kj and frame lj are in same map
      Transform const Tkili = fki->Twb().inverse() * fli->Twb();
      Transform const Tkjlj = fkj->Twb().inverse() * flj->Twb();
      Transform const Tkikj = info[k].Tij;
      Transform const Tljli = info[l].Tij.inverse();

      Transform const Terr = Tkili.inverse() * (Tkikj * Tkjlj * Tljli);
      double loss_r = LogSO3(Terr.dcm()).norm();
      double loss_p = Terr.p().norm();
      double dist = std::max((Tkili.p().norm() + Tkjlj.p().norm()) * 0.5, 1.0);
      
      if(loss_r < 0.1 && loss_p / dist < 0.1) {
        mutual_affinity_binary_mat(k, l) = 1;
        mutual_affinity_binary_mat(l, k) = 1;
        adj_list[k].push_back(l);
        adj_list[l].push_back(k);
      }
    }
  }
  // std::cout << mutual_affinity_mat << std::endl;
  // std::cout << std::endl;
  // std::cout << mutual_affinity_r_mat << std::endl;
  // std::cout << std::endl;
  // std::cout << mutual_affinity_binary_mat << std::endl;
  adj_list_.resize(mutual_affinity_size);
  num_edges_ = 0;
  for (const auto& e_list : adj_list) {
    const auto& v = e_list.first;
    adj_list_[e_list.first] = e_list.second;
    num_edges_ += e_list.second.size();
  }
  num_edges_ /= 2;
}

void AssociationGraph::build(const std::vector<BilateralEdge>& info) {
  int mutual_affinity_size = info.size();
  Eigen::MatrixXf mutual_affinity_mat(mutual_affinity_size, mutual_affinity_size);
  Eigen::MatrixXf mutual_affinity_binary_mat(mutual_affinity_size, mutual_affinity_size);
  Eigen::MatrixXf mutual_affinity_r_mat(mutual_affinity_size, mutual_affinity_size);
  mutual_affinity_mat.setZero();
  mutual_affinity_binary_mat.setZero();
  mutual_affinity_r_mat.setZero(); 

  std::map<int, std::vector<int>> adj_list;
  for(int k = 0; k < mutual_affinity_size; ++k) {
    for(int l = k; l < mutual_affinity_size; ++l) {
      if(k == l) {
        mutual_affinity_binary_mat(k, l) = 1;
        continue;
      }
      Frame::Ptr const fki = info[k].fi;
      Frame::Ptr const fkj = info[k].fj;
      Frame::Ptr const fli = info[l].fi;
      Frame::Ptr const flj = info[l].fj;

      // graph i and graph j, frame ki and frame li are in same map, similarly, frame kj and frame lj are in same map
      Transform const Tkili = fki->Twb().inverse() * fli->Twb();
      Transform const Tkjlj = fkj->Twb().inverse() * flj->Twb();
      Transform const Tkikj = info[k].Tij;
      Transform const Tljli = info[l].Tij.inverse();

      Transform const Terr = Tkili.inverse() * (Tkikj * Tkjlj * Tljli);
      double loss_r = LogSO3(Terr.dcm()).norm();
      double loss_p = Terr.p().norm();
      double dist = std::max((Tkili.p().norm() + Tkjlj.p().norm()) * 0.5, 1.0);
      // mutual_affinity_mat(k, l) = loss_p / dist;
      // mutual_affinity_mat(l, k) = loss_p / dist;
      // mutual_affinity_r_mat(k, l) = loss_r;
      // mutual_affinity_r_mat(l, k) = loss_r;

      if(loss_r < 0.05 && loss_p / dist < 0.05) {
        mutual_affinity_binary_mat(k, l) = 1;
        mutual_affinity_binary_mat(l, k) = 1;
        adj_list[k].push_back(l);
        adj_list[l].push_back(k);
      }
    }
  }
  adj_list_.resize(mutual_affinity_size);
  num_edges_ = 0;
  for (const auto& e_list : adj_list) {
    const auto& v = e_list.first;
    adj_list_[e_list.first] = e_list.second;
    num_edges_ += e_list.second.size();
  }
  num_edges_ /= 2;
}

void AssociationGraph::buildMutualAffinity(const SceneGraph::Ptr& g1, const SceneGraph::Ptr& g2) {
  A_ = createMatch(g1->nodes_, g2->nodes_);
  const size_t num_corr = A_.rows();
  int num_tims = num_corr * (num_corr - 1) / 2;
  // inlier_graphs_[0].populateVertices(num_corr);
  auto &afm1 = g1->self_affinity_mat_;
  auto &afm2 = g2->self_affinity_mat_;
  this->populateVertices(num_corr);
  printf("[AssociationGraph::buildMutualAffinity] G1: %ld, G2: %ld\n", g1->nodes_.size(), g2->nodes_.size());

  // std::map<int, std::vector<int>> adj_list; 
// #pragma omp parallel for default(none) shared(num_corr, num_tims, afm1, afm2, adj_list_, num_edges_, A_)
  for (size_t k = 0; k < num_tims; ++k) {
    size_t i, j; std::tie(i, j) = k2ij(k, num_corr);
    if (A_(i,0) == A_(j,0) || A_(i,1) == A_(j,1)) {
      continue;
    }
    const auto l1 = afm1[A_(i,0)][A_(j,0)];
    const auto l2 = afm2[A_(i,1)][A_(j,1)];
    const double c = std::abs(l1 - l2);
    if (c < 0.02) {
      this->addEdge(i, j);
    }
  }
}

Association GlobalRegister::solveMatch(const SceneGraph::Ptr& g1, const SceneGraph::Ptr& g2) {
  A_ = createMatch(g1->nodes_, g2->nodes_);

  const size_t num_corr = A_.rows();
  int num_tims = num_corr * (num_corr - 1) / 2;
  inlier_graphs_[0].populateVertices(num_corr);
  auto &afm1 = g1->self_affinity_mat_;
  auto &afm2 = g2->self_affinity_mat_;

  std::map<int, std::vector<int>> adj_list;
  int edge_num = 0;
  // printf("[GlobalRegister] Build Affinity Matrix!\n");
// #pragma omp parallel for default(none) shared(num_corr, num_tims, afm1, afm2, inlier_graphs_, A_)
  std::map<double, std::pair<size_t, size_t>> edges;
  for (size_t k = 0; k < num_tims; ++k) {
    size_t i, j; std::tie(i, j) = k2ij(k, num_corr);

    if (A_(i,0) == A_(j,0) || A_(i,1) == A_(j,1)) {
      // violates distinctness constraint
      continue;
    }
    const auto l1 = afm1[A_(i,0)][A_(j,0)];
    const auto l2 = afm2[A_(i,1)][A_(j,1)];
    const double c = std::abs(l1 - l2);
    if (c < 0.02) {
      // edges.insert(std::make_pair(c, std::make_pair(i, j)));
      inlier_graphs_[0].addEdge(i, j);
      edge_num++;
    }
  }

  if (edge_num > num_corr * 40) {
    return Association();
  }

  MaxCliqueSolver solver;
  auto clique = solver.findMaxClique(inlier_graphs_[0]);
  Association association = Association::Zero(clique.size(), 2);
  for (size_t i=0; i<clique.size(); ++i) {
    association.row(i) = A_.row(clique[i]);
  } 
  return association;
}

std::vector<int> GlobalRegister::checkSolution(const SceneGraph::Ptr& gref, const SceneGraph::Ptr& gcur, 
                                               const Association& association, const Transform& Trc) {
  std::vector<int> inliers;
  // int size = association.rows();
  // const double inlier_threshold = 0.3;
  // for(int i = 0; i < size; ++i) {
  //   SceneGraphNode::Ptr const& ref_node = gref->nodes_[association(i, 0)];
  //   SceneGraphNode::Ptr const& cur_node = gcur->nodes_[association(i, 1)];
  //   auto const geo_type = ref_node->geometry_type();
  //   if(geo_type == SceneGraphNode::GeoType::LINE) {
  //     noiseModel::Diagonal::shared_ptr noise = noiseModel::Unit::Create(3);
  //     Eigen::Vector3d const &p = cur_node->node_value();
  //     Eigen::Vector3d const &q = ref_node->node_value();
  //     Eigen::Vector3d const &n = ref_node->normal_value();
  //     Eigen::Vector3d residual = (Eigen::Matrix3d::Identity() - n * n.transpose()) * (Trc * p - q);
  //     if(residual.norm() < inlier_threshold) {
  //       inliers.push_back(i);
  //     }
  //   }
  //   else if(geo_type == SceneGraphNode::GeoType::SURFACE) {
  //     noiseModel::Diagonal::shared_ptr noise = noiseModel::Unit::Create(1);
  //     Eigen::Vector3d const &p = cur_node->node_value();
  //     Eigen::Vector3d const &q = ref_node->node_value();
  //     Eigen::Vector3d const &n = ref_node->normal_value();
  //     double residual = std::abs(n.transpose() * (Trc * p - q));
  //     if(residual < inlier_threshold) {
  //       inliers.push_back(i);
  //     }
  //   }
  // }
  return inliers;
}

std::vector<int> MaxCliqueSolver::findMaxClique(const AssociationGraph& graph, const int lower_bound) {

  // Handle deprecated field
  if (!params_.solve_exactly) {
    params_.solver_mode = CLIQUE_SOLVER_MODE::PMC_HEU;
  }

  std::vector<int> edges;
  std::vector<long long> vertices;
  vertices.push_back(edges.size());

  const auto all_vertices = graph.getVertices();
  for (const auto& i : all_vertices) {
    const auto& c_edges = graph.getEdges(i);
    edges.insert(edges.end(), c_edges.begin(), c_edges.end());
    vertices.push_back(edges.size());
  }
  // printf("[MaxCliqueSolver] Vertices: %ld, Edges: %ld\n", vertices.size(), edges.size());

  // Use PMC to calculate
  pmc::pmc_graph G(vertices, edges);

  // Prepare PMC input
  // TODO: Incorporate this to the constructor
  pmc::input in;
  in.algorithm = 1;
  in.threads = 8;
  in.experiment = 0;
  in.lb = lower_bound;
  in.ub = 0;
  in.param_ub = 0;
  in.adj_limit = 20000;
  in.time_limit = params_.time_limit;
  in.remove_time = 4;
  in.graph_stats = false;
  in.verbose = false;
  in.help = false;
  in.MCE = false;
  in.decreasing_order = false;
  in.heu_strat = "kcore";
  in.vertex_search_order = "deg";

  // std::vector to represent max clique
  std::vector<int> C;

  // upper-bound of max clique
  G.compute_cores();
  auto max_core = G.get_max_core();

  // check for k-core heuristic threshold
  // check whether threshold equals 1 to short circuit the comparison
  if (params_.solver_mode == CLIQUE_SOLVER_MODE::KCORE_HEU &&
      params_.kcore_heuristic_threshold != 1 &&
      max_core > static_cast<int>(params_.kcore_heuristic_threshold *
                                  static_cast<double>(all_vertices.size()))) {
    // remove all nodes with core number less than max core number
    // k_cores is a std::vector saving the core number of each vertex
    auto k_cores = G.get_kcores();
    for (int i = 1; i < k_cores->size(); ++i) {
      // Note: k_core has size equals to num vertices + 1
      if ((*k_cores)[i] >= max_core) {
        C.push_back(i-1);
      }
    }
    return C;
  }

  if (in.ub == 0) {
    in.ub = max_core + 1;
  }

  // lower-bound of max clique
  if (in.lb == 0 && in.heu_strat != "0") { // skip if given as input
    pmc::pmc_heu maxclique(G, in);
    in.lb = maxclique.search(G, C);
  }

  assert(in.lb != 0);
  if (in.lb == 0) {
    // This means that max clique has a size of one
    return C;
  }

  if (in.lb == in.ub) {
    return C;
  }

  // Optional exact max clique finding
  if (params_.solver_mode == CLIQUE_SOLVER_MODE::PMC_EXACT) {
    // The following methods are used:
    // 1. k-core pruning
    // 2. neigh-core pruning/ordering
    // 3. dynamic coloring bounds/sort
    // see the original PMC paper and implementation for details:
    // R. A. Rossi, D. F. Gleich, and A. H. Gebremedhin, “Parallel Maximum Clique Algorithms with
    // Applications to Network Analysis,” SIAM J. Sci. Comput., vol. 37, no. 5, pp. C589–C616, Jan.
    // 2015.
    if (G.num_vertices() < in.adj_limit) {
      G.create_adj();
      pmc::pmcx_maxclique finder(G, in);
      finder.search_dense(G, C);
    } else {
      pmc::pmcx_maxclique finder(G, in);
      finder.search(G, C);
    }
  }

  return C;
}
  
} // namespace SLIM


