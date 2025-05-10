/**
 ============================================================================
 Name        : Parallel Maximum Clique (PMC) Library
 Author      : Ryan A. Rossi   (rrossi@purdue.edu)
 Description : A general high-performance parallel framework for computing
               maximum cliques. The library is designed to be fast for large
               sparse graphs.

 Copyright (C) 2012-2013, Ryan A. Rossi, All rights reserved.

 Please cite the following paper if used:
   Ryan A. Rossi, David F. Gleich, Assefaw H. Gebremedhin, Md. Mostofa
   Patwary, A Fast Parallel Maximum Clique Algorithm for Large Sparse Graphs
   and Temporal Strong Components, arXiv preprint 1302.6256, 2013.

 See http://ryanrossi.com/pmc for more information.
 ============================================================================
 */

#include "pmc.h"

using namespace std;
using namespace pmc;

extern "C" {
    
// a list of edges, where index_offset is the starting index
int max_clique(long long nedges, int *ei, int *ej, int index_offset, 
                int outsize, int *clique) {
    input in;
    
    pmc_graph G(nedges, ei, ej, index_offset); 
    
    //! ensure wait time is greater than the time to recompute the graph data structures
    if (G.num_edges() > 1000000000 && in.remove_time < 120)  in.remove_time = 120;
    else if (G.num_edges() > 250000000 && in.remove_time < 10) in.remove_time = 10;
    cout << "explicit reduce is set to " << in.remove_time << " seconds" <<endl;
    
    double seconds = get_time();
    G.compute_cores();
    if (in.ub == 0) {
        in.ub = G.get_max_core() + 1;
        cout << "K: " << in.ub <<endl;
        cout << "k-cores time: " << get_time() - seconds << ", ub: " << in.ub << endl;
    }
    
    //! lower-bound of max clique
    vector<int> C;
    if (in.lb == 0 && in.heu_strat != "0") { // skip if given as input
        pmc_heu maxclique(G,in);
        in.lb = maxclique.search(G, C);
        cout << "Heuristic found clique of size " << in.lb;
        cout << " in " << get_time() - seconds << " seconds" <<endl;
        cout << "[pmc: heuristic]  ";
        print_max_clique(C);
    }

    //! check solution found by heuristic
    if (in.lb == in.ub && !in.MCE) {
        cout << "Heuristic found optimal solution." << endl;
    }
    else if (in.algorithm >= 0) {
        switch(in.algorithm) {
            case 0: {
                //! k-core pruning, neigh-core pruning/ordering, dynamic coloring bounds/sort
                if (G.num_vertices() < in.adj_limit) {
                    G.create_adj();
                    pmcx_maxclique finder(G,in);
                    finder.search_dense(G,C);
                    break;
                }
                else {
                    pmcx_maxclique finder(G,in);
                    finder.search(G,C);
                    break;
                }
            }
            case 1: {
                //! k-core pruning, dynamic coloring bounds/sort
                if (G.num_vertices() < in.adj_limit) {
                    G.create_adj();
                    pmcx_maxclique_basic finder(G,in);
                    finder.search_dense(G,C);
                    break;
                }
                else {
                    pmcx_maxclique_basic finder(G,in);
                    finder.search(G,C);
                    break;
                }
            }
            case 2: {
                //! simple k-core pruning (four new pruning steps)
                pmc_maxclique finder(G,in);
                finder.search(G,C);
                break;
            }
            default:
                cout << "algorithm " << in.algorithm << " not found." <<endl;
                break;
        }
        seconds = (get_time() - seconds);
        cout << "Time taken: " << seconds << " SEC" << endl;
        cout << "Size (omega): " << C.size() << endl;
        print_max_clique(C);
    }
    
    // save the output
    for(int i = 0; i < C.size() && i < outsize; i++)
        clique[i] = C[i] + index_offset;

    return C.size();
    return 0;
}

};
