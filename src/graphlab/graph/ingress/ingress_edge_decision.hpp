/**  
 * Copyright (c) 2009 Carnegie Mellon University. 
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://www.graphlab.ml.cmu.edu
 *
 */
#ifndef GRAPHLAB_DISTRIBUTED_INGRESS_EDGE_DECISION_HPP
#define GRAPHLAB_DISTRIBUTED_INGRESS_EDGE_DECISION_HPP

#include <graphlab/graph/distributed_graph.hpp>
#include <graphlab/graph/graph_basic_types.hpp>
#include <graphlab/graph/graph_hash.hpp>
#include <graphlab/rpc/distributed_event_log.hpp>
#include <graphlab/util/dense_bitset.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <math.h>

namespace graphlab {
  template<typename VertexData, typename EdgeData>
  class distributed_graph;
 
 template<typename VertexData, typename EdgeData>
 class ingress_edge_decision {

    public:
      typedef graphlab::vertex_id_type vertex_id_type;
      typedef distributed_graph<VertexData, EdgeData> graph_type;
      typedef fixed_dense_bitset<RPC_MAX_N_PROCS> bin_counts_type;     

    public:
      /** \brief A decision object for computing the edge assingment. */
      ingress_edge_decision(distributed_control& dc) { }

      /** Random assign (source, target) to a machine p in {0, ... numprocs-1} */
      procid_t edge_to_proc_random (const vertex_id_type source, 
          const vertex_id_type target,
          size_t numprocs) {
        typedef std::pair<vertex_id_type, vertex_id_type> edge_pair_type;
        const edge_pair_type edge_pair(std::min(source, target), 
            std::max(source, target));
        return graph_hash::hash_edge(edge_pair) % (numprocs);
      };

      /** Random assign (source, target) to a machine p in a list of candidates */
      procid_t edge_to_proc_random (const vertex_id_type source, 
          const vertex_id_type target,
          const std::vector<procid_t> & candidates) {
        typedef std::pair<vertex_id_type, vertex_id_type> edge_pair_type;
        const edge_pair_type edge_pair(std::min(source, target), 
            std::max(source, target));

        return candidates[graph_hash::hash_edge(edge_pair) % (candidates.size())];
      };


      /** Random assign (source, target) to a machine p in a list of candidates */
      procid_t edge_to_proc_random_skew (const vertex_id_type source, 
          const vertex_id_type target,
          const std::vector<procid_t> & candidates,
	  std::vector<int> skew_vector) {
        typedef std::pair<vertex_id_type, vertex_id_type> edge_pair_type;
        const edge_pair_type edge_pair(std::min(source, target), 
            std::max(source, target));

	std::vector<int> new_skew_list;
	for (int it = 0; it < candidates.size(); ++it){
		int weight = skew_vector[candidates[it]];
		for(int it2 = 0; it2 < weight; ++it2) 
			new_skew_list.push_back(candidates[it]);
	}
		
	return new_skew_list[graph_hash::hash_edge(edge_pair) % new_skew_list.size()];



	
        //return candidates[graph_hash::hash_edge(edge_pair) % (candidates.size())];
      };



      /**Shuang:Random assign (source, target) to a machine p in a list of candidates*/
      procid_t edge_to_proc_random_skew (const vertex_id_type source,
          const vertex_id_type target,
          std::vector<int> skew_vector,
  	  const std::vector<procid_t> & candidates,
	  std::vector<size_t>& proc_num_edges,
	  std::vector<int>& skew_list,
          double skew_sum, double total_edges, int local_skew_sum){
		const std::pair<vertex_id_type, vertex_id_type> edge_pair(std::min(source,target), std::max(source,target));
		std::vector<procid_t>  new_candidates;
		procid_t best_proc = candidates[0];
		double best_score = -2;
		for (int i =0; i < candidates.size(); ++i){
			procid_t cand = candidates[i];
			int cand_edges = proc_num_edges[cand];
			double edge_balance = cand_edges / total_edges;
			double target_balance = 1.00 * skew_vector[cand] / skew_sum;
			double score = target_balance - edge_balance;
 	 	//	logstream(LOG_EMPH) <<"Balance: " <<  edge_balance << " || Target: " << target_balance << std::endl;
			if(score > 0){
				new_candidates.push_back(cand);
			}
			if(score > best_score) {
				best_score = score;
				best_proc = cand;
			}
			
		}
		
 	 	//logstream(LOG_EMPH) <<"Cand : " <<  candidates.size() << " || New Cand: " << new_candidates.size() << std::endl;
		
	//	if(new_candidates.size() > 0){
//			best_proc = new_candidates[graph_hash::hash_edge(edge_pair) % new_candidates.size()];
//		} else {
//			best_proc = best_proc;
//		}
		proc_num_edges[best_proc]++;
		return best_proc;

	}
 
      /*Shuang:Random assign (source, target) to a machine p in (0, ..... numproc - 1)
 	* accounting for machine performance differentials*/
      procid_t edge_to_proc_random_skew (const vertex_id_type source,
       	 const vertex_id_type target,
	 size_t numprocs,
 	 std::vector<int> skew_list){
		typedef std::pair<vertex_id_type, vertex_id_type> edge_pair_type;
        	const edge_pair_type edge_pair(std::min(source, target), std::max(source, target));
		return skew_list[graph_hash::hash_edge(edge_pair) % skew_list.size()];
	}

      /** Shuang: Greedy assign (source, target) to a machine using: 
       *  bitset<MAX_MACHINE> src_degree : the degree presence of source over machines
       *  bitset<MAX_MACHINE> dst_degree : the degree presence of target over machines
       *  vector<size_t>      proc_num_edges : the edge counts over machines
       * */
      procid_t edge_to_proc_greedy_skew (const vertex_id_type source, 
          const vertex_id_type target,
          bin_counts_type& src_degree,
          bin_counts_type& dst_degree,
          std::vector<size_t>& proc_num_edges,
          std::vector<int> skew_vector,
  	  bool usehash = false,
          bool userecent = false) {
        size_t numprocs = proc_num_edges.size();

        // Compute the score of each proc.
        procid_t best_proc = -1; 
        double maxscore = 0.0;
        double epsilon = 1.0; 
        std::vector<double> proc_score(numprocs); 
        size_t minedges = *std::min_element(proc_num_edges.begin(), proc_num_edges.end());
        size_t maxedges = *std::max_element(proc_num_edges.begin(), proc_num_edges.end());

	double skew_sum = 0;
	double edge_balance = 0;

	for (size_t i = 0; i < numprocs; ++i){
		skew_sum += skew_vector[i];
		edge_balance += proc_num_edges[i];
	}
	
	std::vector<double> balance(numprocs);

        for (size_t i = 0; i < numprocs; ++i) {
          size_t sd = src_degree.get(i) + (usehash && (source % numprocs == i));
          size_t td = dst_degree.get(i) + (usehash && (target % numprocs == i));
          double bal = (maxedges - proc_num_edges[i])/(epsilon + maxedges - minedges);
	  double weight = skew_vector[i] / skew_sum;
 	 // logstream(LOG_EMPH) << proc_num_edges[i] << " " << edge_balance << " " << weight << " " << abs(proc_num_edges[i] / edge_balance - weight) << std::endl;
	  if (proc_num_edges[i] / edge_balance <= (1.15 * weight)) 
	  	proc_score[i] = (weight * bal) + (sd > 0) + (td > 0);
	  else
		proc_score[i] = 0;
// 	  logstream(LOG_EMPH) << "score: " << proc_score[i] << std::endl;

        }
        maxscore = *std::max_element(proc_score.begin(), proc_score.end());

	
        std::vector<procid_t> top_procs; 
        for (size_t i = 0; i < numprocs; ++i){
          if (std::fabs(proc_score[i] - maxscore) < 1e-5){
             	top_procs.push_back(i);
	  }
	}


        // Hash the edge to one of the best procs.
        typedef std::pair<vertex_id_type, vertex_id_type> edge_pair_type;
        const edge_pair_type edge_pair(std::min(source, target), std::max(source, target));

/*	std::vector<int> new_skew_list;
	for (int it = 0; it < top_procs.size(); ++it){
		int weight = skew_vector[top_procs[it]];
		for(int it2 = 0; it2 < weight; ++it2) 
			new_skew_list.push_back(top_procs[it]);
	}
		
	best_proc = new_skew_list[graph_hash::hash_edge(edge_pair) % new_skew_list.size()];

*/
	best_proc = top_procs[graph_hash::hash_edge(edge_pair) % top_procs.size()];	
	
        ASSERT_LT(best_proc, numprocs);
        if (userecent) {
          src_degree.clear();
          dst_degree.clear();
        }
        src_degree.set_bit(best_proc);
        dst_degree.set_bit(best_proc);
        ++proc_num_edges[best_proc];
        return best_proc;
      };





      /** Greedy assign (source, target) to a machine using: 
       *  bitset<MAX_MACHINE> src_degree : the degree presence of source over machines
       *  bitset<MAX_MACHINE> dst_degree : the degree presence of target over machines
       *  vector<size_t>      proc_num_edges : the edge counts over machines
       * */
      procid_t edge_to_proc_greedy (const vertex_id_type source, 
          const vertex_id_type target,
          bin_counts_type& src_degree,
          bin_counts_type& dst_degree,
          std::vector<size_t>& proc_num_edges,
          bool usehash = false,
          bool userecent = false) {
        size_t numprocs = proc_num_edges.size();


//	usehash= true;
//	userecent=true;
        // Compute the score of each proc.
        procid_t best_proc = -1; 
        double maxscore = 0.0;
        double epsilon = 1.0; 
        std::vector<double> proc_score(numprocs); 
        size_t minedges = *std::min_element(proc_num_edges.begin(), proc_num_edges.end());
        size_t maxedges = *std::max_element(proc_num_edges.begin(), proc_num_edges.end());

        for (size_t i = 0; i < numprocs; ++i) {
          size_t sd = src_degree.get(i) + (usehash && (source % numprocs == i));
          size_t td = dst_degree.get(i) + (usehash && (target % numprocs == i));
          double bal = (maxedges - proc_num_edges[i])/(epsilon + maxedges - minedges);
          ////i think bal may always be 0
	  proc_score[i] = bal + ((sd > 0) + (td > 0));
        }
        maxscore = *std::max_element(proc_score.begin(), proc_score.end());

//	logstream(LOG_EMPH) << "shuang song, greedy algorithm enabled by batch" << std::endl;

        std::vector<procid_t> top_procs; 
        for (size_t i = 0; i < numprocs; ++i)
          if (std::fabs(proc_score[i] - maxscore) < 1e-5)
            top_procs.push_back(i);

        // Hash the edge to one of the best procs.
        typedef std::pair<vertex_id_type, vertex_id_type> edge_pair_type;
        const edge_pair_type edge_pair(std::min(source, target), 
            std::max(source, target));
        best_proc = top_procs[graph_hash::hash_edge(edge_pair) % top_procs.size()];

        ASSERT_LT(best_proc, numprocs);
        if (userecent) {
          src_degree.clear();
          dst_degree.clear();
        }
        src_degree.set_bit(best_proc);
        dst_degree.set_bit(best_proc);
        ++proc_num_edges[best_proc];
        return best_proc;
      };

      /** Greedy assign (source, target) to a machine using: 
       *  bitset<MAX_MACHINE> src_degree : the degree presence of source over machines
       *  bitset<MAX_MACHINE> dst_degree : the degree presence of target over machines
       *  vector<size_t>      proc_num_edges : the edge counts over machines
       * */
      procid_t edge_to_proc_greedy (const vertex_id_type source, 
          const vertex_id_type target,
          bin_counts_type& src_degree,
          bin_counts_type& dst_degree,
          std::vector<procid_t>& candidates,
          std::vector<size_t>& proc_num_edges,
          bool usehash = false,
          bool userecent = false
          ) {
        size_t numprocs = proc_num_edges.size();

        // Compute the score of each proc.
        procid_t best_proc = -1; 
        double maxscore = 0.0;
        double epsilon = 1.0; 
        std::vector<double> proc_score(candidates.size()); 
        size_t minedges = *std::min_element(proc_num_edges.begin(), proc_num_edges.end());
        size_t maxedges = *std::max_element(proc_num_edges.begin(), proc_num_edges.end());

        for (size_t j = 0; j < candidates.size(); ++j) {
          size_t i = candidates[j];
          size_t sd = src_degree.get(i) + (usehash && (source % numprocs == i));
          size_t td = dst_degree.get(i) + (usehash && (target % numprocs == i));
          double bal = (maxedges - proc_num_edges[i])/(epsilon + maxedges - minedges);
          proc_score[j] = bal + ((sd > 0) + (td > 0));
        }
        maxscore = *std::max_element(proc_score.begin(), proc_score.end());

        std::vector<procid_t> top_procs; 
        for (size_t j = 0; j < candidates.size(); ++j)
          if (std::fabs(proc_score[j] - maxscore) < 1e-5)
            top_procs.push_back(candidates[j]);

        // Hash the edge to one of the best procs.
        typedef std::pair<vertex_id_type, vertex_id_type> edge_pair_type;
        const edge_pair_type edge_pair(std::min(source, target), std::max(source, target));
        best_proc = top_procs[graph_hash::hash_edge(edge_pair) % top_procs.size()];

        ASSERT_LT(best_proc, numprocs);
        if (userecent) {
          src_degree.clear();
          dst_degree.clear();
        }
        src_degree.set_bit(best_proc);
        dst_degree.set_bit(best_proc);
        ++proc_num_edges[best_proc];
        return best_proc;
      };

  };// end of ingress_edge_decision
}

#endif
