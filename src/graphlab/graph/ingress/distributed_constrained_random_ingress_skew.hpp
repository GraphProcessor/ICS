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

#ifndef GRAPHLAB_DISTRIBUTED_CONSTRAINED_RANDOM_INGRESS_SKEW_HPP
#define GRAPHLAB_DISTRIBUTED_CONSTRAINED_RANDOM_INGRESS_SKEW_HPP

#include <boost/functional/hash.hpp>

#include <graphlab/rpc/buffered_exchange.hpp>
#include <graphlab/graph/graph_basic_types.hpp>
#include <graphlab/graph/ingress/distributed_ingress_base.hpp>
#include <graphlab/graph/distributed_graph.hpp>
#include <graphlab/graph/ingress/sharding_constraint.hpp>
#include <graphlab/graph/ingress/ingress_edge_decision.hpp>
#include <map>

#include <graphlab/macros_def.hpp>
namespace graphlab {
  template<typename VertexData, typename EdgeData>
  class distributed_graph;

  /**
   * \brief Ingress object assigning edges using randoming hash function.
   */
  template<typename VertexData, typename EdgeData>
  class distributed_constrained_random_ingress_skew : 
    public distributed_ingress_base<VertexData, EdgeData> {
  public:
    typedef distributed_graph<VertexData, EdgeData> graph_type;
    /// The type of the vertex data stored in the graph 
    typedef VertexData vertex_data_type;
    /// The type of the edge data stored in the graph 
    typedef EdgeData edge_data_type;


    typedef distributed_ingress_base<VertexData, EdgeData> base_type;

    /* New Edge Count */
    std::vector<size_t> proc_num_edges; 
    int edge_balance;

    sharding_constraint* constraint;    ///Shuang Song, this is where it diffs from random
    boost::hash<vertex_id_type> hashvid;

  public:
    distributed_constrained_random_ingress_skew(distributed_control& dc, graph_type& graph,
                                           const std::string& method) :
    base_type(dc, graph), proc_num_edges(dc.numprocs()) {
      constraint = new sharding_constraint(dc.numprocs(), method);
      form_new_skew_list(dc.numprocs());
    } // end of constructor

    ~distributed_constrained_random_ingress_skew() { 
      delete constraint;
    }

    std::vector<int> skewed_shard_list;
    void form_new_skew_list(size_t numshards){
        for(int i = 0; i < numshards; i++) {
	    std::vector<procid_t>& shard = constraint->constraint_graph[i];
            int shard_weight = 0;
            for(procid_t j = 0; j < shard.size(); j++) {
		 shard_weight += base_type::graph.skew_vector[j];
            }
            shard_weight = shard_weight / shard.size();
            for(procid_t j = 0; j < shard_weight; j++) {
		 skewed_shard_list.push_back(i);
            }
         }
    }

    /** Add an edge to the ingress object using random assignment. */
    void add_edge(vertex_id_type source, vertex_id_type target,
                  const EdgeData& edata) {
      typedef typename base_type::edge_buffer_record edge_buffer_record;
      typedef std::pair<vertex_id_type, vertex_id_type> edge_pair_type;
      const edge_pair_type edge_pair(std::min(source, target), std::max(source, target));

      int shard_i = skewed_shard_list[graph_hash::hash_vertex(source) % skewed_shard_list.size()];
      int shard_j = skewed_shard_list[graph_hash::hash_vertex(target) % skewed_shard_list.size()];


      const std::vector<procid_t>& candidates_skewed = constraint->get_joint_neighbors(shard_i,shard_j);
			
      int edge_balance =0;
      for(int i = 0; i < proc_num_edges.size() ; i++){
          edge_balance += proc_num_edges[i];
      }

      procid_t owning_proc = base_type::edge_decision.edge_to_proc_random_skew(source, target, base_type::graph.skew_vector, candidates_skewed, proc_num_edges,base_type::graph.skew_list, base_type::graph.skew_sum, edge_balance, 0);

      const edge_buffer_record record(source, target, edata);

#ifdef _OPENMP
      base_type::edge_exchange.send(owning_proc, record, omp_get_thread_num());
#else
      base_type::edge_exchange.send(owning_proc, record);
#endif
    } // end of add edge
  }; // end of distributed_constrained_random_ingress
}; // end of namespace graphlab
#include <graphlab/macros_undef.hpp>


#endif
