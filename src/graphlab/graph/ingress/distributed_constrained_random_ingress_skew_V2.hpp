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

#ifndef GRAPHLAB_DISTRIBUTED_CONSTRAINED_RANDOM_INGRESS_SKEW_V2_HPP
#define GRAPHLAB_DISTRIBUTED_CONSTRAINED_RANDOM_INGRESS_SKEW_V2_HPP

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
  class distributed_constrained_random_ingress_skew_v2 : 
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


    sharding_constraint* constraint;    ///Shuang Song, this is where it diffs from random
    boost::hash<vertex_id_type> hashvid;

  public:
    distributed_constrained_random_ingress_skew_v2(distributed_control& dc, graph_type& graph,
                                           const std::string& method) :
    base_type(dc, graph), proc_num_edges(dc.numprocs()) {
      constraint = new sharding_constraint(dc.numprocs(), method);
//      form_new_skew_list(dc.numprocs());
    } // end of constructor

    ~distributed_constrained_random_ingress_skew_v2() { 
      delete constraint;
    }


    struct value{
		procid_t shardi;
		procid_t shardj;
		int local_skew_sum;
	};
    int weight;
    double weight_tmp;
    std::vector<procid_t> candidates_tmp;

    std::vector<value> candidate_skew_list;


    void form_new_skew_list(size_t numshards){
		weight = 0;
		for(int i = 0; i < numshards ; i++){
			for(int j = 0; j < numshards; j++){
				value y;
				y.shardi = i;
				y.shardj = j;
				y.local_skew_sum = 0;
				weight_tmp = 0;
				candidates_tmp = constraint->get_joint_neighbors(i,j);
				for(int tmp = 0; tmp < candidates_tmp.size() ; tmp++){
					procid_t id = candidates_tmp[tmp];
					y.local_skew_sum += base_type::graph.skew_vector[id];
/*					for (int ii = 0; ii < base_type::graph.skew_vector[id]; ++ii){
						candidate_skew_list.push_back(y);
					}
*/				}
				weight += y.local_skew_sum;
				for(int i = 0; i < y.local_skew_sum; i++){
					candidate_skew_list.push_back(y);
				}
/*				for(int tmp = 0; tmp < candidates_tmp.size() ; tmp++){
					procid_t id = candidates_tmp[tmp];
					for (int ii = 0; ii < base_type::graph.skew_vector[id];++ii){
						candidate_skew_list.push_back(y);
					}
				}
*/			}
		}
	}


    /** Add an edge to the ingress object using random assignment. */
    void add_edge(vertex_id_type source, vertex_id_type target,
                  const EdgeData& edata) {
      typedef typename base_type::edge_buffer_record edge_buffer_record;
	///Shuang Song; candidates generated
	///trying to hash candidates as well
      typedef std::pair<vertex_id_type, vertex_id_type> edge_pair_type;
      const edge_pair_type edge_pair(std::min(source, target), std::max(source, target));
//      value ij_pair = candidate_skew_list[graph_hash::hash_edge(edge_pair) % weight];
      
      const std::vector<procid_t>& candidates = constraint->get_joint_neighbors(graph_hash::hash_vertex(source) % base_type::rpc.numprocs(),graph_hash::hash_vertex(target) % base_type::rpc.numprocs());

/*      double edge_balance = 0;
      for(int i = 0; i < base_type::rpc.numprocs(); ++i){
          edge_balance += proc_num_edges[i] ;
      }

	  const std::vector<procid_t>& candidates_2 = constraint->get_joint_neighbors(ij_pair.shardi, ij_pair.shardj) ;
	
	  double temp_balance = 0;
	  for(int i = 0; i < candidates_2.size() ; i++){
		temp_balance += proc_num_edges[candidates_2[i]];		
	  }
			
	procid_t owning_proc = 0;

       if( (temp_balance / edge_balance) > ( 1.15 * ij_pair.local_skew_sum / base_type::graph.skew_sum)){
		owning_proc = base_type::edge_decision.edge_to_proc_random_skew(source, target, base_type::graph.skew_vector, candidates_1, proc_num_edges,base_type::graph.skew_list, base_type::graph.skew_sum, edge_balance, ij_pair.local_skew_sum);
	}
	else{
	 	owning_proc = base_type::edge_decision.edge_to_proc_random_skew(source, target, base_type::graph.skew_vector, candidates_2, proc_num_edges, base_type::graph.skew_list,base_type::graph.skew_sum, edge_balance, ij_pair.local_skew_sum);	
	}
*/
      const procid_t owning_proc = base_type::edge_decision.edge_to_proc_random_skew(source, target, candidates, base_type::graph.skew_vector);

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
