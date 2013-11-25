/**
 * Copyright (c) 2013 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University.
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
 *      http://ipads.se.sjtu.edu.cn/projects/powerlyra.html
 *
 */


#ifndef GRAPHLAB_POWERLYRA_SYNC_ENGINE_HPP
#define GRAPHLAB_POWERLYRA_SYNC_ENGINE_HPP

#include <deque>
#include <boost/bind.hpp>

#include <graphlab/engine/iengine.hpp>

#include <graphlab/vertex_program/ivertex_program.hpp>
#include <graphlab/vertex_program/icontext.hpp>
#include <graphlab/vertex_program/context.hpp>

#include <graphlab/engine/execution_status.hpp>
#include <graphlab/options/graphlab_options.hpp>




#include <graphlab/parallel/pthread_tools.hpp>
#include <graphlab/parallel/fiber_barrier.hpp>
#include <graphlab/util/tracepoint.hpp>
#include <graphlab/util/memory_info.hpp>
#include <graphlab/util/triple.hpp>
#include <graphlab/util/tetrad.hpp>

#include <graphlab/rpc/dc_dist_object.hpp>
#include <graphlab/rpc/distributed_event_log.hpp>
#include <graphlab/rpc/buffered_exchange.hpp>
#include <graphlab/rpc/fiber_buffered_exchange.hpp>






#include <graphlab/macros_def.hpp>

#define ALIGN_DOWN(_n, _w) ((_n) & (~((_w)-1)))

#define TUNING
namespace graphlab {


  /**
   * \ingroup engines
   *
   * \brief The synchronous engine executes all active vertex program
   * synchronously in a sequence of super-step (iterations) in both the
   * shared and distributed memory settings.
   *
   * \tparam VertexProgram The user defined vertex program which
   * should implement the \ref graphlab::ivertex_program interface.
   *
   *
   * ### Execution Semantics
   *
   * On start() the \ref graphlab::ivertex_program::init function is invoked
   * on all vertex programs in parallel to initialize the vertex program,
   * vertex data, and possibly signal vertices.
   * The engine then proceeds to execute a sequence of
   * super-steps (iterations) each of which is further decomposed into a
   * sequence of minor-steps which are also executed synchronously:
   * \li Receive all incoming messages (signals) by invoking the
   * \ref graphlab::ivertex_program::init function on all
   * vertex-programs that have incoming messages.  If a
   * vertex-program does not have any incoming messages then it is
   * not active during this super-step.
   * \li Execute all gathers for active vertex programs by invoking
   * the user defined \ref graphlab::ivertex_program::gather function
   * on the edge direction returned by the
   * \ref graphlab::ivertex_program::gather_edges function.  The gather
   * functions can modify edge data but cannot modify the vertex
   * program or vertex data and therefore can be executed on multiple
   * edges in parallel.  The gather type is used to accumulate (sum)
   * the result of the gather function calls.
   * \li Execute all apply functions for active vertex-programs by
   * invoking the user defined \ref graphlab::ivertex_program::apply
   * function passing the sum of the gather functions.  If \ref
   * graphlab::ivertex_program::gather_edges returns no edges then
   * the default gather value is passed to apply.  The apply function
   * can modify the vertex program and vertex data.
   * \li Execute all scatters for active vertex programs by invoking
   * the user defined \ref graphlab::ivertex_program::scatter function
   * on the edge direction returned by the
   * \ref graphlab::ivertex_program::scatter_edges function.  The scatter
   * functions can modify edge data but cannot modify the vertex
   * program or vertex data and therefore can be executed on multiple
   * edges in parallel.
   *
   * ### Construction
   *
   * The synchronous engine is constructed by passing in a
   * \ref graphlab::distributed_control object which manages coordination
   * between engine threads and a \ref graphlab::distributed_graph object
   * which is the graph on which the engine should be run.  The graph should
   * already be populated and cannot change after the engine is constructed.
   * In the distributed setting all program instances (running on each machine)
   * should construct an instance of the engine at the same time.
   *
   * Computation is initiated by signaling vertices using either
   * \ref graphlab::powerlyra_sync_engine::signal or
   * \ref graphlab::powerlyra_sync_engine::signal_all.  In either case all
   * machines should invoke signal or signal all at the same time.  Finally,
   * computation is initiated by calling the
   * \ref graphlab::powerlyra_sync_engine::start function.
   *
   * ### Example Usage
   *
   * The following is a simple example demonstrating how to use the engine:
   * \code
   * #include <graphlab.hpp>
   *
   * struct vertex_data {
   *   // code
   * };
   * struct edge_data {
   *   // code
   * };
   * typedef graphlab::distributed_graph<vertex_data, edge_data> graph_type;
   * typedef float gather_type;
   * struct pagerank_vprog :
   *   public graphlab::ivertex_program<graph_type, gather_type> {
   *   // code
   * };
   *
   * int main(int argc, char** argv) {
   *   // Initialize control plain using mpi
   *   graphlab::mpi_tools::init(argc, argv);
   *   graphlab::distributed_control dc;
   *   // Parse command line options
   *   graphlab::command_line_options clopts("PageRank algorithm.");
   *   std::string graph_dir;
   *   clopts.attach_option("graph", &graph_dir, graph_dir,
   *                        "The graph file.");
   *   if(!clopts.parse(argc, argv)) {
   *     std::cout << "Error in parsing arguments." << std::endl;
   *     return EXIT_FAILURE;
   *   }
   *   graph_type graph(dc, clopts);
   *   graph.load_structure(graph_dir, "tsv");
   *   graph.finalize();
   *   std::cout << "#vertices: " << graph.num_vertices()
   *             << " #edges:" << graph.num_edges() << std::endl;
   *   graphlab::powerlyra_sync_engine<pagerank_vprog> engine(dc, graph, clopts);
   *   engine.signal_all();
   *   engine.start();
   *   std::cout << "Runtime: " << engine.elapsed_time();
   *   graphlab::mpi_tools::finalize();
   * }
   * \endcode
   *
   *
   *
   * <a name=engineopts>Engine Options</a>
   * =====================
   * The synchronous engine supports several engine options which can
   * be set as command line arguments using \c --engine_opts :
   *
   * \li <b>max_iterations</b>: (default: infinity) The maximum number
   * of iterations (super-steps) to run.
   *
   * \li <b>timeout</b>: (default: infinity) The maximum time in
   * seconds that the engine may run. When the time runs out the
   * current iteration is completed and then the engine terminates.
   *
   * \li <b>use_cache</b>: (default: false) This is used to enable
   * caching.  When caching is enabled the gather phase is skipped for
   * vertices that already have a cached value.  To use caching the
   * vertex program must either clear (\ref icontext::clear_gather_cache)
   * or update (\ref icontext::post_delta) the cache values of
   * neighboring vertices during the scatter phase.
   *
   * \li \b snapshot_interval If set to a positive value, a snapshot
   * is taken every this number of iterations. If set to 0, a snapshot
   * is taken before the first iteration. If set to a negative value,
   * no snapshots are taken. Defaults to -1. A snapshot is a binary
   * dump of the graph.
   *
   * \li \b snapshot_path If snapshot_interval is set to a value >=0,
   * this option must be specified and should contain a target basename
   * for the snapshot. The path including folder and file prefix in
   * which the snapshots should be saved.
   *
   * \see graphlab::omni_engine
   * \see graphlab::async_consistent_engine
   * \see graphlab::semi_synchronous_engine
   * \see graphlab::power_sync_engine
   * \see graphlab::lyra_sync_engine
   * \see graphlab::powerlyra_sync_engine
   */
  template<typename VertexProgram>
  class powerlyra_sync_engine :
    public iengine<VertexProgram> {

  public:
    /**
     * \brief The user defined vertex program type. Equivalent to the
     * VertexProgram template argument.
     *
     * The user defined vertex program type which should implement the
     * \ref graphlab::ivertex_program interface.
     */
    typedef VertexProgram vertex_program_type;

    /**
     * \brief The user defined type returned by the gather function.
     *
     * The gather type is defined in the \ref graphlab::ivertex_program
     * interface and is the value returned by the
     * \ref graphlab::ivertex_program::gather function.  The
     * gather type must have an <code>operator+=(const gather_type&
     * other)</code> function and must be \ref sec_serializable.
     */
    typedef typename VertexProgram::gather_type gather_type;


    /**
     * \brief The user defined message type used to signal neighboring
     * vertex programs.
     *
     * The message type is defined in the \ref graphlab::ivertex_program
     * interface and used in the call to \ref graphlab::icontext::signal.
     * The message type must have an
     * <code>operator+=(const gather_type& other)</code> function and
     * must be \ref sec_serializable.
     */
    typedef typename VertexProgram::message_type message_type;

    /**
     * \brief The type of data associated with each vertex in the graph
     *
     * The vertex data type must be \ref sec_serializable.
     */
    typedef typename VertexProgram::vertex_data_type vertex_data_type;

    /**
     * \brief The type of data associated with each edge in the graph
     *
     * The edge data type must be \ref sec_serializable.
     */
    typedef typename VertexProgram::edge_data_type edge_data_type;

    /**
     * \brief The type of graph supported by this vertex program
     *
     * See graphlab::distributed_graph
     */
    typedef typename VertexProgram::graph_type  graph_type;

    /**
     * \brief The type used to represent a vertex in the graph.
     * See \ref graphlab::distributed_graph::vertex_type for details
     *
     * The vertex type contains the function
     * \ref graphlab::distributed_graph::vertex_type::data which
     * returns a reference to the vertex data as well as other functions
     * like \ref graphlab::distributed_graph::vertex_type::num_in_edges
     * which returns the number of in edges.
     *
     */
    typedef typename graph_type::vertex_type          vertex_type;

    /**
     * \brief The type used to represent an edge in the graph.
     * See \ref graphlab::distributed_graph::edge_type for details.
     *
     * The edge type contains the function
     * \ref graphlab::distributed_graph::edge_type::data which returns a
     * reference to the edge data.  In addition the edge type contains
     * the function \ref graphlab::distributed_graph::edge_type::source and
     * \ref graphlab::distributed_graph::edge_type::target.
     *
     */
    typedef typename graph_type::edge_type            edge_type;

    /**
     * \brief The type of the callback interface passed by the engine to vertex
     * programs.  See \ref graphlab::icontext for details.
     *
     * The context callback is passed to the vertex program functions and is
     * used to signal other vertices, get the current iteration, and access
     * information about the engine.
     */
    typedef icontext<graph_type, gather_type, message_type> icontext_type;

  private:

    /**
     * \brief Local vertex type used by the engine for fast indexing
     */
    typedef typename graph_type::local_vertex_type    local_vertex_type;

    /**
     * \brief Local edge type used by the engine for fast indexing
     */
    typedef typename graph_type::local_edge_type      local_edge_type;

    /**
     * \brief Local vertex id type used by the engine for fast indexing
     */
    typedef typename graph_type::lvid_type            lvid_type;

    std::vector<double> per_thread_compute_time;
    /**
     * \brief The actual instance of the context type used by this engine.
     */
    typedef context<powerlyra_sync_engine> context_type;
    friend class context<powerlyra_sync_engine>;


    /**
     * \brief The type of the distributed aggregator inherited from iengine
     */
    typedef typename iengine<vertex_program_type>::aggregator_type aggregator_type;

    /**
     * \brief The object used to communicate with remote copies of the
     * synchronous engine.
     */
    dc_dist_object< powerlyra_sync_engine<VertexProgram> > rmi;

    /**
     * \brief A reference to the distributed graph on which this
     * synchronous engine is running.
     */
    graph_type& graph;

    /**
     * \brief The number of CPUs used.
     */
    size_t ncpus;

    /**
     * \brief The local worker threads used by this engine
     */
    fiber_group threads;

    /**
     * \brief A thread barrier that is used to control the threads in the
     * thread pool.
     */
    fiber_barrier thread_barrier;

    /**
     * \brief The maximum number of super-steps (iterations) to run
     * before terminating.  If the max iterations is reached the
     * engine will terminate if their are no messages remaining.
     */
    size_t max_iterations;


   /* 
    * \brief When caching is enabled the gather phase is skipped for
    * vertices that already have a cached value.  To use caching the
    * vertex program must either clear (\ref icontext::clear_gather_cache)
    * or update (\ref icontext::post_delta) the cache values of
    * neighboring vertices during the scatter phase.
    */
    bool use_cache;

    /**
     * \brief A snapshot is taken every this number of iterations.
     * If snapshot_interval == 0, a snapshot is only taken before the first
     * iteration. If snapshot_interval < 0, no snapshots are taken.
     */
    int snapshot_interval;

    /// \brief The target base name the snapshot is saved in.
    std::string snapshot_path;

    /**
     * \brief A counter that tracks the current iteration number since
     * start was last invoked.
     */
    size_t iteration_counter;

    /**
     * \brief The time in seconds at which the engine started.
     */
    float start_time;

    /**
     * \brief The total execution time.
     */
    double exec_time;

    /**
     * \brief The time spends on exch-msgs phase.
     */
    double exch_time;

    /**
     * \brief The time spends on recv-msgs phase.
     */
    double recv_time;

    /**
     * \brief The time spends on gather phase.
     */
    double gather_time;

    /**
     * \brief The time spends on apply phase.
     */
    double apply_time;

    /**
     * \brief The time spends on scatter phase.
     */
    double scatter_time;

    /**
     * \brief The interval time to print status.
     */
    float print_interval;

    /**
     * \brief The timeout time in seconds
     */
    float timeout;

    /**
     * \brief Schedules all vertices every iteration
     */
    bool sched_allv;

    /**
     * \brief Used to stop the engine prematurely
     */
    bool force_abort;

    /**
     * \brief The vertex locks protect access to vertex specific
     * data-structures including
     * \ref graphlab::powerlyra_sync_engine::gather_accum
     * and \ref graphlab::powerlyra_sync_engine::messages.
     */
    std::vector<simple_spinlock> vlocks;

    /**
     * \brief The egde dirs associated with each vertex on this
     * machine.
     */
    std::vector<edge_dir_type> edge_dirs;

    /**
     * \brief The vertex programs associated with each vertex on this
     * machine.
     */
    std::vector<vertex_program_type> vertex_programs;

    /**
     * \brief Vector of messages associated with each vertex.
     */
    std::vector<message_type> messages;

    /**
     * \brief Bit indicating whether a message is present for each vertex.
     */
    dense_bitset has_message;


    /**
     * \brief Gather accumulator used for each master vertex to merge
     * the result of all the machine specific accumulators (or
     * caches).
     *
     * The gather accumulator can be accessed by multiple threads at
     * once and therefore must be guarded by a vertex locks in
     * \ref graphlab::powerlyra_sync_engine::vlocks
     */
    std::vector<gather_type>  gather_accum;

    /**
     * \brief Bit indicating if the gather has accumulator contains any
     * values.
     *
     * While dense bitsets are thread safe the value of this bit must
     * change concurrently with the
     * \ref graphlab::powerlyra_sync_engine::gather_accum and therefore is
     * set while holding the lock in
     * \ref graphlab::powerlyra_sync_engine::vlocks.
     */
    dense_bitset has_gather_accum;


    /**
     * \brief This optional vector contains caches of previous gather
     * contributions for each machine.
     *
     * Caching is done locally and therefore a high-degree vertex may
     * have multiple caches (one per machine).
     */
    std::vector<gather_type>  gather_cache;

    /**
     * \brief A bit indicating if the local gather for that vertex is
     * available.
     */
    dense_bitset has_cache;

    /**
     * \brief A bit (for master vertices) indicating if that vertex is active
     * (received a message on this iteration).
     */
    dense_bitset active_superstep;

    /**
     * \brief  The number of local vertices (masters) that are active on this
     * iteration.
     */
    atomic<size_t> num_active_vertices;

    /**
     * \brief A bit indicating (for all vertices) whether to
     * participate in the current minor-step (gather or scatter).
     */
    dense_bitset active_minorstep;

    /**
     * \brief A counter measuring the number of gathers that have been completed
     */
    atomic<size_t> completed_gathers;

    /**
     * \brief A counter measuring the number of applys that have been completed
     */
    atomic<size_t> completed_applys;

    /**
     * \brief A counter measuring the number of scatters that have been completed
     */
    atomic<size_t> completed_scatters;


    /**
     * \brief The shared counter used coordinate operations between
     * threads.
     */
    atomic<size_t> shared_lvid_counter;
    
    /**
     * \brief The engine type used to create express.
     */
    typedef powerlyra_sync_engine<VertexProgram> engine_type;

    /**
     * \brief The pair type used to synchronize vertex programs across machines.
     */
    typedef std::pair<vertex_id_type, vertex_program_type> vid_vprog_pair_type;

    /**
     * \brief The type of the express used to activate mirrors
     */
    typedef fiber_buffered_exchange<vid_vprog_pair_type> 
      activ_exchange_type;

    /**
     * \brief The type of buffer used by the express to activate mirrors
     */
    typedef typename activ_exchange_type::buffer_type activ_buffer_type;

    /**
     * \brief The distributed express used to activate mirrors
     * vertex programs.
     */
    activ_exchange_type activ_exchange;

    /**
     * \brief The tetrad type used to update vertex data and activate mirrors.
     */
    typedef tetrad<vertex_id_type, vertex_data_type, edge_dir_type, 
      vertex_program_type> vid_vdata_edir_vprog_tetrad_type;

    /**
     * \brief The type of the express used to update mirrors
     */
    typedef fiber_buffered_exchange<vid_vdata_edir_vprog_tetrad_type> 
      update_exchange_type;

    /**
     * \brief The type of buffer used by the express to update mirrors
     */
    typedef typename update_exchange_type::buffer_type update_buffer_type;

    /**
     * \brief The distributed express used to update mirrors
     * vertex programs.
     */
    update_exchange_type update_exchange;

    /**
     * \brief The pair type used to synchronize the results of the gather phase
     */
    typedef std::pair<vertex_id_type, gather_type> vid_gather_pair_type;

    /**
     * \brief The type of the exchange used to synchronize accums
     * accumulators
     */
    typedef fiber_buffered_exchange<vid_gather_pair_type> accum_exchange_type;

    /**
     * \brief The distributed exchange used to synchronize accums
     * accumulators.
     */
    accum_exchange_type accum_exchange;

    /**
     * \brief The pair type used to synchronize messages
     */
    typedef std::pair<vertex_id_type, message_type> vid_message_pair_type;

    /**
     * \brief The type of the exchange used to synchronize messages
     */
    typedef fiber_buffered_exchange<vid_message_pair_type> message_exchange_type;

    /**
     * \brief The distributed exchange used to synchronize messages
     */
    message_exchange_type message_exchange;


    /**
     * \brief The distributed aggregator used to manage background
     * aggregation.
     */
    aggregator_type aggregator;

    DECLARE_EVENT(EVENT_APPLIES);
    DECLARE_EVENT(EVENT_GATHERS);
    DECLARE_EVENT(EVENT_SCATTERS);
    DECLARE_EVENT(EVENT_ACTIVE_CPUS);
  public:

    /**
     * \brief Construct a synchronous engine for a given graph and options.
     *
     * The synchronous engine should be constructed after the graph
     * has been loaded (e.g., \ref graphlab::distributed_graph::load)
     * and the graphlab options have been set
     * (e.g., \ref graphlab::command_line_options).
     *
     * In the distributed engine the synchronous engine must be called
     * on all machines at the same time (in the same order) passing
     * the \ref graphlab::distributed_control object.  Upon
     * construction the synchronous engine allocates several
     * data-structures to store messages, gather accumulants, and
     * vertex programs and therefore may require considerable memory.
     *
     * The number of threads to create are read from
     * \ref graphlab_options::get_ncpus "opts.get_ncpus()".
     *
     * See the <a href="#engineopts">main class documentation</a>
     * for details on the available options.
     *
     * @param [in] dc Distributed controller to associate with
     * @param [in,out] graph A reference to the graph object that this
     * engine will modify. The graph must be fully constructed and
     * finalized.
     * @param [in] opts A graphlab::graphlab_options object specifying engine
     *                  parameters.  This is typically constructed using
     *                  \ref graphlab::command_line_options.
     */
    powerlyra_sync_engine(distributed_control& dc, graph_type& graph,
                       const graphlab_options& opts = graphlab_options());


    /**
     * \brief Start execution of the synchronous engine.
     *
     * The start function begins computation and does not return until
     * there are no remaining messages or until max_iterations has
     * been reached.
     *
     * The start() function modifies the data graph through the vertex
     * programs and so upon return the data graph should contain the
     * result of the computation.
     *
     * @return The reason for termination
     */
    execution_status::status_enum start();

    // documentation inherited from iengine
    size_t num_updates() const;

    // documentation inherited from iengine
    void signal(vertex_id_type vid,
                const message_type& message = message_type());

    // documentation inherited from iengine
    void signal_all(const message_type& message = message_type(),
                    const std::string& order = "shuffle");

    void signal_vset(const vertex_set& vset,
                    const message_type& message = message_type(),
                    const std::string& order = "shuffle");


    // documentation inherited from iengine
    float elapsed_seconds() const;

    // documentation inherited from iengine
    double execution_time() const;
    

    /**
     * \brief Get the current iteration number since start was last
     * invoked.
     *
     *  \return the current iteration
     */
    int iteration() const;


    /**
     * \brief Compute the total memory used by the entire distributed
     * system.
     *
     * @return The total memory used in bytes.
     */
    size_t total_memory_usage() const;

    /**
     * \brief Get a pointer to the distributed aggregator object.
     *
     * This is currently used by the \ref graphlab::iengine interface to
     * implement the calls to aggregation.
     *
     * @return a pointer to the local aggregator.
     */
    aggregator_type* get_aggregator();

    /**
     * \brief Initialize the engine and allocate datastructures for vertex, and lock,
     * clear all the messages.
     */
    void init();


  private:


    /**
     * \brief Resize the datastructures to fit the graph size (in case of dynamic graph). Keep all the messages
     * and caches.
     */
    void resize();

    /**
     * \brief This internal stop function is called by the \ref graphlab::context to
     * terminate execution of the engine.
     */
    void internal_stop();

    /**
     * \brief This function is called remote by the rpc to force the
     * engine to stop.
     */
    void rpc_stop();

    /**
     * \brief Signal a vertex.
     *
     * This function is called by the \ref graphlab::context.
     *
     * @param [in] lvid the local vertex id of the vertex to signal
     * @param [in] message the message to send to that vertex.
     */
    void internal_signal(const vertex_type& vertex,
                         const message_type& message);

    void internal_signal(const vertex_type& vertex);

    /**
     * \brief Called by the context to signal an arbitrary vertex.
     * This must be done by finding the owner of that vertex.
     *
     * @param [in] gvid the global vertex id of the vertex to signal
     * @param [in] message the message to send to that vertex.
     */
    void internal_signal_gvid(vertex_id_type gvid,
                                   const message_type& message = message_type());

    /**
     * \brief This function tests if this machine is the master of
     * gvid and signals if successful.
     */
    void internal_signal_rpc(vertex_id_type gvid,
                              const message_type& message = message_type());


    /**
     * \brief Post a to a previous gather for a give vertex.
     *
     * This function is called by the \ref graphlab::context.
     *
     * @param [in] vertex The vertex to which to post a change in the sum
     * @param [in] delta The change in that sum
     */
    void internal_post_delta(const vertex_type& vertex,
                             const gather_type& delta);

    /**
     * \brief Clear the cached gather for a vertex if one is
     * available.
     *
     * This function is called by the \ref graphlab::context.
     *
     * @param [in] vertex the vertex for which to clear the cache
     */
    void internal_clear_gather_cache(const vertex_type& vertex);


    // Program Steps ==========================================================


    void thread_launch_wrapped_event_counter(boost::function<void(void)> fn) {
      INCREMENT_EVENT(EVENT_ACTIVE_CPUS, 1);
      fn();
      DECREMENT_EVENT(EVENT_ACTIVE_CPUS, 1);
    }

    /**
     * \brief Executes ncpus copies of a member function each with a
     * unique consecutive id (thread id).
     *
     * This function is used by the main loop to execute each of the
     * stages in parallel.
     *
     * The member function must have the type:
     *
     * \code
     * void powerlyra_sync_engine::member_fun(size_t threadid);
     * \endcode
     *
     * This function runs an rmi barrier after termination
     *
     * @tparam the type of the member function.
     * @param [in] member_fun the function to call.
     */
    template<typename MemberFunction>
    void run_synchronous(MemberFunction member_fun) {
      shared_lvid_counter = 0;
      if (ncpus <= 1) {
        INCREMENT_EVENT(EVENT_ACTIVE_CPUS, 1);
      }
      // launch the initialization threads
      for(size_t i = 0; i < ncpus; ++i) {
        fiber_control::affinity_type affinity;
        affinity.clear(); affinity.set_bit(i);
        boost::function<void(void)> invoke = boost::bind(member_fun, this, i);
        threads.launch(boost::bind(
              &powerlyra_sync_engine::thread_launch_wrapped_event_counter,
              this,
              invoke), affinity);
      }
      // Wait for all threads to finish
      threads.join();
      rmi.barrier();
      if (ncpus <= 1) {
        DECREMENT_EVENT(EVENT_ACTIVE_CPUS, 1);
      }
    } // end of run_synchronous

    inline bool high_master_lvid(const lvid_type lvid);  
    inline bool low_master_lvid(const lvid_type lvid);
    inline bool high_mirror_lvid(const lvid_type lvid);  
    inline bool low_mirror_lvid(const lvid_type lvid);
    
    // /**
    //  * \brief Initialize all vertex programs by invoking
    //  * \ref graphlab::ivertex_program::init on all vertices.
    //  *
    //  * @param thread_id the thread to run this as which determines
    //  * which vertices to process.
    //  */
    // void initialize_vertex_programs(size_t thread_id);

    /**
     * \brief Synchronize all message data.
     *
     * @param thread_id the thread to run this as which determines
     * which vertices to process.
     */
    void exchange_messages(size_t thread_id);


    /**
     * \brief Invoke the \ref graphlab::ivertex_program::init function
     * on all vertex programs that have inbound messages.
     *
     * @param thread_id the thread to run this as which determines
     * which vertices to process.
     */
    void receive_messages(size_t thread_id);


    /**
     * \brief Execute the \ref graphlab::ivertex_program::gather function on all
     * vertices that received messages for the edges specified by the
     * \ref graphlab::ivertex_program::gather_edges.
     *
     * @param thread_id the thread to run this as which determines
     * which vertices to process.
     */
    void execute_gathers(size_t thread_id);




    /**
     * \brief Execute the \ref graphlab::ivertex_program::apply function on all
     * all vertices that received messages in this super-step (active).
     *
     * @param thread_id the thread to run this as which determines
     * which vertices to process.
     */
    void execute_applys(size_t thread_id);

    /**
     * \brief Execute the \ref graphlab::ivertex_program::scatter function on all
     * vertices that received messages for the edges specified by the
     * \ref graphlab::ivertex_program::scatter_edges.
     *
     * @param thread_id the thread to run this as which determines
     * which vertices to process.
     */
    void execute_scatters(size_t thread_id);

    // Data Synchronization ===================================================
    /**
     * \brief Send the activation messages (vertex program and edge set) 
     * for the local vertex id to all of its mirrors.
     *
     * @param [in] lvid the vertex to sync.  It must be the master of that vertex.
     */
    void send_activs(lvid_type lvid, size_t thread_id);

    /**
     * \brief do activation to local mirros.
     *
     * This function is a callback of express, and will be invoked when receives 
     * activation message.
     */
    void recv_activs();

    /**
     * \brief Send the update messages (vertex data, program and edge set) 
     * for the local vertex id to all of its mirrors.
     *
     * @param [in] lvid the vertex to sync.  It must be the master of that vertex.
     */
    void send_updates(lvid_type lvid, size_t thread_id);

    /**
     * \brief do update to local mirros.
     *
     * This function is a callback of express, and will be invoked when receives 
     * update message.
     */
    void recv_updates();

    /**
     * \brief Send the gather accum for the vertex id to its master.
     *
     * @param [in] lvid the vertex to send the gather value to
     * @param [in] accum the locally computed gather value.
     */
    void send_accum(lvid_type lvid, const gather_type& accum,
                        const size_t thread_id);


    /**
     * \brief Receive the gather accums from the buffered exchange.
     *
     * This function returns when there is nothing left in the
     * buffered exchange and should be called after the buffered
     * exchange has been flushed
     */
    void recv_accums();

    /**
     * \brief Send the scatter messages for the vertex id to its master.
     *
     * @param [in] lvid the vertex to send
     */
    void send_message(lvid_type lvid, const message_type& message, 
                        const size_t thread_id);

    /**
     * \brief Receive the scatter messages from the buffered exchange.
     *
     * This function returns when there is nothing left in the
     * buffered exchange and should be called after the buffered
     * exchange has been flushed
     */
    void recv_messages();


  }; // end of class powerlyra_sync_engine

























  /**
   * Constructs an synchronous distributed engine.
   * The number of threads to create are read from
   * opts::get_ncpus().
   *
   * Valid engine options (graphlab_options::get_engine_args()):
   * \arg \c max_iterations Sets the maximum number of iterations the
   * engine will run for.
   * \arg \c use_cache If set to true, partial gathers are cached.
   * See \ref gather_caching to understand the behavior of the
   * gather caching model and how it may be used to accelerate program
   * performance.
   *
   * \param dc Distributed controller to associate with
   * \param graph The graph to schedule over. The graph must be fully
   *              constructed and finalized.
   * \param opts A graphlab_options object containing options and parameters
   *             for the engine.
   */
  template<typename VertexProgram>
  powerlyra_sync_engine<VertexProgram>::
  powerlyra_sync_engine(distributed_control& dc,
                     graph_type& graph,
                     const graphlab_options& opts) :
    rmi(dc, this), graph(graph),
    ncpus(opts.get_ncpus()),
    threads(2*1024*1024 /* 2MB stack per fiber*/),
    thread_barrier(opts.get_ncpus()),
    max_iterations(-1), snapshot_interval(-1), iteration_counter(0),
    print_interval(5), timeout(0), sched_allv(false),
    activ_exchange(dc),
    update_exchange(dc),
    accum_exchange(dc),
    message_exchange(dc),
    aggregator(dc, graph, new context_type(*this, graph)) {
    // Process any additional options
    std::vector<std::string> keys = opts.get_engine_args().get_option_keys();
    per_thread_compute_time.resize(opts.get_ncpus());
    use_cache = false;
    foreach(std::string opt, keys) {
      if (opt == "max_iterations") {
        opts.get_engine_args().get_option("max_iterations", max_iterations);
        if (rmi.procid() == 0)
          logstream(LOG_EMPH) << "Engine Option: max_iterations = "
            << max_iterations << std::endl;
      } else if (opt == "timeout") {
        opts.get_engine_args().get_option("timeout", timeout);
        if (rmi.procid() == 0)
          logstream(LOG_EMPH) << "Engine Option: timeout = "
            << timeout << std::endl;
      } else if (opt == "use_cache") {
        opts.get_engine_args().get_option("use_cache", use_cache);
        if (rmi.procid() == 0)
          logstream(LOG_EMPH) << "Engine Option: use_cache = "
            << use_cache << std::endl;
      } else if (opt == "snapshot_interval") {
        opts.get_engine_args().get_option("snapshot_interval", snapshot_interval);
        if (rmi.procid() == 0)
          logstream(LOG_EMPH) << "Engine Option: snapshot_interval = "
            << snapshot_interval << std::endl;
      } else if (opt == "snapshot_path") {
        opts.get_engine_args().get_option("snapshot_path", snapshot_path);
        if (rmi.procid() == 0)
          logstream(LOG_EMPH) << "Engine Option: snapshot_path = "
            << snapshot_path << std::endl;
      } else if (opt == "sched_allv") {
        opts.get_engine_args().get_option("sched_allv", sched_allv);
        if (rmi.procid() == 0)
          logstream(LOG_EMPH) << "Engine Option: sched_allv = "
            << sched_allv << std::endl;
      } else {
        logstream(LOG_FATAL) << "Unexpected Engine Option: " << opt << std::endl;
      }
    }

    if (snapshot_interval >= 0 && snapshot_path.length() == 0) {
      logstream(LOG_FATAL)
        << "Snapshot interval specified, but no snapshot path" << std::endl;
    }
    INITIALIZE_EVENT_LOG(dc);
    ADD_CUMULATIVE_EVENT(EVENT_APPLIES, "Applies", "Calls");
    ADD_CUMULATIVE_EVENT(EVENT_GATHERS , "Gathers", "Calls");
    ADD_CUMULATIVE_EVENT(EVENT_SCATTERS , "Scatters", "Calls");
    ADD_INSTANTANEOUS_EVENT(EVENT_ACTIVE_CPUS, "Active Threads", "Threads");

    // Graph should has been finalized
    ASSERT_TRUE(graph.is_finalized());
    // Only support zone cuts
    ASSERT_TRUE(graph.get_cuts_type() == graph_type::HYBRID_CUTS 
                || graph.get_cuts_type() == graph_type::HYBRID_GINGER_CUTS);
    // if (rmi.procid() == 0) graph.dump_graph_info();

    init();
  } // end of powerlyra_sync_engine


  template<typename VertexProgram>
  void powerlyra_sync_engine<VertexProgram>:: init() {
    memory_info::log_usage("Before Engine Initialization");
    
    resize();
    
    // Clear up
    force_abort = false;
    iteration_counter = 0;
    completed_gathers = 0;
    completed_applys = 0;
    completed_scatters = 0;
    has_message.clear();
    has_gather_accum.clear();
    has_cache.clear();
    active_superstep.clear();
    active_minorstep.clear();

    memory_info::log_usage("After Engine Initialization");
  }


  template<typename VertexProgram>
  void powerlyra_sync_engine<VertexProgram>:: resize() {
    size_t l_nverts = graph.num_local_vertices();

    // Allocate vertex locks and vertex programs
    vlocks.resize(l_nverts);
    vertex_programs.resize(l_nverts);
    edge_dirs.resize(l_nverts);
    
    // Allocate messages and message bitset
    messages.resize(l_nverts, message_type());
    has_message.resize(l_nverts);
    
    // Allocate gather accumulators and accumulator bitset
    gather_accum.resize(l_nverts, gather_type());
    has_gather_accum.resize(l_nverts);

    // If caching is used then allocate cache data-structures
    if (use_cache) {
      gather_cache.resize(l_nverts, gather_type());
      has_cache.resize(l_nverts);
    }
    // Allocate bitset to track active vertices on each bitset.
    active_superstep.resize(l_nverts);
    active_minorstep.resize(l_nverts);
  }


  template<typename VertexProgram>
  typename powerlyra_sync_engine<VertexProgram>::aggregator_type*
  powerlyra_sync_engine<VertexProgram>::get_aggregator() {
    return &aggregator;
  } // end of get_aggregator


  template<typename VertexProgram>
  void powerlyra_sync_engine<VertexProgram>::internal_stop() {
    for (size_t i = 0; i < rmi.numprocs(); ++i)
      rmi.remote_call(i, &engine_type::rpc_stop);
  } // end of internal_stop


  template<typename VertexProgram>
  void powerlyra_sync_engine<VertexProgram>::rpc_stop() {
    force_abort = true;
  } // end of rpc_stop


  template<typename VertexProgram>
  void powerlyra_sync_engine<VertexProgram>::
  signal(vertex_id_type gvid, const message_type& message) {
    if (vlocks.size() != graph.num_local_vertices())
      resize();
    rmi.barrier();
    internal_signal_rpc(gvid, message);
    rmi.barrier();
  } // end of signal



  template<typename VertexProgram>
  void powerlyra_sync_engine<VertexProgram>::
  signal_all(const message_type& message, const std::string& order) {
    if (vlocks.size() != graph.num_local_vertices())
      resize();
    for(lvid_type lvid = 0; lvid < graph.num_local_vertices(); ++lvid) {
      if(graph.l_is_master(lvid)) {
        internal_signal(vertex_type(graph.l_vertex(lvid)), message);
      }
    }
  } // end of signal all


  template<typename VertexProgram>
  void powerlyra_sync_engine<VertexProgram>::
  signal_vset(const vertex_set& vset,
             const message_type& message, const std::string& order) {
    if (vlocks.size() != graph.num_local_vertices())
      resize();
    for(lvid_type lvid = 0; lvid < graph.num_local_vertices(); ++lvid) {
      if(graph.l_is_master(lvid) && vset.l_contains(lvid)) {
        internal_signal(vertex_type(graph.l_vertex(lvid)), message);
      }
    }
  } // end of signal all


  template<typename VertexProgram>
  void powerlyra_sync_engine<VertexProgram>::
  internal_signal(const vertex_type& vertex,
                  const message_type& message) {
    const lvid_type lvid = vertex.local_id();
    vlocks[lvid].lock();
    if( has_message.get(lvid) ) {
      messages[lvid] += message;
    } else {
      messages[lvid] = message;
      has_message.set_bit(lvid);
    }
    vlocks[lvid].unlock();
  } // end of internal_signal

  template<typename VertexProgram>
  void powerlyra_sync_engine<VertexProgram>::
  internal_signal(const vertex_type& vertex) {
    const lvid_type lvid = vertex.local_id();
    // set an empty message
    messages[lvid] = message_type();
    // atomic set is enough, without acquiring and releasing lock
    has_message.set_bit(lvid);
  } // end of internal_signal

  template<typename VertexProgram>
  void powerlyra_sync_engine<VertexProgram>::
  internal_signal_gvid(vertex_id_type gvid, const message_type& message) {
    procid_t proc = graph.master(gvid);
    if(proc == rmi.procid()) internal_signal_rpc(gvid, message);
    else rmi.remote_call(proc, 
                         &powerlyra_sync_engine<VertexProgram>::internal_signal_rpc,
                         gvid, message);
  } // end of internal_signal_gvid

  template<typename VertexProgram>
  void powerlyra_sync_engine<VertexProgram>::
  internal_signal_rpc(vertex_id_type gvid,
                      const message_type& message) {
    if (graph.is_master(gvid)) {
      internal_signal(graph.vertex(gvid), message);
    }
  } // end of internal_signal_rpc





  template<typename VertexProgram>
  void powerlyra_sync_engine<VertexProgram>::
  internal_post_delta(const vertex_type& vertex, const gather_type& delta) {
    const bool caching_enabled = !gather_cache.empty();
    if(caching_enabled) {
      const lvid_type lvid = vertex.local_id();
      vlocks[lvid].lock();
      if( has_cache.get(lvid) ) {
        gather_cache[lvid] += delta;
      } else {
        // You cannot add a delta to an empty cache.  A complete
        // gather must have been run.
        // gather_cache[lvid] = delta;
        // has_cache.set_bit(lvid);
      }
      vlocks[lvid].unlock();
    }
  } // end of post_delta


  template<typename VertexProgram>
  void powerlyra_sync_engine<VertexProgram>::
  internal_clear_gather_cache(const vertex_type& vertex) {
    const bool caching_enabled = !gather_cache.empty();
    const lvid_type lvid = vertex.local_id();
    if(caching_enabled && has_cache.get(lvid)) {
      vlocks[lvid].lock();
      gather_cache[lvid] = gather_type();
      has_cache.clear_bit(lvid);
      vlocks[lvid].unlock();
    }
  } // end of clear_gather_cache




  template<typename VertexProgram>
  size_t powerlyra_sync_engine<VertexProgram>::
  num_updates() const { return completed_applys.value; }

  template<typename VertexProgram>
  float powerlyra_sync_engine<VertexProgram>::
  elapsed_seconds() const { return timer::approx_time_seconds() - start_time; }

  template<typename VertexProgram>
  double powerlyra_sync_engine<VertexProgram>::
  execution_time() const { return exec_time; }

  template<typename VertexProgram>
  int powerlyra_sync_engine<VertexProgram>::
  iteration() const { return iteration_counter; }



  template<typename VertexProgram>
  size_t powerlyra_sync_engine<VertexProgram>::total_memory_usage() const {
    size_t allocated_memory = memory_info::allocated_bytes();
    rmi.all_reduce(allocated_memory);
    return allocated_memory;
  } // compute the total memory usage of the GraphLab system


  template<typename VertexProgram> 
  execution_status::status_enum powerlyra_sync_engine<VertexProgram>::
  start() {
    if (vlocks.size() != graph.num_local_vertices())
      resize();
    completed_gathers = 0;
    completed_applys = 0;
    completed_scatters = 0;
    rmi.barrier();

    // Initialization code ==================================================
    // Reset event log counters?
    // Start the timer
    start_time = timer::approx_time_seconds();
    exec_time = exch_time = recv_time =
      gather_time = apply_time = scatter_time = 0.0;
    graphlab::timer ti, bk_ti;
    iteration_counter = 0;
    force_abort = false;
    execution_status::status_enum termination_reason = execution_status::UNSET;
    aggregator.start();
    rmi.barrier();

    if (snapshot_interval == 0) {
      graph.save_binary(snapshot_path);
    }

    float last_print = -print_interval; // print the first iteration
    if (rmi.procid() == 0) {
      logstream(LOG_EMPH) << "Iteration counter will only output every "
                          << print_interval << " seconds."
                          << std::endl;
    }

    // Program Main loop ====================================================
    ti.start();
    while(iteration_counter < max_iterations && !force_abort ) {
      
      // Check first to see if we are out of time
      if(timeout != 0 && timeout < elapsed_seconds()) {
        termination_reason = execution_status::TIMEOUT;
        break;
      }

      bool print_this_round = (elapsed_seconds() - last_print) >= print_interval;
      if(rmi.procid() == 0 && print_this_round) {
        logstream(LOG_DEBUG)
          << rmi.procid() << ": Starting iteration: " << iteration_counter
          << std::endl;
        last_print = elapsed_seconds();
      }
      // Reset Active vertices ----------------------------------------------
      // Clear the active super-step and minor-step bits which will
      // be set upon receiving messages
      active_superstep.clear(); active_minorstep.clear();
      has_gather_accum.clear();
      num_active_vertices = 0;
      rmi.barrier();

      
      // Exchange Messages --------------------------------------------------
      // Powergraph: send messages from replicas to master
      //    - set messages and has_message
      // Lyra: none
      //
      // if (rmi.procid() == 0) std::cout << "Exchange messages..." << std::endl;
      bk_ti.start();
      run_synchronous( &powerlyra_sync_engine::exchange_messages );
      exch_time += bk_ti.current_time();
      /**
       * Post conditions:
       *   1) master (high and low) vertices have messages
       */

      // Receive Messages ---------------------------------------------------
      // 1. calculate the number of active vertices
      // 2. call init and gather_edges
      // 3. set active_superstep, active_minorstep and edge_dirs
      // 4. clear has_message
      // Powergraph: send vprog and edge_dirs from master to replicas
      //    - set vprog, edge_dirs and set active_minorstep 
      // Lyra: none
      //
      // if (rmi.procid() == 0) std::cout << "Receive messages..." << std::endl;
      bk_ti.start();
      run_synchronous( &powerlyra_sync_engine::receive_messages );
      if (sched_allv) active_minorstep.fill();
      has_message.clear();
      recv_time += bk_ti.current_time();
      /**
       * Post conditions:
       *   1) there are no messages remaining
       *   2) All masters that received messages have their
       *      active_superstep bit set
       *   3) All masters and mirrors that are to participate in the
       *      next gather phases have their active_minorstep bit
       *      set.
       *   4) num_active_vertices is the number of vertices that
       *      received messages.
       */

      // Check termination condition  ---------------------------------------
      size_t total_active_vertices = num_active_vertices;
      rmi.all_reduce(total_active_vertices);
      if (rmi.procid() == 0 && print_this_round)
        logstream(LOG_EMPH)
          << "\tActive vertices: " << total_active_vertices << std::endl;      
      if(total_active_vertices == 0 ) {
        termination_reason = execution_status::TASK_DEPLETION;
        break;
      }

      // Execute gather operations-------------------------------------------
      // 1. call pre_local_gather, gather and post_local_gather
      // 2. (master) set gather_accum and has_gather_accum
      // 3. clear active_minorstep
      // Powergraph: send gather_accum from replicas to master
      //    - set gather_accum and has_gather_accum
      // Lyra: none
      //
      // if (rmi.procid() == 0) std::cout << "Gathering..." << std::endl;
      bk_ti.start();
      run_synchronous( &powerlyra_sync_engine::execute_gathers );
      // Clear the minor step bit since only super-step vertices
      // (only master vertices are required to participate in the
      // apply step)
      active_minorstep.clear();
      gather_time += bk_ti.current_time();
      /**
       * Post conditions:
       *   1) gather_accum for all master vertices contains the
       *      result of all the gathers (even if they are drawn from
       *      cache)
       *   2) No minor-step bits are set
       */

      // Execute Apply Operations -------------------------------------------
      // 1. call apply and scatter_edges
      // 2. set edge_dirs and active_minorstep
      // 3. send vdata, vprog and edge_dirs from master to replicas
      //    - set vdata, vprog, edge_dirs and active_minorstep
      //
      // if (rmi.procid() == 0) std::cout << "Applying..." << std::endl;
      bk_ti.start();
      run_synchronous( &powerlyra_sync_engine::execute_applys );
      apply_time += bk_ti.current_time();
      /**
       * Post conditions:
       *   1) any changes to the vertex data have been synchronized
       *      with all mirrors.
       *   2) all gather accumulators have been cleared
       *   3) If a vertex program is participating in the scatter
       *      phase its minor-step bit has been set to active (both
       *      masters and mirrors) and the vertex program has been
       *      synchronized with the mirrors.
       */


      // Execute Scatter Operations -----------------------------------------
      // 1. call scatter (signal: set messages and has_message)
      //
      // if (rmi.procid() == 0) std::cout << "Scattering..." << std::endl;
      bk_ti.start();
      run_synchronous( &powerlyra_sync_engine::execute_scatters );
      scatter_time += bk_ti.current_time();
      /**
       * Post conditions:
       *   1) NONE
       */
      if(rmi.procid() == 0 && print_this_round)
        logstream(LOG_DEBUG) << "\t Running Aggregators" << std::endl;
      // probe the aggregator
      aggregator.tick_synchronous();

      ++iteration_counter;

      if (snapshot_interval > 0 && iteration_counter % snapshot_interval == 0) {
        graph.save_binary(snapshot_path);
      }
    }
    exec_time = ti.current_time();

    if (rmi.procid() == 0) {
      logstream(LOG_EMPH) << iteration_counter
                          << " iterations completed." << std::endl;
    }
    // Final barrier to ensure that all engines terminate at the same time
    double total_compute_time = 0;
    for (size_t i = 0;i < per_thread_compute_time.size(); ++i) {
      total_compute_time += per_thread_compute_time[i];
    }
    std::vector<double> all_compute_time_vec(rmi.numprocs());
    all_compute_time_vec[rmi.procid()] = total_compute_time;
    rmi.all_gather(all_compute_time_vec);

    /*logstream(LOG_INFO) << "Local Calls(G|A|S): "
                        << completed_gathers.value << "|" 
                        << completed_applys.value << "|"
                        << completed_scatters.value 
                        << std::endl;*/
    
    size_t global_completed = completed_applys;
    rmi.all_reduce(global_completed);
    completed_applys = global_completed;
    rmi.cout() << "Updates: " << completed_applys.value << "\n";

#ifdef TUNING
    global_completed = completed_gathers;
    rmi.all_reduce(global_completed);
    completed_gathers = global_completed;

    global_completed = completed_scatters;
    rmi.all_reduce(global_completed);
    completed_scatters = global_completed;
#endif

    if (rmi.procid() == 0) {
#ifdef TUNING
      logstream(LOG_INFO) << "Total Calls(G|A|S): " 
                          << completed_gathers.value << "|" 
                          << completed_applys.value << "|"
                          << completed_scatters.value 
                          << std::endl;
#endif
      logstream(LOG_INFO) << "Compute Balance: ";
      for (size_t i = 0;i < all_compute_time_vec.size(); ++i) {
        logstream(LOG_INFO) << all_compute_time_vec[i] << " ";
      }
      logstream(LOG_INFO) << std::endl;      
      logstream(LOG_EMPH) << "      Execution Time: " << exec_time << std::endl;
      logstream(LOG_EMPH) << "Breakdown(X|R|G|A|S): " 
                          << exch_time << "|"
                          << recv_time << "|"
                          << gather_time << "|"
                          << apply_time << "|"
                          << scatter_time
                          << std::endl;
    }

    rmi.full_barrier();
    // Stop the aggregator
    aggregator.stop();
    // return the final reason for termination
    return termination_reason;
  } // end of start

  template<typename VertexProgram>
  inline bool powerlyra_sync_engine<VertexProgram>::
  high_master_lvid(const lvid_type lvid) {
    return graph.l_type(lvid) == graph_type::HIGH_MASTER;
  }

  template<typename VertexProgram>
  inline bool powerlyra_sync_engine<VertexProgram>::
  low_master_lvid(const lvid_type lvid) {
    return graph.l_type(lvid) == graph_type::LOW_MASTER;
  }

  template<typename VertexProgram>
  inline bool powerlyra_sync_engine<VertexProgram>::
  high_mirror_lvid(const lvid_type lvid) {
    return graph.l_type(lvid) == graph_type::HIGH_MIRROR;
  }

  template<typename VertexProgram>
  inline bool powerlyra_sync_engine<VertexProgram>::
  low_mirror_lvid(const lvid_type lvid) {
    return graph.l_type(lvid) == graph_type::LOW_MIRROR;
  }

  template<typename VertexProgram>
  void powerlyra_sync_engine<VertexProgram>::
  exchange_messages(const size_t thread_id) {
    context_type context(*this, graph);
    const size_t TRY_RECV_MOD = 100;
    size_t vcount = 0;
    fixed_dense_bitset<8 * sizeof(size_t)> local_bitset; // a word-size = 64 bit

    while (1) {
      // increment by a word at a time
      lvid_type lvid_block_start =
                  shared_lvid_counter.inc_ret_last(8 * sizeof(size_t));
      if (lvid_block_start >= graph.num_local_vertices()) break;
      // get the bit field from has_message
      size_t lvid_bit_block = has_message.containing_word(lvid_block_start);
      if (lvid_bit_block == 0) continue;
      // initialize a word sized bitfield
      local_bitset.clear();
      local_bitset.initialize_from_mem(&lvid_bit_block, sizeof(size_t));
      foreach(size_t lvid_block_offset, local_bitset) {
        lvid_type lvid = lvid_block_start + lvid_block_offset;
        if (lvid >= graph.num_local_vertices()) break;

        // [TARGET]: High/Low-degree Mirrors
        // only if scatter via in-edges will set has_message of low_mirror
        if(!graph.l_is_master(lvid)) {        
          send_message(lvid, messages[lvid], thread_id);
          has_message.clear_bit(lvid);
          // clear the message to save memory
          messages[lvid] = message_type();
          ++vcount;
        }
        if(vcount % TRY_RECV_MOD == 0) recv_messages();
      }
    } // end of loop over vertices to send messages
    message_exchange.partial_flush();
    thread_barrier.wait();
    if(thread_id == 0) message_exchange.flush();
    thread_barrier.wait();
    recv_messages();    
  } // end of exchange_messages


  template<typename VertexProgram>
  void powerlyra_sync_engine<VertexProgram>::
  receive_messages(const size_t thread_id) {
    context_type context(*this, graph);
    fixed_dense_bitset<8 * sizeof(size_t)> local_bitset; // a word-size = 64 bit
    const size_t TRY_RECV_MOD = 100;
    size_t vcount = 0;
    size_t nactive_inc = 0;
    
    while (1) {
      // increment by a word at a time
      lvid_type lvid_block_start =
                  shared_lvid_counter.inc_ret_last(8 * sizeof(size_t));
      if (lvid_block_start >= graph.num_local_vertices()) break;
      // get the bit field from has_message
      size_t lvid_bit_block = has_message.containing_word(lvid_block_start);
      if (lvid_bit_block == 0) continue;
      // initialize a word sized bitfield
      local_bitset.clear();
      local_bitset.initialize_from_mem(&lvid_bit_block, sizeof(size_t));
      foreach(size_t lvid_block_offset, local_bitset) {
        lvid_type lvid = lvid_block_start + lvid_block_offset;
        if (lvid >= graph.num_local_vertices()) break;

        // [TARGET]: High/Low-degree Masters
        ASSERT_TRUE(graph.l_is_master(lvid));
        // The vertex becomes active for this superstep
        active_superstep.set_bit(lvid);
        ++nactive_inc;
        // Pass the message to the vertex program
        const vertex_type vertex(graph.l_vertex(lvid));
        vertex_programs[lvid].init(context, vertex, messages[lvid]);
        // clear the message to save memory
        messages[lvid] = message_type();
        if (sched_allv) continue;
        // Determine if the gather should be run
        const vertex_program_type& const_vprog = vertex_programs[lvid];
        edge_dirs[lvid] = const_vprog.gather_edges(context, vertex);
        if(edge_dirs[lvid] != graphlab::NO_EDGES) {
          active_minorstep.set_bit(lvid);
          // send Gx1 msgs
          if (high_master_lvid(lvid)
              || (low_master_lvid(lvid) // only if gather via out-edge
                && ((edge_dirs[lvid] == graphlab::OUT_EDGES) 
                    || (edge_dirs[lvid] == graphlab::ALL_EDGES)))) {
            send_activs(lvid, thread_id);
            ++vcount;
          }
        }
      }
      if(vcount % TRY_RECV_MOD == 0) recv_activs();
    }
    num_active_vertices += nactive_inc;
    activ_exchange.partial_flush();
    thread_barrier.wait();
    // Flush the buffer and finish receiving any remaining activations.
    if(thread_id == 0) activ_exchange.flush(); // call full_barrier
    thread_barrier.wait();
    recv_activs();
  } // end of receive_messages


  template<typename VertexProgram>
  void powerlyra_sync_engine<VertexProgram>::
  execute_gathers(const size_t thread_id) {
    context_type context(*this, graph);
    const bool caching_enabled = !gather_cache.empty();
    fixed_dense_bitset<8 * sizeof(size_t)> local_bitset; // a word-size = 64 bit    
    const size_t TRY_RECV_MOD = 1000;
    size_t vcount = 0;
    size_t ngather_inc = 0;
    timer ti;
    
    while (1) {
      // increment by a word at a time
      lvid_type lvid_block_start =
                  shared_lvid_counter.inc_ret_last(8 * sizeof(size_t));
      if (lvid_block_start >= graph.num_local_vertices()) break;
      // get the bit field from has_message
      size_t lvid_bit_block = active_minorstep.containing_word(lvid_block_start);
      if (lvid_bit_block == 0) continue;
      // initialize a word sized bitfield
      local_bitset.clear();
      local_bitset.initialize_from_mem(&lvid_bit_block, sizeof(size_t));
      foreach(size_t lvid_block_offset, local_bitset) {
        lvid_type lvid = lvid_block_start + lvid_block_offset;
        if (lvid >= graph.num_local_vertices()) break;

        // [TARGET]: High/Low-degree Masters, and High/Low-degree Mirrors
        bool accum_is_set = false;
        gather_type accum = gather_type();
        // if caching is enabled and we have a cache entry then use
        // that as the accum
        if (caching_enabled && has_cache.get(lvid)) {
          accum = gather_cache[lvid];
          accum_is_set = true;
        } else {
          // recompute the local contribution to the gather
          const vertex_program_type& vprog = vertex_programs[lvid];
          local_vertex_type local_vertex = graph.l_vertex(lvid);
          const vertex_type vertex(local_vertex);
          const edge_dir_type gather_dir = vprog.gather_edges(context, vertex);
          
          size_t edges_touched = 0;
          vprog.pre_local_gather(accum);
          // Loop over in edges
          if (gather_dir == IN_EDGES || gather_dir == ALL_EDGES) {
            foreach(local_edge_type local_edge, local_vertex.in_edges()) {
              edge_type edge(local_edge);
              if(accum_is_set) { // \todo hint likely
                accum += vprog.gather(context, vertex, edge);
              } else {
                accum = vprog.gather(context, vertex, edge);
                accum_is_set = true;
              }
              ++edges_touched;
            }
          } // end of if in_edges/all_edges
          // Loop over out edges
          if(gather_dir == OUT_EDGES || gather_dir == ALL_EDGES) {
            foreach(local_edge_type local_edge, local_vertex.out_edges()) {
              edge_type edge(local_edge);
              if(accum_is_set) { // \todo hint likely
                accum += vprog.gather(context, vertex, edge);
              } else {
                accum = vprog.gather(context, vertex, edge);
                accum_is_set = true;
              }
              ++edges_touched;
            }
          } // end of if out_edges/all_edges
          INCREMENT_EVENT(EVENT_GATHERS, edges_touched);
          ++ngather_inc;
          vprog.post_local_gather(accum);
          
          // If caching is enabled then save the accumulator to the
          // cache for future iterations.  Note that it is possible
          // that the accumulator was never set in which case we are
          // effectively "zeroing out" the cache.
          if(caching_enabled && accum_is_set) {
            gather_cache[lvid] = accum; has_cache.set_bit(lvid);
          } // end of if caching enabled
        }

        // If the accum contains a value for the gather
        if (accum_is_set) { send_accum(lvid, accum, thread_id); ++vcount; }
        if(!graph.l_is_master(lvid)) {
          // if this is not the master clear the vertex program
          vertex_programs[lvid] = vertex_program_type();
        }

        // try to recv gathers if there are any in the buffer
        if(vcount % TRY_RECV_MOD == 0) recv_accums();
      }
    } // end of loop over vertices to compute gather accumulators
    completed_gathers += ngather_inc;
    per_thread_compute_time[thread_id] += ti.current_time();
    accum_exchange.partial_flush();
    thread_barrier.wait();
    if(thread_id == 0) accum_exchange.flush(); // full_barrier
    thread_barrier.wait();
    recv_accums();
  } // end of execute_gathers


  template<typename VertexProgram>
  void powerlyra_sync_engine<VertexProgram>::
  execute_applys(const size_t thread_id) {
    context_type context(*this, graph);
    fixed_dense_bitset<8 * sizeof(size_t)> local_bitset;  // allocate a word size = 64bits
    const size_t TRY_RECV_MOD = 1000;
    size_t vcount = 0;
    size_t napply_inc = 0;
    timer ti;
    
    while (1) {
      // increment by a word at a time
      lvid_type lvid_block_start =
                  shared_lvid_counter.inc_ret_last(8 * sizeof(size_t));
      if (lvid_block_start >= graph.num_local_vertices()) break;
      // get the bit field from has_message
      size_t lvid_bit_block = active_superstep.containing_word(lvid_block_start);
      if (lvid_bit_block == 0) continue;
      // initialize a word sized bitfield
      local_bitset.clear();
      local_bitset.initialize_from_mem(&lvid_bit_block, sizeof(size_t));
      foreach(size_t lvid_block_offset, local_bitset) {
        lvid_type lvid = lvid_block_start + lvid_block_offset;
        if (lvid >= graph.num_local_vertices()) break;

        // [TARGET]: High/Low-degree Masters
        // Only master vertices can be active in a super-step
        ASSERT_TRUE(graph.l_is_master(lvid));
        vertex_type vertex(graph.l_vertex(lvid));
        // Get the local accumulator.  Note that it is possible that
        // the gather_accum was not set during the gather.
        const gather_type& accum = gather_accum[lvid];
        INCREMENT_EVENT(EVENT_APPLIES, 1);
        vertex_programs[lvid].apply(context, vertex, accum);
        // record an apply as a completed task
        ++napply_inc;
        // clear the accumulator to save some memory
        gather_accum[lvid] = gather_type();
        // determine if a scatter operation is needed
        const vertex_program_type& const_vprog = vertex_programs[lvid];
        const vertex_type const_vertex = vertex;
        edge_dirs[lvid] = const_vprog.scatter_edges(context, const_vertex);

        if (edge_dirs[lvid] != graphlab::NO_EDGES)
          active_minorstep.set_bit(lvid);
        else
          vertex_programs[lvid] = vertex_program_type();

        // send Ax1 and Sx1
        send_updates(lvid, thread_id);
        
        if(++vcount % TRY_RECV_MOD == 0) recv_updates();
      }
    } // end of loop over vertices to run apply
    completed_applys += napply_inc;
    per_thread_compute_time[thread_id] += ti.current_time();
    update_exchange.partial_flush();
    thread_barrier.wait();
    // Flush the buffer and finish receiving any remaining updates.
    if(thread_id == 0) update_exchange.flush(); // full_barrier
    thread_barrier.wait();
    recv_updates();
  } // end of execute_applys


  template<typename VertexProgram>
  void powerlyra_sync_engine<VertexProgram>::
  execute_scatters(const size_t thread_id) {
    context_type context(*this, graph);
    fixed_dense_bitset<8 * sizeof(size_t)> local_bitset; // allocate a word size = 64 bits
    size_t nscatter_inc = 0;
    timer ti;
    
    while (1) {
      // increment by a word at a time
      lvid_type lvid_block_start =
                  shared_lvid_counter.inc_ret_last(8 * sizeof(size_t));
      if (lvid_block_start >= graph.num_local_vertices()) break;
      // get the bit field from has_message
      size_t lvid_bit_block = active_minorstep.containing_word(lvid_block_start);
      if (lvid_bit_block == 0) continue;
      // initialize a word sized bitfield
      local_bitset.clear();
      local_bitset.initialize_from_mem(&lvid_bit_block, sizeof(size_t));
      foreach(size_t lvid_block_offset, local_bitset) {
        lvid_type lvid = lvid_block_start + lvid_block_offset;
        if (lvid >= graph.num_local_vertices()) break;

        // [TARGET]: High/Low-degree Masters, and High/Low-degree Mirrors
        const vertex_program_type& vprog = vertex_programs[lvid];
        local_vertex_type local_vertex = graph.l_vertex(lvid);
        const vertex_type vertex(local_vertex);
        const edge_dir_type scatter_dir = edge_dirs[lvid];

        size_t edges_touched = 0;
        // Loop over in edges
        if(scatter_dir == IN_EDGES || scatter_dir == ALL_EDGES) {
          foreach(local_edge_type local_edge, local_vertex.in_edges()) {
            edge_type edge(local_edge);
            vprog.scatter(context, vertex, edge);
            ++edges_touched;
          }
        } // end of if in_edges/all_edges
        // Loop over out edges
        if(scatter_dir == OUT_EDGES || scatter_dir == ALL_EDGES) {
          foreach(local_edge_type local_edge, local_vertex.out_edges()) {
            edge_type edge(local_edge);
            vprog.scatter(context, vertex, edge);
            ++edges_touched;
          }
        } // end of if out_edges/all_edges
        INCREMENT_EVENT(EVENT_SCATTERS, edges_touched);
        // Clear the vertex program
        vertex_programs[lvid] = vertex_program_type();
        ++nscatter_inc;
      } // end of if active on this minor step
    } // end of loop over vertices to complete scatter operation
    completed_scatters += nscatter_inc;
    per_thread_compute_time[thread_id] += ti.current_time();
  } // end of execute_scatters



  // Data Synchronization ===================================================
  template<typename VertexProgram>
  inline void powerlyra_sync_engine<VertexProgram>::
  send_activs(lvid_type lvid, const size_t thread_id) {
    ASSERT_TRUE(graph.l_is_master(lvid));
    const vertex_id_type vid = graph.global_vid(lvid);
    local_vertex_type vertex = graph.l_vertex(lvid);
    foreach(const procid_t& mirror, vertex.mirrors()) {
      activ_exchange.send(mirror,
                          std::make_pair(vid, vertex_programs[lvid]));
    }
  } // end of send_activ

  template<typename VertexProgram>
  inline void powerlyra_sync_engine<VertexProgram>::
  recv_activs() {
    typename activ_exchange_type::recv_buffer_type recv_buffer;
    while(activ_exchange.recv(recv_buffer)) {
      for (size_t i = 0;i < recv_buffer.size(); ++i) {
        typename activ_exchange_type::buffer_type& buffer = recv_buffer[i].buffer;
        foreach(const vid_vprog_pair_type& pair, buffer) {
          const lvid_type lvid = graph.local_vid(pair.first);
          ASSERT_FALSE(graph.l_is_master(lvid));
          vertex_programs[lvid] = pair.second;
          active_minorstep.set_bit(lvid);
        }
      }
    }
  } // end of recv activs programs

  template<typename VertexProgram>
  inline void powerlyra_sync_engine<VertexProgram>::
  send_updates(lvid_type lvid, const size_t thread_id) {
    ASSERT_TRUE(graph.l_is_master(lvid));
    const vertex_id_type vid = graph.global_vid(lvid);
    local_vertex_type vertex = graph.l_vertex(lvid);
    foreach(const procid_t& mirror, vertex.mirrors()) {
      update_exchange.send(mirror, 
                           make_tetrad(vid, 
                                       vertex.data(), 
                                       edge_dirs[lvid],
                                       vertex_programs[lvid]));
    }
  } // end of send_update

  template<typename VertexProgram>
  inline void powerlyra_sync_engine<VertexProgram>::
  recv_updates() {
    typename update_exchange_type::recv_buffer_type recv_buffer;
    while(update_exchange.recv(recv_buffer)) {
      for (size_t i = 0;i < recv_buffer.size(); ++i) {
        typename update_exchange_type::buffer_type& buffer = recv_buffer[i].buffer;
        foreach(const vid_vdata_edir_vprog_tetrad_type& t, buffer) {
          const lvid_type lvid = graph.local_vid(t.first);
          ASSERT_FALSE(graph.l_is_master(lvid));
          graph.l_vertex(lvid).data() = t.second;
          if (t.third != graphlab::NO_EDGES) {
            edge_dirs[lvid] = t.third;
            vertex_programs[lvid] = t.fourth;
            active_minorstep.set_bit(lvid);
          }
        }
      }
    }
  } // end of recv_updates

  template<typename VertexProgram>
  inline void powerlyra_sync_engine<VertexProgram>::
  send_accum(lvid_type lvid, const gather_type& accum, const size_t thread_id) {
    if(graph.l_is_master(lvid)) {
      vlocks[lvid].lock();
      if(has_gather_accum.get(lvid)) {
        gather_accum[lvid] += accum;
      } else {
        gather_accum[lvid] = accum;
        has_gather_accum.set_bit(lvid);
      }
      vlocks[lvid].unlock();
    } else {
      const procid_t master = graph.l_master(lvid);
      const vertex_id_type vid = graph.global_vid(lvid);
      accum_exchange.send(master, std::make_pair(vid, accum));
    }
  } // end of send_accum

  template<typename VertexProgram>
  inline void powerlyra_sync_engine<VertexProgram>::
  recv_accums() {
    typename accum_exchange_type::recv_buffer_type recv_buffer;
    while(accum_exchange.recv(recv_buffer)) {
      for (size_t i = 0; i < recv_buffer.size(); ++i) {
        typename accum_exchange_type::buffer_type& buffer = recv_buffer[i].buffer;
        foreach(const vid_gather_pair_type& pair, buffer) {
          const lvid_type lvid = graph.local_vid(pair.first);
          const gather_type& acc = pair.second;
          ASSERT_TRUE(graph.l_is_master(lvid));
          vlocks[lvid].lock();
          if(has_gather_accum.get(lvid)) {
            gather_accum[lvid] += acc;
          } else {
            gather_accum[lvid] = acc;
            has_gather_accum.set_bit(lvid);
          }
          vlocks[lvid].unlock();
        }
      }
    }
  } // end of recv_accums


  template<typename VertexProgram>
  inline void powerlyra_sync_engine<VertexProgram>::
  send_message(lvid_type lvid, const message_type& message, const size_t thread_id) {
    ASSERT_FALSE(graph.l_is_master(lvid));
    const procid_t master = graph.l_master(lvid);
    const vertex_id_type vid = graph.global_vid(lvid);
    message_exchange.send(master, std::make_pair(vid, message));
  } // end of send_message

  template<typename VertexProgram>
  inline void powerlyra_sync_engine<VertexProgram>::
  recv_messages() {
    typename message_exchange_type::recv_buffer_type recv_buffer;
    while(message_exchange.recv(recv_buffer)) {
      for (size_t i = 0;i < recv_buffer.size(); ++i) {
        typename message_exchange_type::buffer_type& buffer = recv_buffer[i].buffer;
        foreach(const vid_message_pair_type& pair, buffer) {
          const lvid_type lvid = graph.local_vid(pair.first);
          const message_type& msg = pair.second;
          ASSERT_TRUE(graph.l_is_master(lvid));
          vlocks[lvid].lock();
          if(has_message.get(lvid)) {
            messages[lvid] += msg;
          } else {
            messages[lvid] = msg;
            has_message.set_bit(lvid);
          }
          vlocks[lvid].unlock();
        }
      }
    }
  } // end of recv_messages

}; // namespace


#include <graphlab/macros_undef.hpp>

#endif
