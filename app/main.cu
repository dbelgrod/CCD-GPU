#include <gpubf/simulation.cuh>
#include <gpubf/groundtruth.cuh>
#include <gpubf/util.cuh>
#include <gpubf/io.cuh>

#include <ccdgpu/helper.cuh>

int main( int argc, char **argv )
{
    vector<char*> compare;

    char* filet0;
    char* filet1;

    filet0 = argv[1];
    if (is_file_exist(argv[2])) //CCD
        filet1 = argv[2];
    else //static CD
        filet1 = argv[1];
    
    vector<Aabb> boxes;
    Eigen::MatrixXd vertices_t0;
    Eigen::MatrixXd vertices_t1;
    Eigen::MatrixXi faces; 
    Eigen::MatrixXi edges;

    parseMesh(filet0, filet1, vertices_t0, vertices_t1, faces, edges);
    constructBoxes(vertices_t0, vertices_t1, faces, edges, boxes);
    int N = boxes.size();
    int nbox = 0;
    int parallel = 0;
    int devcount = 1;

    // std::copy(from_vector.begin(), from_vector.end(), to_vector.begin());
    
    int o;
    while ((o = getopt (argc, argv, "c:n:b:")) != -1)
    {
        switch (o)
        {
            case 'c':
                optind--;
                for( ;optind < argc && *argv[optind] != '-'; optind++)
                {
                    compare.push_back(argv[optind]);
                    // compare_mathematica(overlaps, argv[optind]); 
                }
                break;
            case 'n':
                N = atoi(optarg);
                break;
            case 'b':
                nbox = atoi(optarg);
                break;
        }
    }
    vector<pair<int,int>> overlaps;
    int2 * d_overlaps;
    int * d_count;
    run_sweep_pieces(boxes.data(), N, nbox, overlaps, d_overlaps, d_count, parallel, devcount);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaGetLastError() ); 
    
    // copy overlap count
    int count;
    gpuErrchk(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Count %i\n", count);
    gpuErrchk( cudaGetLastError() ); 

    // Allocate boxes to GPU 
    Aabb * d_boxes;
    cudaMalloc((void**)&d_boxes, sizeof(Aabb)*N);
    cudaMemcpy(d_boxes, boxes.data(), sizeof(Aabb)*N, cudaMemcpyHostToDevice);
    gpuErrchk( cudaGetLastError() ); 

    
    float3 * d_queries;
    cudaMalloc((void**)&d_queries, sizeof(float3)*8*count);
    gpuErrchk( cudaGetLastError() ); 

    printf("Copying vertices\n");
    double * d_vertices_t0;
    double * d_vertices_t1;
    cudaMalloc((void**)&d_vertices_t0, sizeof(double)*vertices_t0.size());
    cudaMalloc((void**)&d_vertices_t1, sizeof(double)*vertices_t1.size());
    cudaMemcpy(d_vertices_t0, vertices_t0.data(), sizeof(double)*vertices_t0.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vertices_t1, vertices_t1.data(), sizeof(double)*vertices_t1.size(), cudaMemcpyHostToDevice);
    printf("c++ V0 rows: %i cols: %i\n", vertices_t0.rows(), vertices_t0.cols());
    printf("c++ V0[0] %.6f\n", vertices_t0.cast<float>().row(0)[1]);

    int Vrows = vertices_t0.rows();
    assert(Vrows == vertices_t1.rows());
    
    gpuErrchk( cudaGetLastError() ); 
    addData<<<count / parallel + 1, parallel>>>(d_overlaps, d_boxes, d_vertices_t0, d_vertices_t1, Vrows, count, d_queries );
    cudaDeviceSynchronize();
    gpuErrchk( cudaGetLastError() ); 

    cudaFree(d_overlaps);
    cudaFree(d_boxes);
    cudaFree(d_vertices_t0);
    cudaFree(d_vertices_t1);

    cudaDeviceSynchronize();
    
    vector<array<array<float, 3>, 8>> queries;
    for (int i=0; i < overlaps.size(); i++)
    {
        int aid = overlaps[i].first;
        int bid = overlaps[i].second;

        Aabb a = boxes[aid];
        Aabb b = boxes[bid];  

        addData(a, b, vertices_t0, vertices_t1, queries);
    }
    printf("c++ queries[0].x %.6f\n", queries[0][0]);
    

    int size = queries.size();
    cout << "data loaded, size " << queries.size() << endl;
    double tavg = 0;
    int max_query_cp_size = 1e7;
    int start_id = 0;

    
    vector<float> tois;
    vector<bool> result_list;
    result_list.resize(size);
    tois.resize(size);

    float3 * d_tmp_queries;
    while (1)
    {
        vector<bool> tmp_results;
        vector<array<array<Scalar, 3>, 8>> tmp_queries;
        vector<Scalar> tmp_tois;

        int remain = size - start_id;
        double tmp_tall;

        if (remain <= 0)
            break;

        int tmp_nbr = min(remain, max_query_cp_size);
        tmp_results.resize(tmp_nbr);
        tmp_queries.resize(tmp_nbr);
        tmp_tois.resize(tmp_nbr);

        cudaMalloc((void**)&d_tmp_queries, sizeof(float3)*8*tmp_nbr);
        cudaMemcpy(d_tmp_queries, d_queries + start_id, sizeof(float3)*8*tmp_nbr, cudaMemcpyDeviceToDevice);
        for (int i = 0; i < tmp_nbr; i++)
        {
            tmp_queries[i] = queries[start_id + i];
        }
        bool is_edge_edge = true;
        // all_ccd_run(tmp_queries, is_edge_edge, tmp_results, tmp_tall, tmp_tois, parallel);
        
        all_ccd_run(d_tmp_queries, tmp_nbr, is_edge_edge, tmp_results, tmp_tall, tmp_tois, parallel);

        tavg += tmp_tall;
        for (int i = 0; i < tmp_nbr; i++)
        {
            result_list[start_id + i] = tmp_results[i];
            tois[start_id + i] = tmp_tois[i];
        }

        start_id += tmp_nbr;
    }
    tavg /= size;
    cout << "avg time " << tavg << endl;
    
    for (auto i : compare)
    {
        compare_mathematica(overlaps, result_list, i);
    }
    cout << endl;

    // Mesh --> Boxes --> Broadphase --> (Boxes[2] ->float/double[8]) --> Narrowphase
    // Go back to old code and make overlaps as pairs
    
}