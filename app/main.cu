#include <unistd.h>
#include <iostream>

#include <gpubf/groundtruth.cuh>
#include <gpubf/util.cuh>
#include <gpubf/io.cuh>

#include <ccdgpu/helper.cuh>

using namespace ccdgpu;

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
    // run_sweep_pieces(boxes.data(), N, nbox, overlaps, d_overlaps, d_count, parallel, devcount);
    vector<pair<int,int>> overlaps;
    vector<float> tois;
    vector<bool> result_list;
    run_ccd(boxes, vertices_t0, vertices_t1, N, nbox, parallel, devcount, overlaps, result_list, tois);
    
    for (auto i : compare)
    {
        compare_mathematica(overlaps, result_list, i);
    }
    cout << endl;

    // Mesh --> Boxes --> Broadphase --> (Boxes[2] ->float/double[8]) --> Narrowphase
    // Go back to old code and make overlaps as pairs
    
}