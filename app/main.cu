
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <ctype.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/edges.h>

#include <gpubf/simulation.h>
#include <gpubf/groundtruth.h>
#include <gpubf/util.cuh>
#include <gpubf/io.cuh>


// #include <gpubf/klee.cuh>

using namespace std;

void addData(
    const Aabb &a, 
    const Aabb&b, 
    std::vector<std::array<std::array<float, 3>, 8>>& queries)
{
    auto is_vertex = [&](Aabb x){return x.vertexIds.y < 0 ;};
    auto is_edge = [&](Aabb x){return !is_vertex(x) && x.vertexIds.z < 0 ;};
    auto is_face = [&](Aabb x){return !is_vertex(x) && !is_edge(x);};

    if (is_vertex(a) && is_face(b))
    {
            // Point at t=0
        // auto vertex_start = V0.row(vi);
        // // Triangle at t = 0
        // auto face_vertex0_start = V0.row(F(fi, 0));
        // auto face_vertex1_start = V0.row(F(fi, 1));
        // auto face_vertex2_start = V0.row(F(fi, 2));
        // // Point at t=1
        // auto vertex_end = V1.row(vi);
        // // Triangle at t = 1
        // auto face_vertex0_end = V1.row(F(fi, 0));
        // auto face_vertex1_end = V1.row(F(fi, 1));
        // auto face_vertex2_end = V1.row(F(fi, 2));

    }
    else if (is_face(a) && is_vertex(b))
        return addData(b,a, queries);
    else if (is_edge(a) && is_edge(b))
    {
    //     // Edge 1 at t=0
    // auto edge0_vertex0_start = V0.row(E(ei, 0));
    // auto edge0_vertex1_start = V0.row(E(ei, 1));
    // // Edge 2 at t=0
    // auto edge1_vertex0_start = V0.row(E(ej, 0));
    // auto edge1_vertex1_start = V0.row(E(ej, 1));
    // // Edge 1 at t=1
    // auto edge0_vertex0_end = V1.row(E(ei, 0));
    // auto edge0_vertex1_end = V1.row(E(ei, 1));
    // // Edge 2 at t=1
    // auto edge1_vertex0_end = V1.row(E(ej, 0));
    // auto edge1_vertex1_end = V1.row(E(ej, 1));
    }
}


int main( int argc, char **argv )
{
    vector<char*> compare;

    const char* filet0 = argv[1];
    const char* filet1 = argv[2];
    
    vector<Aabb> boxes;
    parseMesh(filet0, filet1, boxes);
    int N = boxes.size();
    int nbox = 0;
    
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

    vector<unsigned long> overlaps;
    run_sweep(boxes.data(), N, nbox, overlaps);

    std::vector<std::array<std::array<float, 3>, 8>> queries;
    for (int i=0; i < overlaps.size() / 2; i++)
    {
        int aid = overlaps[2*i];
        int bid = overlaps[2*i+1];

        Aabb a = boxes[aid];
        Aabb b = boxes[bid];  
    
    }

    
    // for (auto i : compare)
    // {
    //     compare_mathematica(overlaps, i);
    // }
    // cout << endl;

    // Mesh --> Boxes --> Broadphase --> (Boxes[2] ->float/double[8]) --> Narrowphase
    // Go back to old code and make overlaps as pairs
    
    // std::array<std::array<Scalar, 3>, 8> V = substract_ccd(all_V, i);
    // bool expected_result = results[i * 8];
    // queries.push_back(V);
    // expect_list.push_back(expected_result);

    // https://github.com/dbelgrod/broad-phase-benchmark/blob/main/src/narrowphase/symbolic.cpp
    // each overlap has 4 vertices over t0, t1 -> 8
    // just get vids and check them against V0 and V1
    // the 3 is the x,y,z coord of the vertex

    // fill in the rest of gputi main() to finish integration
    // also fix that bug

    


}