#include <iostream>
#include <unistd.h>

#include <gpubf/groundtruth.cuh>
#include <gpubf/io.cuh>
#include <gpubf/util.cuh>

#include <ccdgpu/CType.cuh>
#include <ccdgpu/helper.cuh>
#include <ccdgpu/record.hpp>
#include <ccdgpu/timer.hpp>

// using namespace ccdgpu;
// using namespace ccd;

int main(int argc, char **argv) {
  vector<char *> compare;
  ccdgpu::Record r;

  char *filet0;
  char *filet1;

  filet0 = argv[1];
  if (is_file_exist(argv[2])) // CCD
    filet1 = argv[2];
  else // static CD
    filet1 = argv[1];

  vector<ccdgpu::Aabb> boxes;
  Eigen::MatrixXd vertices_t0;
  Eigen::MatrixXd vertices_t1;
  Eigen::MatrixXi faces;
  Eigen::MatrixXi edges;

  r.Start("parseMesh");
  parseMesh(filet0, filet1, vertices_t0, vertices_t1, faces, edges);
  r.Stop();

  json j;
  r.Start("constructBoxes", j);
  constructBoxes(vertices_t0, vertices_t1, edges, faces, boxes);
  r.Stop();
  int N = boxes.size();
  int nbox = 0;
  int parallel = 64;
  int devcount = 1;

  // std::copy(from_vector.begin(), from_vector.end(), to_vector.begin());

  int o;
  while ((o = getopt(argc, argv, "c:n:b:p:")) != -1) {
    switch (o) {
    case 'c':
      optind--;
      for (; optind < argc && *argv[optind] != '-'; optind++) {
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
    case 'p':
      parallel = atoi(optarg);
      break;
    }
  }
  vector<pair<int, int>> overlaps;
  vector<int> result_list;
  ccd::Scalar toi;

  bool use_ms = false;
  bool allow_zero_toi = true;
  ccd::Scalar min_distance = 0;

  compute_toi_strategy(vertices_t0, vertices_t1, edges, faces, 1e6, 0.0, 1e-6,
                       toi);

  // void compute_toi_strategy(const Eigen::MatrixXd &V0,
  //                           const Eigen::MatrixXd &V1, const Eigen::MatrixXi
  //                           &E, const Eigen::MatrixXi &F, int max_iter,
  //                           ccd::Scalar min_distance, ccd::Scalar tolerance,
  //                           ccd::Scalar &earliest_toi) {

  // run_ccd(boxes, vertices_t0, vertices_t1, r, N, nbox, parallel, devcount,
  //         overlaps, result_list, use_ms, allow_zero_toi, min_distance,
  //         toi);
  r.Print();

  // cout << "result_list " << result_list.size() << endl;
  // for (int i = 0; i < result_list.size(); i++)
  //   result_list[i] = 1;

  for (auto i : compare) {
    compare_mathematica(overlaps, result_list, i);
  }
  cout << endl;
}