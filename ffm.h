#ifndef _LIBFFM_H
#define _LIBFFM_H

#include <fstream>
#include <string>
#include <unordered_map>
#include <cassert>
#include <fstream>

namespace ffm {

using namespace std;

typedef float ffm_float;
typedef double ffm_double;
typedef int ffm_int;
typedef long long ffm_long;

struct ffm_node {
    ffm_int f; // field index
    ffm_int j; // feature index
    ffm_float v; // value
};

struct ffm_model {
    ffm_int n; // number of features
    ffm_int m; // number of fields
    ffm_int k; // number of latent factors
    ffm_float *W = nullptr;
    unordered_map<ffm_int, ffm_float *> W_map;
    ffm_float *WL = nullptr;
    unordered_map<ffm_int, ffm_float *> WL_map;
    ffm_float *WB = nullptr;
    bool normalization;
    ~ffm_model();
    bool use_map = false;
};

struct ffm_parameter {
    ffm_float eta = 0.2; // learning rate
    ffm_float lambda = 0.00002; // regularization parameter
    ffm_int nr_iters = 15;
    ffm_int k = 4; // number of latent factors
    bool normalization = true;
    bool auto_stop = false;
	bool do_auc = false;
    string ws_model_path;
    bool use_map = false;
};

struct disk_problem_meta {
    ffm_int n = 0;
    ffm_int m = 0;
    ffm_int l = 0;
    ffm_int num_blocks = 0;
    ffm_long B_pos = 0;
	ffm_long nnz = 0;
    uint64_t hash1;
    uint64_t hash2;
};

struct problem_on_disk {
    disk_problem_meta meta;
    vector<ffm_float> Y;
    vector<ffm_float> R;
    vector<ffm_long> P;
    vector<ffm_node> X;
    vector<ffm_long> B;

    problem_on_disk(string path);

    int load_block(int block_index);

    bool is_empty() {
        return meta.l == 0;
    }

private:
    ifstream f;
};

ffm_float cal_auc(vector<ffm_float>& va_orders, vector<ffm_float>& va_scores, vector<ffm_float>& va_labels);

void ffm_problem_info(problem_on_disk &prob, string &path);

void ffm_read_problem_to_disk(string txt_path, string bin_path);

void ffm_save_model(ffm_model &model, string path);

void ffm_save_model_map(ffm_model &model, string path);

ffm_int ffm_save_model_plain_text(ffm_model& model, char const *path);

ffm_model ffm_load_model_map(string path);

ffm_int ffm_save_model_plain_text(ffm_model& model, char const *path);

ffm_model ffm_load_model(string path, ffm_int new_n=0);

ffm_model ffm_load_model_plain_txt(string path);

ffm_model ffm_train_on_disk(string Tr_path, string Va_path, ffm_parameter param);

ffm_float ffm_predict(ffm_node *begin, ffm_node *end, ffm_model &model);

ffm_float ffm_predict_on_disk(string te_path, ffm_model &model, vector<ffm_float>& va_scores, vector<ffm_float>& va_orders, vector<ffm_float>& va_labels, ffm_double subratio);

} // namespace ffm

#endif // _LIBFFM_H
