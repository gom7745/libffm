#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <cstdlib>

#include <algorithm>
#include "ffm.h"
#include "timer.h"

using namespace std;
using namespace ffm;

struct Option {
    string test_path, model_path, output_path,b_dir;
    ffm_int nr_threads = 1;
    ffm_double subratio = 1;
};

string basename(string path){

    const char *ptr = strrchr(&*path.begin(),'/');//get the last occurance of '/'
    if(!ptr)
        ptr = path.c_str();
    else
        ptr++;
    return string(ptr);
}

string predict_help(){
    return string(
            "usage: ffm-predict test_file model_file output_file\n"
            "\n"
            "options:\n"
            "-s <nr_threads>:set number of threads (default 1)\n"
            "-bd <bin data directory> set directory to the binary data\n"
			"-sr <subratio>: set subratio for validation\n"
    );
}

Option parse_option(int argc, char **argv){
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(predict_help());

    Option opt;

    ffm_int i =1;

    for(;i < argc; i++) {
        if (args[i].compare("-s") == 0) {
            if (i == argc - 1)
                throw invalid_argument("need to specify number of threads after -s");
            i++;
            opt.nr_threads = atoi(args[i].c_str());
            if (opt.nr_threads <= 0)
                throw invalid_argument("number of threads should be greater than zero");
        } else if(args[i].compare("-bd") == 0) {
			if(i == argc - 1)
                throw invalid_argument("need to specify validation path after -bd");
			i++;
			opt.b_dir = args[i];
        } else if(args[i].compare("-sr") == 0) {
            if(i == argc-1)
                throw invalid_argument("need to specify subratio after -sr");
            i++;
            opt.subratio = atof(args[i].c_str());
            if(opt.subratio < 0 || opt.subratio > 1)
                throw invalid_argument("subratio should be between zero and one");
		} else {
            break;
        }
    }

    if (i != argc - 3)
        throw invalid_argument("cannot parse command\n");

    opt.test_path = string(args[i++]);
    opt.model_path = string(args[i++]);
    opt.output_path = string(args[i++]);

    return opt;
}

void predict_on_disk(Option opt) {
    string te_bin_path = "./" + opt.b_dir + "/" + basename(opt.test_path) + ".bin";
    ffm_read_problem_to_disk(opt.test_path, te_bin_path);
    //ffm_model model =  ffm_load_model(opt.model_path);
    //ffm_model model =  ffm_load_model_plain_txt(opt.model_path);
    ffm_model model = ffm_load_model_map(opt.model_path);
	vector<ffm_float> va_labels, va_scores, va_orders;
    Timer timer;
    ffm_float va_logloss = ffm_predict_on_disk(te_bin_path,  model, va_scores, va_orders, va_labels, opt.subratio);
    ofstream f_out(opt.output_path);
	for(ffm_float y_bar: va_scores) {
		ffm_float expnyt = exp(-y_bar);
        f_out<<1/(1+expnyt)<<"\n";
	}
	ffm_float va_auc = cal_auc(va_orders, va_scores, va_labels);
    cout << "logloss = "<< fixed << setprecision(5)<< va_logloss << endl;
    cout << "auc = "<< fixed << setprecision(5)<< va_auc << endl;
	cout << "predict time = " << timer.get() << endl;
}

void predict(string test_path, string model_path, string output_path) {
    int const kMaxLineSize = 1000000;

    FILE *f_in = fopen(test_path.c_str(), "r");
    ofstream f_out(output_path);
    char line[kMaxLineSize];

    ffm_model model = ffm_load_model(model_path);

    ffm_double loss = 0;
    vector<ffm_node> x;
    ffm_int i = 0;

    for(; fgets(line, kMaxLineSize, f_in) != nullptr; i++) {
        x.clear();
        char *y_char = strtok(line, " \t");
        ffm_float y = (atoi(y_char)>0)? 1.0f : -1.0f;

        while(true) {
            char *field_char = strtok(nullptr,":");
            char *idx_char = strtok(nullptr,":");
            char *value_char = strtok(nullptr," \t");
            if(field_char == nullptr || *field_char == '\n')
                break;

            ffm_node N;
            N.f = atoi(field_char);
            N.j = atoi(idx_char);
            N.v = atof(value_char);

            x.push_back(N);
        }

        ffm_float y_bar = ffm_predict(x.data(), x.data()+x.size(), model);

        loss -= y==1? log(y_bar) : log(1-y_bar);

        f_out << y_bar << "\n";
    }

    loss /= i;

    cout << "logloss = " << fixed << setprecision(5) << loss << endl;

    fclose(f_in);
}

int main(int argc, char **argv) {
    Option option;
    try {
        option = parse_option(argc, argv);
    } catch(invalid_argument const &e) {
        cout << e.what() << endl;
        return 1;
    }

    //predict(option.test_path, option.model_path, option.output_path);
	predict_on_disk(option);

    return 0;
}
