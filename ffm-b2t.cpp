#include <vector>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <iomanip>
#include <unordered_set>
#include "ffm.h"

using namespace std;
using namespace ffm;

struct Option{
    string model_path,output_path;
};

string predict_help(){
    return string(
            "usage: ffm-b2t model_file\n"
    );
}

Option parse_option(int argc, char **argv){
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(predict_help());

    Option option;

    if(argc != 2)
        throw invalid_argument("cannot parse argument");

    option.model_path = string(args[1]);

    return option;
}

int main(int argc, char ** argv){
    Option option;
    try{
        option = parse_option(argc, argv);
    } catch (invalid_argument const &e){
        cout << e.what() << endl;
        return 1;
    }

    ffm_model model =  ffm_load_model(option.model_path);
    string model_txt_path = option.model_path + ".txt";
	ffm_save_model_plain_text(model, model_txt_path);

    return 0;
}
