#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mf.h"

using namespace std;
using namespace mf;

struct Option
{
    Option() : param(mf_get_default_param()), nr_folds(1), on_disk(false), do_cv(false) {}
    string tr_path, va_path, model_path;
    mf_parameter param;
    mf_int nr_folds;
    bool on_disk;
    bool do_cv;
};

string train_help()
{
    return string(
"usage: mf-train [options] training_set_file [model_file]\n"
"\n"
"options:\n"
"-l1 <lambda>,<lambda>: set L1-regularization parameters for P and Q (default 0)\n"
"  P and Q share the same lambda if only one lambda is specified\n"
"-l2 <lambda>,<lambda>: set L2-regularization parameters for P and Q (default 0.1)\n"
"  P and Q share the same lambda if only one lambda is specified\n"
"-f <loss>: set loss function (default 0)\n"
"  for real-valued matrix factorization\n"
"\t 0 -- squared error (L2-norm)\n"
"\t 1 -- absolute error (L1-norm)\n"
"\t 2 -- generalized KL-divergence\n"
"  for binary matrix factorization\n"
"\t 5 -- logarithmic loss\n"
"\t 6 -- squared hinge loss\n"
"\t 7 -- hinge loss\n"
"  for one-class matrix factorization\n"
"\t10 -- row-oriented pairwise logarithmic loss\n"
"\t11 -- column-oriented pairwise logarithmic loss\n"
"\t12 -- squared error (L2-norm)\n"
"-k <dimensions>: set number of dimensions (default 8)\n"
"-t <iter>: set number of iterations (default 20)\n"
"-r <eta>: set learning rate (default 0.1)\n"
"-a <alpha>: set coefficient of negative entries' loss (default 1)\n"
"-c <c>: set value of negative entries (default 0.0001). Positive entry is always 1.\n"
"-s <threads>: set number of threads (default 12)\n"
"-n <bins>: set number of bins (may be adjusted by LIBMF)\n"
"-p <path>: set path to the validation set\n"
"-v <fold>: set number of folds for cross validation\n"
"--quiet: quiet mode (no outputs)\n"
"--nmf: perform non-negative matrix factorization\n"
"--disk: perform disk-level training (will generate a buffer file)\n");
}

bool is_numerical(char *str)
{
    int c = 0;
    while(*str != '\0')
    {
        if(isdigit(*str))
            c++;
        str++;
    }
    return c > 0;
}

Option parse_option(int argc, char **argv)
{
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(train_help());

    Option option;

    mf_int i;
    for(i = 1; i < argc; i++)
    {
        if(args[i].compare("-l1") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify lambda after -l1");
            i++;

            char *pch = strtok(argv[i], ",");
            if(!is_numerical(pch))
                throw invalid_argument("regularization coefficient\
                                        should be a number");
            option.param.lambda_p1 = (mf_float)strtod(pch, NULL);
            option.param.lambda_q1 = (mf_float)strtod(pch, NULL);
            pch = strtok(NULL, ",");
            if(pch != NULL)
            {
                if(!is_numerical(pch))
                    throw invalid_argument("regularization coefficient\
                                            should be a number");
                option.param.lambda_q1 = (mf_float)strtod(pch, NULL);
            }
        }
        else if(args[i].compare("-l2") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify lambda after -l2");
            i++;

            char *pch = strtok(argv[i], ",");
            if(!is_numerical(pch))
                throw invalid_argument("regularization coefficient\
                                        should be a number");
            option.param.lambda_p2 = (mf_float)strtod(pch, NULL);
            option.param.lambda_q2 = (mf_float)strtod(pch, NULL);
            pch = strtok(NULL, ",");
            if(pch != NULL)
            {
                if(!is_numerical(pch))
                    throw invalid_argument("regularization coefficient\
                                            should be a number");
                option.param.lambda_q2 = (mf_float)strtod(pch, NULL);
            }
        }
        else if(args[i].compare("-k") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify number of factors\
                                        after -k");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-k should be followed by a number");
            option.param.k = atoi(argv[i]);
        }
        else if(args[i].compare("-t") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify number of iterations\
                                        after -t");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-i should be followed by a number");
            option.param.nr_iters = atoi(argv[i]);
        }
        else if(args[i].compare("-r") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify eta after -r");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-r should be followed by a number");
            option.param.eta = (mf_float)atof(argv[i]);
        }
        else if(args[i].compare("-s") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify number of threads\
                                        after -s");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-s should be followed by a number");
            option.param.nr_threads = atoi(argv[i]);
        }
        else if(args[i].compare("-a") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify negative weight\
                                        after -a");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-a should be followed by a number");
            option.param.alpha = static_cast<mf_float>(atof(argv[i]));
        }
        else if(args[i].compare("-c") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify negative rating\
                                        after -c");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-c should be followed by a number");

            if (argv[i][0] == '-')
                // Negative number starts with - but atof only recognize numbers.
                // Thus, we pass all but the first symbol to atof.
                option.param.c = -static_cast<mf_float>(atof(argv[i] + 1));
            else
                // Non-negative numbers such as 0 and 0.5 can be handled by atof.
                option.param.c = static_cast<mf_float>(atof(argv[i]));
        }
        else if(args[i].compare("-p") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -p");
            i++;

            option.va_path = string(args[i]);
        }
        else if(args[i].compare("-v") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify number of folds\
                                        after -v");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-v should be followed by a number");
            option.nr_folds = atoi(argv[i]);

            if(option.nr_folds < 2)
                throw invalid_argument("number of folds\
                                        must be greater than one");
            option.do_cv = true;
        }
        else if(args[i].compare("-f") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify loss function\
                                        after -f");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-f should be followed by a number");
            option.param.fun = atoi(argv[i]);
        }
        else if(args[i].compare("-n") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify the number of blocks\
                                        after -n");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-n should be followed by a number");
            option.param.nr_bins = atoi(argv[i]);
        }
        else if(args[i].compare("--nmf") == 0)
        {
            option.param.do_nmf = true;
        }
        else if(args[i].compare("--quiet") == 0)
        {
            option.param.quiet = true;
        }
        else if(args[i].compare("--disk") == 0)
        {
            option.on_disk = true;
        }
        else
        {
            break;
        }
    }

    if(option.nr_folds > 1 && !option.va_path.empty())
        throw invalid_argument("cannot specify both -p and -v");

    if(i >= argc)
        throw invalid_argument("training data not specified");

    option.tr_path = string(args[i++]);

    if(i < argc)
    {
        option.model_path = string(args[i]);
    }
    else if(i == argc)
    {
        const char *ptr = strrchr(&*option.tr_path.begin(), '/');
        if(!ptr)
            ptr = option.tr_path.c_str();
        else
            ++ptr;
        option.model_path = string(ptr) + ".model";
    }
    else
    {
        throw invalid_argument("invalid argument");
    }

    option.param.nr_bins = max(option.param.nr_bins,
                               2*option.param.nr_threads+1);
    option.param.copy_data = false;

    return option;
}

int main(int argc, char **argv)
{
    Option option;
    try
    {
        option = parse_option(argc, argv);
    }
    catch(invalid_argument &e)
    {
        cout << e.what() << endl;
        return 1;
    }

    mf_problem tr = {};
    mf_problem va = {};
    if(!option.on_disk)
    {
        try
        {
            tr = read_problem(option.tr_path);
            va = read_problem(option.va_path);
        }
        catch(runtime_error &e)
        {
            cout << e.what() << endl;
            return 1;
        }
    }

    if(option.do_cv)
    {
        if(!option.on_disk)
            mf_cross_validation(&tr, option.nr_folds, option.param);
        else
            mf_cross_validation_on_disk(
                option.tr_path.c_str(), option.nr_folds, option.param);
    }
    else
    {
        mf_model *model;
        if(!option.on_disk)
            model = mf_train_with_validation(&tr, &va, option.param);
        else
            model = mf_train_with_validation_on_disk(option.tr_path.c_str(),
                                                     option.va_path.c_str(),
                                                     option.param);

        // use the following function if you do not have a validation set

        // mf_model model =
        //     mf_train_with_validation(&tr, option.param);

        mf_int status = mf_save_model(model, option.model_path.c_str());

        mf_destroy_model(&model);

        if(status != 0)
        {
            cout << "cannot save model to " << option.model_path << endl;

            if(!option.on_disk)
            {
                delete[] tr.R;
                delete[] va.R;
            }

            return 1;
        }
    }

    if(!option.on_disk)
    {
        delete[] tr.R;
        delete[] va.R;
    }

    return 0;
}
