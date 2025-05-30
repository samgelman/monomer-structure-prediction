import argparse

def prep_run(args):
    pass


def main(args):
    prep_run(args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars="@")

    parser.add_argument("--run_name",
                        help="name for this condor run, used for log directory",
                        type=str,
                        default="unnamed")

    parser.add_argument("--output_dir",
                        help="directory to place output files",
                        type=str,
                        default="condor/runs")

    parser.add_argument("--executable_script",
                        help="path to the run.sh htcondor executable",
                        type=str,
                        default="condor/templates/run.sh")

    parser.add_argument("--submit_fn",
                        help="the path to the htcondor submit template",
                        type=str,
                        default="condor/templates/condor.sub")

    parser.add_argument("--args_template_fns",
                        type=str,
                        help="template files (.yaml) containing arguments to be expanded into multiple args files",
                        nargs="*")

    parser.add_argument("--args_per_job",
                        type=int,
                        help="how many argument files should each job be responsible for",
                        default=1)

    # https://stackoverflow.com/questions/27146262/create-variable-key-value-pairs-with-argparse-python
    parser.add_argument("--override_submit_file",
                        metavar="KEY=VALUE",
                        nargs="*",
                        help="set a number of key-value pairs to OVERRIDE the defaults in the template submit file. "
                             "if you are supplying this argument on a command line, do not put spaces before or "
                             "after the = sign, and place double quotes around the value if it has spaces. "
                             "if you are supplying this argument in an args file and calling python condor.py @file, "
                             "then you do not need to place double quotes around the value if it has spaces. "
                             "it is up to your judgement whether to use this argument or create a new template "
                             "submit file. i would use this arg for simple changes like needing more GPUs or "
                             "a different amount of memory. i would create a new template submit file for big "
                             "structural changes like restructuring data flow from submit file to execute nodes.")

    parser.add_argument("--github_tag",
                        type=str,
                        help="GitHub tag specifying which version of code to retrieve for this run",
                        default="master")

    parser.add_argument("--github_token",
                        type=str,
                        help="authorization token for private github repository. if None, script will check "
                             "environment variables for GITHUB_TOKEN and use that if available",
                        default=None)

    main(parser.parse_args())