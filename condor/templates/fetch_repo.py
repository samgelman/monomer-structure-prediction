import argparse

import urllib3
import shutil
from os.path import join


def fetch_repo(github_url, github_token, github_tag, out_dir):
    """ fetches the codebase from GitHub """

    url = join(github_url, "archive", "{}.tar.gz".format(github_tag))

    http = urllib3.PoolManager()

    if github_token is None:
        # todo: haven't actually tried this path for public repos, i've only used private repos
        headers = None
    else:
        headers = {"Authorization": "token {}".format(github_token)}

    response = http.request("GET", url, preload_content=False, headers=headers)

    # use static output filename to make transfer/unzipping easier (less need to fill in github_tag everywhere)
    save_fn = join(out_dir, "code.tar.gz")

    with open(save_fn, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

    # unclear whether this is needed or not, can't hurt though
    response.release_conn()
    response.close()


def main(args):
    fetch_repo(args.github_url, args.github_token, args.github_tag, out_dir=".")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars="@")

    parser.add_argument("--github_url",
                        type=str,
                        help="URL of the GitHub repository to download")

    parser.add_argument("--github_token",
                        type=str,
                        help="authorization token for private repository",
                        default=None)

    parser.add_argument("--github_tag",
                        type=str,
                        help="github tag specifying which version of code to retrieve for this run",
                        default="master")

    main(parser.parse_args())
