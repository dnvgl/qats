# -*- coding: utf-8 -*-
"""
Set new version tag, using git command line interface.

In order to obtain the current version (before setting new), git command 'git describe --tag' is used. This will
yield version tags like '1.1.0' or '1.1.0-1-g82869e0'.

Note: user is prompted for approval on command line before new tag is actually set.
"""
import argparse
import os
import sys
import textwrap

PREFIX = ""

def get_version_git(return_dev=False):
    global PREFIX
    # get version (will raise DistributionNotFound error if package is not found/installed)
    version_string = os.popen("git describe --tag").read().strip()

    if version_string.startswith("fatal"):
        # git failed, probably because not invoked at root of a git repo (.git not found)
        raise Exception("Not able to extract version using git")
    elif version_string.startswith("v"):
        PREFIX = "v"
        version_string = version_string.lstrip(PREFIX)

    version = version_string.split(".", maxsplit=2)
    assert len(version) == 3, f"Not able to interpret version string: {version_string}"

    # extract major, minor, micro
    major, minor, micro = version

    # interpret/correct micro
    if "-" in micro:
        # dev info included in micro
        micro, dev = micro.split("-", maxsplit=1)
    else:
        # pure major.minor.micro version, no dev part of tag
        dev = ""

    if return_dev:
        return major, minor, micro, dev
    else:
        return major, minor, micro


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    This function is an adjusted copy of: https://stackoverflow.com/a/3041990
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}

    assert default is None or default in valid, f"Invalid default option: {default}"

    if default is None:
        prompt = " (y/n) "
    elif default == "yes":
        prompt = " ([y]/n) "
    elif default == "no":
        prompt = " (y/[n]) "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def construct_version_string(major, minor, micro, dev=None, prefix=None):
    """
    Construct version tag: "[prefix]major.minor.micro" (or if 'dev' is specified: "major.minor.micro-dev").
    """
    if prefix is None:
        # if prefix is not specified, use the global variable
        prefix = PREFIX
    version_tag = f"{prefix}{major}.{minor}.{micro}"
    if dev:
        version_tag += f"-{dev}"
    return version_tag


def main():
    parser = argparse.ArgumentParser(
        description="Set new version tag using git command line interface. The tag is set by augmenting either "
                    "'major', 'minor' or 'micro' (specified by user) by 1.",
    )
    parser.add_argument("type", choices=("major", "minor", "micro"),
                        help="Which part of version tag to augment by one.")

    parser.add_argument("-m", "--message", default="", help="Commit message to include. Default is empty string")

    # parser.add_argument("--test", action="store_true",
    #                     help="Do not set tag, only print what tag would have been set.")

    args = parser.parse_args()

    # for debug and verification
    # print(f"get_version_setuptools() : {get_version_setuptools()}")
    # print(f"get_version_git()  : {get_version_git(return_dev=True)}")

    # extract current version tag
    major, minor, micro, dev = get_version_git(return_dev=True)
    current_version = construct_version_string(major, minor, micro, dev=dev)

    # determine new version
    if args.type == "major":
        # augment major by 1, reset minor and micro
        major = str(int(major) + 1)
        minor = "0"
        micro = "0"
    elif args.type == "minor":
        # augment minor by 1, reset minor
        minor = str(int(minor) + 1)
        micro = "0"
    else:
        # augment micro by 1
        micro = str(int(micro) + 1)
    # finally, reset dev in any case
    dev = None

    # construct new version tag
    new_version = construct_version_string(major, minor, micro, dev=dev)

    # ask user whether to conduct version tag update
    info_string = textwrap.dedent(f'''
        Current version tag : {current_version}
        New version tag     : {new_version}
        Commit message      : {args.message}

        Set to new version tag? ''')
    set_new_tag = query_yes_no(info_string, default="no")  # bool

    # act according to answer from user
    if set_new_tag:
        # ref: https://git-scm.com/book/en/v2/Git-Basics-Tagging
        _ = os.popen(f'git tag -a {new_version} -m "{args.message}"').read()

        sys.stdout.write(textwrap.dedent(f'''
            New version tag ({new_version}) has been set. 

            Verify the new tag by the following git command:
            git describe --tag

            To reverse (delete) the new tag, use the following git command:
            git tag -d {new_version}
            '''))

    else:
        sys.stdout.write("\nNew tag has NOT been set.")


if __name__ == "__main__":
    main()
