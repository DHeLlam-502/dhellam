import os
import subprocess
from dhellam.common.common import print_rank0

__all__=['initialize_dhellam_megatron']
def check_megatron_available(megapath):
    assert os.path.exists(megapath), f"{megapath} not exist"

    git_hash_cmd = "git rev-parse --short HEAD"
    git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
    
    def command_exists(cmd):
        try:
            subprocess.check_output(f'type {cmd}', shell=True)
            return True
        except subprocess.CalledProcessError:
            return False

    if command_exists('git'):
        try:
            result = subprocess.check_output(git_hash_cmd, cwd=megapath, shell=True)
            git_hash = result.decode('utf-8').strip()
            result = subprocess.check_output(git_branch_cmd, cwd=megapath, shell=True)
            git_branch = result.decode('utf-8').strip()
        except subprocess.CalledProcessError:
            git_hash = "unknown"
            git_branch = "unknown"
    else:
        git_hash = "unknown"
        git_branch = "unknown"
    print_rank0(f'**** Git info for Megatron: git_hash={git_hash} git_branch={git_branch} ****')

def _add_dhellam_args(parser):
    group = parser.add_argument_group(title='dhellam')
    group.add_argument('--dhellam', action='store_true', help='enable forward and backward overlap')
    group.add_argument('--schedule-config',type=str, default=None, help='schedule configuration files')  

    return parser


def initialize_dhellam_megatron(megapath=None, extra_args_provider=None):
    # check if available
    check_megatron_available(megapath)
    import sys
    sys.path.insert(0, megapath)
    from megatron.initialize import initialize_megatron
    from megatron import get_args
    from functools import partial

    def merge_args(extra_args_provider, parser):
        parser = _add_dhellam_args(parser)
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        return parser

    initialize_megatron(partial(merge_args, extra_args_provider))
    args = get_args()

    return args
