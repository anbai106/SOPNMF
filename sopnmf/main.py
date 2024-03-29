from sopnmf import cli

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE file"
__version__ = "0.0.4"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

def main():

    parser = cli.parse_command_line()
    args = parser.parse_args()
    args.func(args)

    
if __name__ == '__main__':
    main()
