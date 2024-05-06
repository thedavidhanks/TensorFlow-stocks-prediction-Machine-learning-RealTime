from Tutorial.CONFIG import *


def print_to_file(string, log_file='log.txt', overwrite=False):
    setting = "w" if overwrite else "a"
    
    nl = "" if overwrite else "\n"
    string = nl + string
    
    # remove bcolors from string
    ansi_colors = [bcolors.HEADER, bcolors.OKBLUE, bcolors.OKCYAN, bcolors.OKGREEN, bcolors.WARNING, bcolors.FAIL, bcolors.ENDC, bcolors.BOLD, bcolors.UNDERLINE]
    
    for color in ansi_colors:
        string = string.replace(color, "")

    with open(log_file, setting) as f:
        f.write(string)

    print(string)