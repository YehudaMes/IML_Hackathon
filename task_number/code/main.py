import os
import sys

# from hackathon_code.task_3 import plots_and_info
from hackathon_code.task_1 import *
from hackathon_code.task_2 import run_task_2
from hackathon_code.task_3 import task_3_plots_and_info

if __name__ == "__main__":
    os.chdir("hackathon_code")
    path_to_task_1_input = "./agoda_data/Agoda_Test_1.csv"
    path_to_task_2_input = "./agoda_data/Agoda_Test_2.csv"
    try:
        run_task_1(path_to_task_1_input)
    except Exception() as e:
        print(e)
    try:
        run_task_2(path_to_task_2_input, train=False,
                   output_path="../../predictions/agoda_cost_of_cancellation.csv")
    except Exception() as e:
        print(e)
    try:
        task_3_plots_and_info()
    except Exception() as e:
        print(e)

    os.chdir("..")

# main(PATH1,PATH2)
