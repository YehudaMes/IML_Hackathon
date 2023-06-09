import os
from hackathon_code.task_3 import plots_and_info
from hackathon_code.task_1 import run_task_1







if __name__ == "__main__":
    os.chdir("hackathon_code")
    path_to_task_1_input = "agoda_data/Agoda_Test_1.csv"
    path_to_task_2_input = ""
    try:
        #     run task 1
        run_task_1(path_to_task_1_input)
    except Exception() as e:
        print(e)
    try:
        #     run task 2
        pass
    except Exception() as e:
        print(e)
    try:
        plots_and_info()
        pass
    except Exception() as e:
        print(e)



    os.chdir("..")
    # Third Block - Question 3 information + plots

# main(PATH1,PATH2)
