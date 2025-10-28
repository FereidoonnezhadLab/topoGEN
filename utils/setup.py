import os
import datetime


def setup_output_directory(job_description, base_dir=r"D:\CollaGEN\TopoGEN\output"):
    """
    Creates a structured output directory based on the current date and job description.

    Parameters:
        job_description (str): Name of the job (used in folder naming).
        base_dir (str): Base directory for storing output files.

    Returns:
        str: The path to the output directory.
    """
    today = datetime.datetime.now()
    date_folder = today.strftime("%Y%m%d")
    output_directory = os.path.join(base_dir, date_folder, job_description)

    os.makedirs(output_directory, exist_ok=True)
    return output_directory


from matplotlib import rcParams, font_manager
font_path = 'D:/FONT/SourceSansPro-Regular.otf'
font_manager.fontManager.addfont(font_path)
rcParams['font.sans-serif'] = ['Source Sans Pro']
rcParams['font.family'] = 'sans-serif'
output_directory = None