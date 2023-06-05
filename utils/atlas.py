import re
import pandas as pd


def split_schafer_names(file_path):
	"""
	This function reads a CSV file containing Schafer parcellation data and extracts the hemisphere, network, and subregion
	information from the ROI names. The extracted information is added as new columns to the DataFrame and the updated
	DataFrame is returned.

	Args:
		file_path (str): The path to the CSV file containing Schafer parcellation data.

	Returns:
		pd.DataFrame: A DataFrame containing the original Schafer parcellation data with additional columns for
		              'Hemisphere', 'Network', and 'Subregion' extracted from the ROI names.
	"""
    schafer = pd.read_csv(file_path)
    regex_pattern = r'(RH|LH)_([A-Za-z]+)(?:_([A-Za-z]+))?(?:_[A-Za-z]+)*'
    def parse_roi_name(roi_name):
        match = re.search(regex_pattern, roi_name)
        if match:
            return match.group(1), match.group(2), match.group(3)
        return None, None, None

    schafer[['Hemisphere', 'Network', 'Subregion']] = schafer['ROI Name'].apply(parse_roi_name).apply(pd.Series)

    return schafer
