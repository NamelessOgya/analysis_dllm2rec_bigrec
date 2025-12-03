import unittest
import os
import sys
import shutil

# Add BIGRec directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../BIGRec')))

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        # Create dummy data for testing
        self.test_data_dir = 'BIGRec/data/test_dataset'
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create dummy ratings.dat
        with open(os.path.join(self.test_data_dir, 'ratings.dat'), 'w') as f:
            # user_id::movie_id::rating::timestamp
            f.write("1::101::5::978300760\n")
            f.write("1::102::4::978302109\n")
            f.write("1::103::3::978301968\n")
            # Need at least 11 interactions for sequence length 10 + 1 target
            for i in range(104, 120):
                f.write(f"1::{i}::5::978300760\n")

        # Create dummy movies.dat
        with open(os.path.join(self.test_data_dir, 'movies.dat'), 'w', encoding='ISO-8859-1') as f:
            # movie_id::title::genres
            for i in range(101, 120):
                f.write(f"{i}::Movie {i}::Comedy\n")

    def tearDown(self):
        # Clean up
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)

    def test_data_files_existence(self):
        self.assertTrue(os.path.exists(os.path.join(self.test_data_dir, 'ratings.dat')))
        self.assertTrue(os.path.exists(os.path.join(self.test_data_dir, 'movies.dat')))

    # Add more tests here to verify logic if we can import the processing code
    # Since the processing code is in a notebook/script that runs globally, 
    # we might need to refactor it to be importable or run it via subprocess.

if __name__ == '__main__':
    unittest.main()
