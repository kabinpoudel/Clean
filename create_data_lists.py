from utils import create_data_lists
# Create lists for images in the training set and each of the test sets.
if __name__ == '__main__':
    create_data_lists(train_folders=['./data/train'],
                      test_folders=['./data/test'],
                      min_size=100,
                      output_folder='./data/output')