class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'Behavior_Data':
            # folder that contains class labels
            root_dir = 'data/'

            # Save preprocess data into output_dir
            output_dir = 'output/'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '/Path/to/hmdb-51'

            output_dir = '/path/to/VAR/hmdb51'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return './pretrained_model/c3d-pretrained.pth'