
import argparse
import sys







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v:v.lower() == 'true')
    
    parser.add_argument(
        '--trainPath',
        type = str,
        help='Direcotory to load train data.')
    
    parser.add_argument(
        '--testPath',
        type = str,
        help='Direcotory to load test data.')
    
    args, unparsed = parser.parse_known_args()
    
    