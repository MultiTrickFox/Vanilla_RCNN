import preprocessor
import parent


if __name__ == '__main__':
    parent.data_size = \
        preprocessor.preproc_bootstrap()
    parent.parent_bootstrap()