import argparse
import base64 as b64
from io import BytesIO
from pickle import UnpicklingError

# noinspection PyUnresolvedReferences
from inference.inference_config import *
from inference.inference_utils import fit_model, apply_model


def load_tensor(path: str):
    try:
        return torch.load(path)
    except UnpicklingError:
        msg = "Can not unpickle %s" % path
        raise argparse.ArgumentTypeError(msg)
    except Exception:
        msg = "Can not open %s" % path
        raise argparse.ArgumentTypeError(msg)


def unpickle_tensor(s_data: str):
    try:
        b_data = b64.b64decode(s_data.encode('utf-8'))
        data = torch.load(BytesIO(b_data))
        return data
    except UnpicklingError:
        msg = "Can not unpickle '%s'" % s_data
        raise argparse.ArgumentTypeError(msg)
    except Exception as e:
        print(e)


def parse_args():
    parser = argparse.ArgumentParser(description='Apply model.')

    parser.add_argument('--model_name', type=str, choices=NAME2FOLDER.keys(), help='Name of the model', required=True)

    support_set_group = parser.add_mutually_exclusive_group()
    support_set_group.add_argument('--support_set_file', type=load_tensor, help='Path to support set tensor')
    support_set_group.add_argument('--support_set_pickle0', type=unpickle_tensor,
                                   help='Pickle encoded (protocol=0) support set numpy array')

    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument('--query_file', type=load_tensor, help='Path to query tensor')
    query_group.add_argument('--query_pickle0', type=unpickle_tensor,
                             help='Pickle encoded (protocol=0) query numpy array')

    # parser.add_argument('--image_resize', help='Size of scaled query (default = 84)', default=84)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    model_name = args.model_name

    query = args.query_file if args.query_file is not None else args.query_pickle0
    support = args.support_set_file if args.support_set_file is not None else args.support_set_pickle0

    if support is None:
        support = input()
        support = unpickle_tensor(support)

    if query is None:
        query = input()
        query = unpickle_tensor(query)

    fitted = fit_model(model_name, support)
    prediction = apply_model(fitted, query)
    buffer = BytesIO()
    torch.save(prediction, buffer)
    prediction_bytes = buffer.getvalue()
    buffer.close()
    print(b64.b64encode(prediction_bytes).decode('utf-8'))
