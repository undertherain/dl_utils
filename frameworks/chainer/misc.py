from chainer.serializers import NpzDeserializer


def load_model_from_trainer_npz(path, model):
    with np.load(path) as f:
        d = NpzDeserializer(f, path="updater/model:main/")
        d.load(model)
